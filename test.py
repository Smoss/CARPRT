import argparse
import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tabulate import tabulate

from models import OPENCLIP_PRETRAINED, load_model, resolve_device

from utils import (
    build_test_data_loader,
    clip_classifier,
)


def get_arguments():
    parser = argparse.ArgumentParser(description="CARPRT test-time prompt reweighting (CLIP).")
    parser.add_argument(
        '--config',
        dest='config',
        default=None,
        help='Optional; reserved for YAML configs (not used by this script).',
    )
    parser.add_argument('--datasets', dest='datasets', type=str, required=True, help="Dataset id(s), slash-separated, e.g. 'caltech101' or 'I/A'.")
    parser.add_argument(
        '--data-root',
        dest='data_root',
        type=str,
        default='/projects/datasets',
        help='Root directory of benchmark datasets.',
    )
    parser.add_argument(
        '--backbone',
        dest='backbone',
        type=str,
        choices=['RN50', 'ViT-B/16'],
        required=True,
        help='CLIP backbone.',
    )
    parser.add_argument(
        '--model-source',
        choices=['openai', 'openclip', 'both'],
        default='both',
        help='VLM implementation to evaluate (default: both comparison models).',
    )
    parser.add_argument(
        '--openclip-pretrained',
        default=OPENCLIP_PRETRAINED,
        help='OpenCLIP pretrained checkpoint tag.',
    )
    parser.add_argument(
        '--device',
        default='auto',
        help='PyTorch device, e.g. cuda, mps, cpu, or auto (default).',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=512,
        help='Evaluation batch size (reduce this if GPU memory is insufficient).',
    )
    parser.add_argument('--temp', dest='temp', type=float, default=1.0, help='Temperature for softmax over prompt weights.')
    parser.add_argument(
        '--results-json',
        default=None,
        help='Optional path for incrementally saved machine-readable results.',
    )
    parser.add_argument(
        '--score-lambdas',
        default='0.01',
        help=(
            'Comma-separated CARPRT score-normalization scales from Appendix D.2 '
            '(default: 0.01). Multiple values are evaluated from cached features.'
        ),
    )
    return parser.parse_args()


def get_matrix(logits, num_class, num_prompt):
    max_values, max_indices = torch.max(logits, dim=1)
    max_values = max_values.float()

    sum_matrix = torch.zeros((num_prompt, num_class), dtype=torch.float32, device=logits.device)
    sum_matrix.scatter_add_(1, max_indices, max_values)

    counts_matrix = torch.zeros((num_prompt, num_class), dtype=torch.long, device=logits.device)
    counts_matrix.scatter_add_(
        1,
        max_indices,
        torch.ones_like(max_values, dtype=torch.long),
    )

    return sum_matrix, counts_matrix


def _encode_images(images, clip_model, device):
    if isinstance(images, list):
        images = torch.cat(images, dim=0)
    model_dtype = next(clip_model.parameters()).dtype
    images = images.to(device=device, dtype=model_dtype)
    image_features = clip_model.encode_image(images)
    return image_features / image_features.norm(dim=-1, keepdim=True)


def run_test_methods(loader, clip_model, text_feature, temp, score_lambdas):
    """Evaluate baselines plus paper/released CARPRT while sharing encoder passes."""
    with torch.no_grad():
        num_prompt, num_class, _ = text_feature.shape
        device = text_feature.device

        cached_features = []
        cached_targets = []
        for images, target in loader:
            cached_features.append(_encode_images(images, clip_model, device).cpu())
            cached_targets.append(target.cpu())
        cached_features = torch.cat(cached_features)
        cached_targets = torch.cat(cached_targets)
        scoring_batch_size = getattr(loader, 'batch_size', None) or len(cached_features)

        carprt_paper_weights = {
            score_lambda: torch.zeros(
                (num_prompt, num_class),
                dtype=torch.float32,
                device=device,
            )
            for score_lambda in score_lambdas
        }
        carprt_release_weight = torch.zeros((num_prompt, num_class), dtype=torch.float32, device=device)
        carprt_count = torch.zeros((num_prompt, num_class), dtype=torch.long, device=device)
        wpe_score_sum = torch.zeros(num_prompt, dtype=torch.float32, device=device)
        logit_scale = clip_model.logit_scale.exp().detach().float()

        for start in range(0, len(cached_features), scoring_batch_size):
            image_features = cached_features[start : start + scoring_batch_size].to(device)
            logits = torch.einsum('pcd,nd -> pcn', text_feature, image_features)

            release_sum_matrix, count_matrix = get_matrix(logits, num_class, num_prompt)

            # Appendix D.2 defines s=softmax(cosine/lambda). The public release
            # omits both this normalization and lambda, aggregating raw logits.
            cosine_logits = logits.float() / logit_scale
            for score_lambda, paper_weight in carprt_paper_weights.items():
                paper_scores = F.softmax(cosine_logits / score_lambda, dim=1)
                paper_sum_matrix, _ = get_matrix(paper_scores, num_class, num_prompt)
                paper_weight += paper_sum_matrix
            carprt_release_weight += release_sum_matrix
            carprt_count += count_matrix
            wpe_score_sum += logits.max(dim=1).values.float().sum(dim=1)

        carprt_safe_count = torch.where(carprt_count == 0, 1, carprt_count)
        carprt_paper_weights = {
            score_lambda: F.softmax(
                (paper_weight / carprt_safe_count) / temp,
                dim=0,
            )
            for score_lambda, paper_weight in carprt_paper_weights.items()
        }
        carprt_release_weight = F.softmax(
            (carprt_release_weight / carprt_safe_count) / temp,
            dim=0,
        )

        wpe_weight = F.softmax((wpe_score_sum / len(cached_features)) / temp, dim=0)
        wpe_weight = wpe_weight[:, None].expand(-1, num_class)
        mpe_weight = torch.full_like(carprt_release_weight, 1.0 / num_prompt)

        weights = {
            'MPE': mpe_weight,
            'WPE': wpe_weight,
            **{
                f'CARPRT-lambda-{score_lambda:g}': paper_weight
                for score_lambda, paper_weight in carprt_paper_weights.items()
            },
            'CARPRT-release': carprt_release_weight,
        }
        classifiers = {
            method: torch.einsum('pc,pcd -> cd', method_weight.to(text_feature.dtype), text_feature)
            for method, method_weight in weights.items()
        }

        correct = {method: 0 for method in classifiers}
        for start in range(0, len(cached_features), scoring_batch_size):
            image_features = cached_features[start : start + scoring_batch_size].to(device)
            target = cached_targets[start : start + scoring_batch_size].to(device)
            for method, classifier in classifiers.items():
                predictions = (image_features @ classifier.t()).argmax(dim=1)
                correct[method] += int(predictions.eq(target).sum().item())
        total = len(cached_targets)

    return {
        method: {
            'accuracy': 100.0 * method_correct / total,
            'correct': method_correct,
            'total': total,
        }
        for method, method_correct in correct.items()
    }


def save_results(path, payload):
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + '.tmp')
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    temporary_path.replace(output_path)


def main():
    args = get_arguments()
    _ = args.config

    if args.model_source in {'openclip', 'both'} and args.backbone != 'ViT-B/16':
        raise SystemExit('--model-source openclip/both requires --backbone ViT-B/16')
    if args.batch_size < 1:
        raise SystemExit('--batch-size must be positive')
    try:
        score_lambdas = [float(value) for value in args.score_lambdas.split(',')]
    except ValueError as error:
        raise SystemExit('--score-lambdas must contain comma-separated numbers') from error
    if not score_lambdas or any(value <= 0 for value in score_lambdas):
        raise SystemExit('--score-lambdas values must be positive')

    device = resolve_device(args.device)
    model_sources = ['openai', 'openclip'] if args.model_source == 'both' else [args.model_source]
    print(f'Using device: {device}')

    datasets = args.datasets.split('/')
    results = {dataset_name: {} for dataset_name in datasets}
    output = {
        'backbone': args.backbone,
        'batch_size': args.batch_size,
        'device': str(device),
        'model_source': args.model_source,
        'openclip_pretrained': args.openclip_pretrained,
        'score_lambdas': score_lambdas,
        'temperature': args.temp,
        'results': results,
    }

    # Load and release models one at a time so the comparison does not require
    # enough memory to hold both VLMs simultaneously.
    for source in model_sources:
        loaded = load_model(
            source,
            args.backbone,
            device,
            openclip_pretrained=args.openclip_pretrained,
        )
        print(f'\nEvaluating {loaded.name}')

        for dataset_name in datasets:
            print(f"Processing {dataset_name} dataset.")
            started_at = time.perf_counter()
            test_loader, classnames, template = build_test_data_loader(
                dataset_name,
                args.data_root,
                loaded.preprocess,
                batch_size=args.batch_size,
            )

            text_feature = clip_classifier(
                classnames,
                template,
                loaded.model,
                loaded.tokenizer,
                device,
            )
            metrics = run_test_methods(
                test_loader,
                loaded.model,
                text_feature,
                args.temp,
                score_lambdas,
            )
            metrics['elapsed_seconds'] = time.perf_counter() - started_at
            metrics['model_name'] = loaded.name
            results[dataset_name][source] = metrics
            save_results(args.results_json, output)

            print(f"---- {dataset_name} / {source} / {metrics['elapsed_seconds']:.1f}s ----")
            for method, method_metrics in metrics.items():
                if not isinstance(method_metrics, dict) or 'accuracy' not in method_metrics:
                    continue
                gain = method_metrics['accuracy'] - metrics['WPE']['accuracy']
                print(f"{method}: {method_metrics['accuracy']:.2f} ({gain:+.2f} pp vs WPE)")
            print()
            del text_feature, test_loader

        del loaded
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    headers = ['Dataset', 'Source', 'Method', 'Accuracy', 'vs WPE', 'Correct', 'Total', 'Seconds']
    rows = []
    for dataset_name in datasets:
        for source in model_sources:
            metrics = results[dataset_name][source]
            for method, method_metrics in metrics.items():
                if not isinstance(method_metrics, dict) or 'accuracy' not in method_metrics:
                    continue
                rows.append([
                    dataset_name,
                    source,
                    method,
                    f"{method_metrics['accuracy']:.2f}",
                    f"{method_metrics['accuracy'] - metrics['WPE']['accuracy']:+.2f}",
                    method_metrics['correct'],
                    method_metrics['total'],
                    f"{metrics['elapsed_seconds']:.1f}",
                ])

    print('\nPrompt-ensembling comparison summary')
    print(tabulate(rows, headers=headers, tablefmt='github'))


if __name__ == "__main__":
    main()
