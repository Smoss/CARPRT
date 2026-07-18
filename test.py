import argparse
import gc

import torch
import torch.nn.functional as F
from tabulate import tabulate

from models import OPENCLIP_PRETRAINED, load_model, resolve_device

from utils import (
    build_test_data_loader,
    clip_classifier,
    cls_acc,
    get_clip_logits,
    get_res_logits,
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
    return parser.parse_args()


def get_matrix(logits, num_class, num_prompt):
    max_values, max_indices = torch.max(logits, dim=1)
    max_values = max_values.float()

    sum_matrix = torch.zeros((num_prompt, num_class), dtype=torch.float32, device=logits.device)
    for p in range(num_prompt):
        sum_matrix[p].scatter_add_(0, max_indices[p], max_values[p])

    counts_matrix = torch.zeros((num_prompt, num_class), dtype=torch.long, device=logits.device)
    for p in range(num_prompt):
        counts_matrix[p].scatter_add_(0, max_indices[p], torch.ones_like(max_values[p], dtype=torch.long))

    return sum_matrix, counts_matrix


def run_test_carprt(loader, clip_model, text_feature, temp):
    with torch.no_grad():
        num_prompt, num_class, _ = text_feature.shape
        device = text_feature.device

        carprt_weight = torch.zeros((num_prompt, num_class), dtype=torch.float32, device=device)
        carprt_count = torch.zeros((num_prompt, num_class), dtype=torch.long, device=device)

        for images, _target in loader:
            images = images.to(device)
            logits = get_clip_logits(images, clip_model, text_feature)

            sum_matrix, count_matrix = get_matrix(logits, num_class, num_prompt)

            carprt_weight += sum_matrix
            carprt_count += count_matrix

        carprt_safe_count = torch.where(carprt_count == 0, 1, carprt_count)
        carprt_weight = carprt_weight / carprt_safe_count
        carprt_weight = F.softmax(carprt_weight / temp, dim=0)

    accuracies_carprt = []
    for images, target in loader:
        images = images.to(device)
        target = target.to(device)
        logits = get_res_logits(images, clip_model, text_feature, carprt_weight)
        acc = cls_acc(logits, target)
        accuracies_carprt.append(acc)

    return sum(accuracies_carprt) / len(accuracies_carprt)


def main():
    args = get_arguments()
    _ = args.config

    if args.model_source in {'openclip', 'both'} and args.backbone != 'ViT-B/16':
        raise SystemExit('--model-source openclip/both requires --backbone ViT-B/16')
    if args.batch_size < 1:
        raise SystemExit('--batch-size must be positive')

    device = resolve_device(args.device)
    model_sources = ['openai', 'openclip'] if args.model_source == 'both' else [args.model_source]
    print(f'Using device: {device}')

    datasets = args.datasets.split('/')
    results = {dataset_name: {} for dataset_name in datasets}

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
            acc_carprt = run_test_carprt(
                test_loader,
                loaded.model,
                text_feature,
                args.temp,
            )
            results[dataset_name][source] = acc_carprt

            print("---- CARPRT's test accuracy: {:.2f}. ----\n".format(acc_carprt))
            del text_feature, test_loader

        del loaded
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    headers = ['Dataset'] + model_sources
    if model_sources == ['openai', 'openclip']:
        headers.append('OpenCLIP - OpenAI')

    rows = []
    for dataset_name in datasets:
        row = [dataset_name]
        row.extend(f'{results[dataset_name][source]:.2f}' for source in model_sources)
        if model_sources == ['openai', 'openclip']:
            delta = results[dataset_name]['openclip'] - results[dataset_name]['openai']
            row.append(f'{delta:+.2f}')
        rows.append(row)

    print('\nCARPRT comparison summary')
    print(tabulate(rows, headers=headers, tablefmt='github'))


if __name__ == "__main__":
    main()
