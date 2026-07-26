"""Aggregate the completed CARPRT model/checkpoint comparison matrix."""

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
METHODS = ('MPE', 'WPE', 'CARPRT-lambda-0.01', 'CARPRT-release')
DATASET_ORDER = (
    'caltech101',
    'dtd',
    'eurosat',
    'fgvc',
    'food101',
    'oxford_flowers',
    'oxford_pets',
    'stanford_cars',
    'sun397',
    'ucf101',
)

PAPER = {
    'OpenAI ViT-B/16': {
        'caltech101': (93.09, 94.16),
        'dtd': (47.04, 48.90),
        'eurosat': (49.60, 55.56),
        'fgvc': (23.28, 24.49),
        'food101': (86.14, 86.31),
        'oxford_flowers': (66.60, 71.36),
        'oxford_pets': (82.38, 89.13),
        'stanford_cars': (65.93, 66.14),
        'sun397': (65.77, 66.93),
        'ucf101': (68.33, 70.41),
    },
    'OpenAI RN50': {
        'caltech101': (86.65, 88.46),
        'dtd': (40.89, 41.31),
        'eurosat': (30.65, 36.84),
        'fgvc': (16.11, 16.88),
        'food101': (76.15, 76.88),
        'oxford_flowers': (58.82, 65.56),
        'oxford_pets': (78.43, 85.69),
        'stanford_cars': (56.02, 56.44),
        'sun397': (59.71, 61.28),
        'ucf101': (61.53, 63.66),
    },
}


def load(path):
    return json.loads((RESULTS / path).read_text())['results']


def extract(source_results, source):
    return {
        dataset: payload[source]
        for dataset, payload in source_results.items()
        if source in payload
    }


def main():
    vit = extract(load('openai-vitb16-main9.json'), 'openai')
    vit.update(extract(load('openai-vitb16-sun397.json'), 'openai'))
    models = {
        'OpenAI ViT-B/16': vit,
        'OpenAI RN50': extract(load('openai-rn50-main10.json'), 'openai'),
        'OpenCLIP ViT-B/16 LAION-2B': extract(
            load('openclip-vitb16-laion2b-main10.json'),
            'openclip',
        ),
    }

    rows = []
    summaries = []
    for model_name, model_results in models.items():
        missing = set(DATASET_ORDER) - set(model_results)
        if missing:
            raise SystemExit(f'{model_name} is missing: {sorted(missing)}')

        for dataset in DATASET_ORDER:
            result = model_results[dataset]
            wpe = result['WPE']
            release = result['CARPRT-release']
            paper = PAPER.get(model_name, {}).get(dataset)
            row = {
                'model': model_name,
                'dataset': dataset,
                'total': wpe['total'],
                'seconds': result['elapsed_seconds'],
                'mpe_accuracy': result['MPE']['accuracy'],
                'wpe_accuracy': wpe['accuracy'],
                'appendix_d_accuracy': result['CARPRT-lambda-0.01']['accuracy'],
                'release_accuracy': release['accuracy'],
                'release_delta_pp': release['accuracy'] - wpe['accuracy'],
                'release_delta_correct': release['correct'] - wpe['correct'],
                'paper_wpe_accuracy': paper[0] if paper else None,
                'paper_release_accuracy': paper[1] if paper else None,
            }
            rows.append(row)

        model_rows = [row for row in rows if row['model'] == model_name]
        summaries.append(
            {
                'model': model_name,
                'datasets': len(model_rows),
                'images': sum(row['total'] for row in model_rows),
                'seconds': sum(row['seconds'] for row in model_rows),
                'mpe_macro': mean(row['mpe_accuracy'] for row in model_rows),
                'wpe_macro': mean(row['wpe_accuracy'] for row in model_rows),
                'appendix_d_macro': mean(row['appendix_d_accuracy'] for row in model_rows),
                'release_macro': mean(row['release_accuracy'] for row in model_rows),
                'release_delta_macro_pp': mean(row['release_delta_pp'] for row in model_rows),
                'positive_datasets': sum(row['release_delta_pp'] > 0 for row in model_rows),
                'negative_datasets': sum(row['release_delta_pp'] < 0 for row in model_rows),
                'net_correct': sum(row['release_delta_correct'] for row in model_rows),
            }
        )

    payload = {
        'scope': {
            'datasets': list(DATASET_ORDER),
            'dataset_count': len(DATASET_ORDER),
            'image_count': sum(row['total'] for row in rows[: len(DATASET_ORDER)]),
            'excluded': {
                'imagenet': 'Licensed ImageNet validation data was not present locally.',
            },
        },
        'summaries': summaries,
        'rows': rows,
    }
    (RESULTS / 'full-matrix-summary.json').write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n'
    )

    with (RESULTS / 'full-matrix.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(payload['summaries'], indent=2))


if __name__ == '__main__':
    main()
