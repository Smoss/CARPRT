"""Extract the exact Zhou SUN397 test split from Hugging Face Parquet shards."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parquet_dir', type=Path)
    parser.add_argument('split_json', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument(
        '--allow-partial',
        action='store_true',
        help='Extract the matching subset without failing when other split images are absent.',
    )
    args = parser.parse_args()

    split = json.loads(args.split_json.read_text())
    expected = {row[0] for row in split['test']}
    written = set()

    for shard in sorted(args.parquet_dir.glob('*.parquet')):
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(columns=['image', 'image_file']):
            for row in batch.to_pylist():
                relative_path = row['image_file']
                if relative_path not in expected:
                    continue
                output_path = args.output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(row['image']['bytes'])
                written.add(relative_path)

    missing = expected - written
    if missing and not args.allow_partial:
        sample = ', '.join(sorted(missing)[:5])
        raise SystemExit(f'Missing {len(missing)} expected images; examples: {sample}')

    print(
        f'Extracted {len(written)} exact Zhou SUN397 test images'
        f' ({len(missing)} expected images not present in these shards).'
    )


if __name__ == '__main__':
    main()
