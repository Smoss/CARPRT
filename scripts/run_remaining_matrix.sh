#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HOME="$repo_dir/.cache/huggingface"

common_args=(
  --data-root "$repo_dir/data"
  --device mps
  --batch-size 85
  --score-lambdas 0.01
)

result_complete() {
  .venv/bin/python -c '
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = set(sys.argv[2].split("/"))
if not path.is_file():
    raise SystemExit(1)
results = json.loads(path.read_text()).get("results", {})
complete = {
    dataset
    for dataset, payload in results.items()
    if any("elapsed_seconds" in source for source in payload.values())
}
raise SystemExit(0 if expected <= complete else 1)
' "$1" "$2"
}

main10='caltech101/dtd/eurosat/fgvc/food101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101'

if ! result_complete "$repo_dir/results/openai-vitb16-sun397.json" 'sun397'; then
  .venv/bin/python test.py \
    --datasets sun397 \
    --backbone ViT-B/16 \
    --model-source openai \
    --results-json "$repo_dir/results/openai-vitb16-sun397.json" \
    "${common_args[@]}"
fi

if ! result_complete "$repo_dir/results/openai-rn50-main10.json" "$main10"; then
  .venv/bin/python test.py \
    --datasets "$main10" \
    --backbone RN50 \
    --model-source openai \
    --results-json "$repo_dir/results/openai-rn50-main10.json" \
    "${common_args[@]}"
fi

if ! result_complete "$repo_dir/results/openclip-vitb16-laion2b-main10.json" "$main10"; then
  .venv/bin/python test.py \
    --datasets "$main10" \
    --backbone ViT-B/16 \
    --model-source openclip \
    --openclip-pretrained laion2b_s34b_b88k \
    --results-json "$repo_dir/results/openclip-vitb16-laion2b-main10.json" \
    "${common_args[@]}"
fi
