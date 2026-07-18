# CARPRT: Class-Aware Zero-Shot Prompt Reweighting for Black-Box Vision-Language Models

> Official implementation of **CARPRT (ICLR 2026)** — a training-free, black-box method for class-aware prompt reweighting in vision-language models.

> **Comparison fork:** this branch adds a side-by-side evaluation using the
> original OpenAI CLIP ViT-B/16 checkpoint and an independently trained
> **OpenCLIP ViT-B/16 checkpoint (`laion2b_s34b_b88k`)**. The CARPRT algorithm,
> prompts, datasets, and temperature are shared between the two runs.

[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/pdf?id=AScQDQqVXY)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)


---

**CARPRT** improves zero-shot classification in CLIP-like models by estimating **class-aware prompt weights** from unlabeled data.
Unlike prior methods that assign a single weight to each prompt across all classes, CARPRT models **class-dependent prompt relevance**, leading to more accurate predictions in a fully **training-free, black-box** setting.

---

![CARPRT overview](figs/overview.png)

> CARPRT estimates class-specific prompt weights from unlabeled images and improves zero-shot prediction via reweighted prompt ensembling.

---

## Method overview

CARPRT estimates prompt importance separately for each class using only unlabeled test images and similarity scores from a pre-trained vision–language model.

The procedure consists of two stages:

1. **Score collection**  
   Compute image–text similarity for all prompt–class pairs.

2. **Weight estimation and inference**  
   Aggregate class-conditional scores to obtain class-specific prompt weights, then reweight prompts during prediction.

---

## Installation
We recommend a conda environment with **Python 3.9+** and a **CUDA** build of
PyTorch. This comparison fork also supports explicit `mps` and `cpu` devices,
although CUDA will be substantially faster for the full benchmark.

```bash
# 1) Create and activate environment 
conda create -y -n carprt python=3.10
conda activate carprt
# 2) Clone and enter the repository
git clone <YOUR_REPO_URL>.git
cd carpr
# 3) Install dependencies
pip install -r requirements.txt
```

---

### Data preparation

Place datasets under **`--data-root`** (default in code: `/projects/datasets` — change to your path). Each dataset loader under `datasets/*.py` defines the expected subdirectory name and layout (e.g. Caltech-101 expects `caltech-101/101_ObjectCategories/` and the Zhou split JSON beside it).

**First-time download:** Several builders use **`gdown`** to fetch archives from Google Drive when paths are missing (`datasets/utils.py`).

---

### Evaluation

**Main entry point:** `test.py`

| Argument | Required | Description |
|----------|----------|-------------|
| `--datasets` | Yes | One or more dataset ids, **slash-separated** (e.g. `caltech101/dtd` or `I/A`). |
| `--backbone` | Yes | CLIP backbone: `RN50` or `ViT-B/16`. OpenCLIP comparison runs require `ViT-B/16`. |
| `--model-source` | No | `openai`, `openclip`, or `both` (default). `both` runs the models sequentially and prints a comparison table. |
| `--openclip-pretrained` | No | OpenCLIP checkpoint tag (default: `laion2b_s34b_b88k`). |
| `--data-root` | No | Root folder for all benchmarks (see default in `test.py`). |
| `--device` | No | PyTorch device: `auto` (default), `cuda`, `mps`, or `cpu`. |
| `--batch-size` | No | Evaluation batch size (default: `512`; reduce if memory is insufficient). |
| `--temp` | No | Temperature for softmax over prompt weights (default `1.0`). |
| `--config` | No | Reserved; unused (kept for backward-compatible command lines). |

**Supported dataset ids** (must match `utils.build_test_data_loader`):

| Id | Benchmark |
|----|-----------|
| `I` | ImageNet |
| `A` | ImageNet-A |
| `V` | ImageNet-V2 |
| `R` | ImageNet-R |
| `S` | ImageNet-Sketch |
| `caltech101`, `dtd`, `eurosat`, `fgvc`, `food101`, `oxford_flowers`, `oxford_pets`, `stanford_cars`, `sun397`, `ucf101` | Fine-grained / generic recognition |
| `cifar10`, `imcifar10`, `cifar100`, `imcifar100` | CIFAR variants |

**Minimal example:**

```bash
CUDA_VISIBLE_DEVICES=0 \
python test.py \
  --datasets caltech101 \
  --backbone ViT-B/16 \
  --data-root /path/to/datasets
```

This now runs both comparison checkpoints because `--model-source` defaults to
`both`. To reproduce the original OpenAI-only behavior, add
`--model-source openai`.

**Batch evaluation** (several sets in one run):

```bash
python test.py \
  --datasets caltech101/dtd/eurosat/food101/oxford_pets \
  --backbone ViT-B/16 \
  --model-source both \
  --data-root /path/to/datasets \
  --temp 1.0
```

### OpenCLIP comparison design

The comparison uses:

| Condition | Model | Pretraining |
|-----------|-------|-------------|
| Reference | OpenAI CLIP `ViT-B/16` | OpenAI checkpoint used by the paper |
| Comparison | OpenCLIP `ViT-B-16` | `laion2b_s34b_b88k` (LAION-2B) |

The architecture is held as close as practical while the training data,
training implementation, and learned checkpoint change. Both conditions use
the same dataset loaders, class names, 247 prompt templates, CARPRT weighting
code, and `--temp` value. Models are loaded one at a time to limit peak GPU
memory.

At the end of a `--model-source both` run, the script prints one row per dataset
with both CARPRT accuracies and `OpenCLIP - OpenAI`. This delta compares the
resulting classifiers; it is **not** the CARPRT-vs-WPE method gain. To test
whether CARPRT's improvement itself is model-independent, evaluate the same
baseline under both checkpoints as a separate follow-up.

OpenCLIP and OpenAI CLIP can learn different similarity scales. Keep
`--temp` fixed for the primary controlled comparison, then repeat a small
temperature sensitivity sweep if the conclusion depends on score calibration.

Examples for running just one condition:

```bash
# Original paper checkpoint
python test.py \
  --datasets caltech101 \
  --backbone ViT-B/16 \
  --model-source openai \
  --data-root /path/to/datasets

# Independent LAION-2B checkpoint
python test.py \
  --datasets caltech101 \
  --backbone ViT-B/16 \
  --model-source openclip \
  --openclip-pretrained laion2b_s34b_b88k \
  --data-root /path/to/datasets
```

---

### Citation

If you use this code or the CARPRT method, please cite **the CARPRT paper** (replace with the official BibTeX once available).

```bibtex
@inproceedings{dong2026carprt,
  title     = {CARPRT: Class-Aware Zero-Shot Prompt Reweighting for Black-Box Vision-Language Models},
  author    = {Dong, Ruijiang and Ye, Zesheng and Qi, Jianzhong and Feng, Lei and Liu, Feng and Niu, Gang and Sugiyama, Masashi},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```

---
