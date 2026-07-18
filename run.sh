#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test.py \
  --datasets caltech101/dtd/eurosat/food101/oxford_pets \
  --backbone ViT-B/16 \
  --model-source both \
  --data-root /path/to/datasets
