#!/usr/bin/env bash
# End-to-end pipeline check on a tiny slice of real data (CPU, about 15 minutes).
# Validates data loading, training, ensembling, the plagiarism engine and the API.
# The resulting accuracy is NOT representative of a full training run.
set -euo pipefail
cd "$(dirname "$0")/.."

plagai prepare-ai-data --out data/ai_small --sources hc3 mage raid wiki ai_pile \
    --per-source 1000 --ood-source wiki

plagai train-ai --data data/ai_small --model google/electra-small-discriminator \
    --out artifacts/ai-detector-smoke --max-length 256 --epochs 1 --batch-size 16 \
    --lr 1e-4 --eval-steps 100 --max-eval 400

# Binoculars with a tiny sibling pair (gpt2 / distilgpt2 share a tokenizer), only to exercise the code.
plagai fit-ensemble --model-dir artifacts/ai-detector-smoke --data data/ai_small \
    --max-length 256 --max-rows 400 --binoculars --observer gpt2 --performer distilgpt2

plagai train-paraphrase --base sentence-transformers/all-MiniLM-L6-v2 \
    --out artifacts/paraphrase-encoder-smoke --max-train 2000 --max-eval 1000

python scripts/demo_api.py artifacts/ai-detector-smoke
