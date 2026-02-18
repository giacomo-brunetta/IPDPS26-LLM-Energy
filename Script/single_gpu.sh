#!/usr/bin/env bash
set -euo pipefail

platform="${1:-cuda}"

# Single-GPU experiments (Section III-C1 / IV-A)
# Models: Llama 3.1 8B, Qwen3 14B, Mistral Small 3.1 24B, Qwen3 32B
# Batch sizes: 16, 32, 64, 128
# Precisions: bf16 (default) and fp8

models=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen3-14B"
  "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  "Qwen/Qwen3-32B"
)

batch_sizes=(16 32 64 128)

# Default: single GPU
TP_SIZE="${TP_SIZE:-1}"

for model in "${models[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    # bf16 (or fp16 on Intel, handled inside the Python runner if needed)
    python3 test_text_dataset.py --model_name "$model" -tp "$TP_SIZE" --batch_size "$bs" --platform "$platform"

    # fp8
    python3 test_text_dataset.py --model_name "$model" -tp "$TP_SIZE" --batch_size "$bs" --platform "$platform" -dtype fp8
  done
done
