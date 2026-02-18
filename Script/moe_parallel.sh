#!/usr/bin/env bash
set -euo pipefail

platform="${1:-cuda}"

# MoE parallelism experiment (Section III-C2 / IV-B, Fig. 6)
# Compare Tensor Parallelism (TP=4) vs Expert Parallelism (EP=4)
# on 4 GPUs, using FP8 precision.

NUM_GPUS="${NUM_GPUS:-4}"
TP_SIZE="${TP_SIZE:-4}"

if (( NUM_GPUS != 4 )); then
  echo "Expected NUM_GPUS=4 for this experiment" >&2
fi

models=(
  "meta-llama/Llama-4-Scout-17B-16E"
  "mistralai/Mixtral-8x22B-Instruct-v0.1"
  "Qwen/Qwen3-235B-A22B"
  "deepseek-ai/DeepSeek-R1-0528"
)

batch_sizes=(16 32 64 128)

for model in "${models[@]}"; do
  for bs in "${batch_sizes[@]}"; do
    # Tensor Parallelism
    python3 test_text_dataset.py --model_name "$model" -tp "$TP_SIZE" --batch_size "$bs" --platform "$platform" -dtype fp8

    # Expert Parallelism
    python3 test_text_dataset.py --model_name "$model" -tp "$TP_SIZE" --batch_size "$bs" --platform "$platform" -dtype fp8 -ep True
  done
done
