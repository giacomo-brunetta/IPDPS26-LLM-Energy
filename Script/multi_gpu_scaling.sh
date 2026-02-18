#!/usr/bin/env bash
set -euo pipefail

platform="${1:-cuda}"

# Multi-GPU scaling experiments (Section III-C2 / IV-B)
# Models: 14 LLMs from Table II
# Tensor parallel sizes: 1, 2, 4
# Batch size: 128 * TP
# Data parallel size: NUM_GPUS / TP
# Precisions: bf16 (default) and fp8

NUM_GPUS="${NUM_GPUS:-8}"

models=(
  "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen3-8B"
  "mistralai/Ministral-8B-Instruct-2410"
  "Qwen/Qwen3-14B"
  "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
  "Qwen/Qwen3-32B"
  "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
  "mistralai/Mixtral-8x7B-Instruct-v0.1"
  "meta-llama/Llama-3.3-70B-Instruct"
  "Qwen/Qwen2.5-72B-Instruct"
  "meta-llama/Llama-4-Scout-17B-16E"
  "Qwen/Qwen3-235B-A22B"
  "mistralai/Mixtral-8x22B-Instruct-v0.1"
  "deepseek-ai/DeepSeek-R1-0528"
)

for model in "${models[@]}"; do
  for tp_size in 1 2 4; do
    if (( NUM_GPUS % tp_size != 0 )); then
      echo "Skipping TP=$tp_size: NUM_GPUS=$NUM_GPUS not divisible" >&2
      continue
    fi
    dp_size=$(( NUM_GPUS / tp_size ))
    batch_size=$(( 128 * tp_size ))

    # bf16 (or fp16 on Intel, handled inside the Python runner if needed)
    python3 data_parallel_test.py --model_name "$model" -tp "$tp_size" -dp "$dp_size" --batch_size "$batch_size" --platform "$platform"

    # fp8
    python3 data_parallel_test.py --model_name "$model" -tp "$tp_size" -dp "$dp_size" --batch_size "$batch_size" --platform "$platform" -dtype fp8
  done
done
