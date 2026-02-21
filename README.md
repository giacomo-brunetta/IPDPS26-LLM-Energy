# IPDPS26-LLM-Energy

Empirical reproducibility package for the IPDPS paper "Beyond Throughput: Performance and Energy Insights of LLM Inference Across AI Accelerators." This repo measures performance and energy for LLM inference across GPUs and dataflow accelerators using a consistent methodology and dataset.

**What this repo does**
- Runs single-GPU and multi-GPU inference experiments with vLLM on NVIDIA and AMD platforms.
- Collects latency, throughput, and energy metrics via vendor power tooling.
- Supports MoE parallelism comparisons (Tensor Parallel vs Expert Parallel).
- Includes templates for dataflow accelerator runs (Cerebras CS3, SambaNova SN40L).

**Repo layout**
- `Script/`: experiment drivers aligned with paper sections.
- `Utils/`: shared parsing, model loading, dataset handling, and results aggregation.
- `Nvidia/`: NVIDIA power profiler utilities.
- `AMD/`: AMD power profiler utilities.
- `Testbeds/`: dataflow accelerator client templates.
- `Dataset/`: dataset inputs used for the inference runs.
- Top-level runners: `test_text_dataset.py`, `data_parallel_test.py`, `synthetic_test.py`.

**Setup**
See the vLLM Docker deployment guide for base images and GPU runtime expectations:
```
https://docs.vllm.ai/en/stable/deployment/docker
```

Recommended Docker workflow (entrypoint set to bash, mount HF cache and project dir, pass HF token/cache):
```bash
export HF_TOKEN=YOUR_TOKEN_HERE
export HF_CACHE_DIR=$HOME/.cache/huggingface

docker run --gpus all -it --rm \
  --entrypoint /bin/bash \
  -e HF_TOKEN="$HF_TOKEN" \
  -e PLATFORM=["Nvidia"/'AMD"/"Intel"] \
  -e HF_HOME="$HF_CACHE_DIR" \
  -v "$HF_CACHE_DIR":"$HF_CACHE_DIR" \
  -v "$PWD":/workspace/LLM-Inference-Power \
  -w /workspace/LLM-Inference-Power \
  vllm/vllm-openai:latest
```

Alternatively, install vllm locally. This step might require compiling the library.
```
https://docs.vllm.ai/en/latest/getting_started/installation/gpu/
```

**Requirements**
- Global: `requirements.txt`
- NVIDIA-only: `Nvidia/requirements.txt`
- AMD-only: `AMD/requirements.txt`

```bash
pip install -r requirements.txt
pip install -r $PLATFORM/requirements.txt
```

**How to run**

Single-GPU runs (Section III-C1 / IV-A):
```bash
./Script/single_gpu.sh [cuda/rocm/xpu]
```

Multi-GPU scaling (Section III-C2 / IV-B):
```bash
NUM_GPUS=8 ./Script/multi_gpu_scaling.sh [cuda/rocm/xpu]
```

MoE parallelism comparison (Section III-C2 / IV-B):
```bash
NUM_GPUS=4 ./Script/moe_parallel.sh [cuda/rocm/xpu]
```


**Running Testbeds scripts**

Set up the inference engine of the CS3 or SN40L system.

Use the lightweight client workflow in `Testbeds/client.py`.

Install only the client dependencies:
```bash
pip install -r Testbeds/requirements.txt
```

Run (batch size is a CLI parameter):
```bash
python Testbeds/client.py --batch-size 8
```

Notes:
- Endpoint defaults to `http://localhost:8001/v1/chat/completions`.
- Replace placeholder Hugging Face `token=''` values in `Testbeds/client.py`.
- `--batch-size` controls the number of parallel worker processes.

**Outputs**
- Results are written by `Utils/results.py`. Check `Results/` for aggregated CSVs and intermediate logs.

**Notes**
- FP8 runs are enabled with `-dtype fp8`. When omitted, bf16 is used (or fp16 where appropriate).
- Dataflow scripts are templates and require vendor-specific client setup.
- For multi-GPU runs, `NUM_GPUS` must match available devices.
