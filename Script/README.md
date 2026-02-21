Script folder cleaned and aligned with the experiments described in the paper.

Scripts
- `Script/single_gpu.sh`: single-GPU runs (Section III-C1 / IV-A)
- `Script/multi_gpu_scaling.sh`: multi-GPU scaling with DP/TP (Section III-C2 / IV-B)
- `Script/moe_parallel.sh`: MoE TP vs EP comparison on 4 GPUs (Section III-C2 / IV-B)
- `Script/dataflow_cerebras.sh`: Cerebras CS3 template (Section III-C3 / IV-C)
- `Script/dataflow_sambanova.sh`: SambaNova SN40L template (Section III-C3 / IV-C)
- `Script/dataflow_client.sh`: runs `Testbeds/client.py` with configurable request parallelism (`batch_size`) for dataflow endpoint testing (Section III-C3 / IV-C)

Notes
- All GPU scripts use the same Python entry points already used in this repo: `test_text_dataset.py` and `data_parallel_test.py`.
- `-dtype fp8` is used for FP8 runs; bf16 is the default when `-dtype` is omitted.
- Use `platform` as the first positional argument (default `cuda`).
- Multi-GPU scripts read `NUM_GPUS` from the environment (default 8).
- Dataflow scripts are templates and require vendor-specific clients.
