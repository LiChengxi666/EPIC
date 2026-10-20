# Training environment

The default configs use four nodes with eight NVIDIA A100 GPUs each, Megatron
for policy optimization, vLLM for rollout and Ray for distributed execution.

## Version reference

The source snapshot was used with this environment:

| Component | Version |
|---|---|
| Python | 3.12.12 |
| PyTorch | 2.9.1+cu129 |
| NumPy | 1.26.4 |
| transformers | 4.57.6 |
| megatron-core | 0.15.0 |
| mbridge | 0.15.1 |
| tensordict | 0.10.0 |
| hydra-core | 1.3.2 |
| omegaconf | 2.3.0 |
| Ray | 2.46.0.11 (custom build) |
| vLLM | 0.12.1.dev0+g4fd9d6a85.d20260116 (patched build) |

This inventory is not a public package lockfile. Custom binaries are not
bundled. Prepare a compatible CUDA/Megatron/vLLM environment and verify its
extensions before training. Avoid independently upgrading PyTorch underneath
binary vLLM, FlashAttention or Transformer Engine installations.

## Install the framework

From the repository root, in the prepared environment:

```bash
uv pip install --no-deps -e vendor/verl
```

The vendored `setup.py`, `pyproject.toml` and `requirements.txt` describe framework
dependencies. The default EPIC and baseline configs do not enable routing replay.

## Distributed setup

1. Use compatible software on every node.
2. Make model/data/output paths available at identical absolute locations.
3. Start a Ray cluster using your infrastructure's normal procedure.
4. Inspect the launcher configuration without `--execute`.
5. Check model loading, rollout, actor update, evaluation and checkpoint I/O
   before committing resources to a full run.

The launcher connects to an existing Ray cluster; it does not allocate nodes.
CPU numerical tests do not verify the distributed accelerator stack.
