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

## Container workflow

`docker/Dockerfile` installs this repository on top of an existing, validated
Linux/CUDA training image. It deliberately has no default base image: the
historical custom Ray and patched vLLM builds cannot be reconstructed from
their version strings alone. This Dockerfile is an overlay recipe, not a
published or GPU-validated reproduction image.

Once a compatible base image is available, build from the repository root:

```bash
export EPIC_BASE_IMAGE='YOUR_REGISTRY/VALIDATED_TRAINING_IMAGE@sha256:YOUR_DIGEST'
docker build --build-arg BASE_IMAGE="$EPIC_BASE_IMAGE" -f docker/Dockerfile -t epic:local .
```

Replace the placeholders with a real immutable image digest. The build uses
`--no-deps --no-build-isolation` to preserve the GPU stack, checks package
dependencies, and records `environment.freeze.txt` inside the image. The base
must already contain Python, pip, setuptools, wheel and training dependencies.
Weights, datasets, outputs and credentials are not copied into the image.

Use the same image digest on all nodes, with shared paths and the NVIDIA
container runtime configured by the cluster operator. Start Ray as appropriate
for that cluster. An image packages dependencies; it does not allocate GPUs or
configure inter-node communication.

Before publishing a reproducible image:

1. Obtain the tested base image or the source patches/build instructions for
   its custom components. Record the base digest and CUDA/driver requirements.
2. If using public replacements, select a compatible full stack and test it;
   replacing only the Ray or vLLM version is not sufficient evidence.
3. Verify model loading, one rollout, one actor update, validation and checkpoint
   save/resume with the supplied EPIC configuration on the target GPU setup.
4. Record the exact image digest, configuration, prepared-data manifest and
   smoke-run logs. Then run the full training/evaluation protocol.

No public base-image digest has yet been validated for this release. Do not
describe the overlay build or CPU unit tests as a completed GPU reproduction.
