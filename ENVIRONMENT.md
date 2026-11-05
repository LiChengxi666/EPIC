# Training environment

The training image builds entirely from public images, Python packages and
source repositories. No private registry, organization account, custom binary
build, cluster queue or user-specific filesystem is required.

The default experiment uses Qwen3-30B-A3B-Base on four nodes with eight A100
GPUs each. Model weights and datasets are prepared separately using [DATA.md](DATA.md).

## Build the image

Use Linux x86_64, Docker, NVIDIA Container Toolkit and an NVIDIA driver
compatible with CUDA 12.8. Building extensions requires CPU/RAM and disk space;
GPUs are not required during the image build.

```bash
git clone https://github.com/LiChengxi666/EPIC.git
cd EPIC
docker build -f docker/Dockerfile -t epic:local .
```

The default base is the public `vllm/vllm-openai:v0.11.0` Linux amd64 image,
pinned by digest in `docker/Dockerfile`. It supplies Python 3.12, CUDA 12.8
including the compiler, PyTorch 2.8 and vLLM 0.11. The build explicitly installs
the remaining runtime packages from `docker/requirements-runtime.txt`, plus:

| Component | Version/source |
|---|---|
| Megatron-Core | 0.15.0 |
| MBridge | 0.15.1 |
| Transformer Engine | 2.9.0, CUDA 12 library and compiled PyTorch extension |
| Apex | NVIDIA source commit `4bdecd06b3c4b2c0a8fb6603829a8f9f05a42b49` |
| FlashAttention | 2.8.1, official wheel matching Python, PyTorch and C++ ABI |
| transformers | 4.56.1 |
| Ray | 2.52.1 |
| NumPy | 1.26.4 |
| tensordict | 0.10.0 |
| Hydra / OmegaConf | 1.3.2 / 2.3.0 |

TE's PyTorch extension and Apex's C++/CUDA extensions compile against the
image's existing PyTorch with build isolation disabled. cuDNN headers and
libraries are located from PyTorch's public `nvidia-cudnn-cu12` package. Apex includes
`fused_weight_gradient_mlp_cuda`, required by gradient accumulation fusion.
FlashAttention 2.8.1 is selected because TE 2.9 rejects versions above 2.8.1.
The build preserves the base's PyTorch, vLLM, Triton and NVIDIA runtime package
versions rather than upgrading them independently.

The default extension architecture is A100 (SM80). To build for another GPU,
pass `--build-arg NVTE_CUDA_ARCHS=... --build-arg TORCH_CUDA_ARCH_LIST=...` using
the architecture formats accepted by TE and PyTorch. Other GPU architectures
and alternative `BASE_IMAGE` values require their own validation.

All downloads use public sources. `PIP_INDEX_URL` may select a public mirror.
For restricted networks, optional wheels can be cached in `docker/wheels/`
and the matching official FlashAttention wheel in `docker/flash-attn/`;
these directories contain only placeholders in Git. No cached wheel is needed
to build from a fresh clone.

TE's PyTorch extension is built from the public 2.9.0 source archive with its
SHA-256 verified. A build-only patch adds the missing return to that release's
force-build branch, avoiding a second wheel build and a prebuilt-wheel probe.
CUDA kernel sources are unchanged.

The TE 2.9 CUDA-only wheel incorrectly records a CPython 3.10 ABI tag internally
despite its published `py3-none` filename. The build corrects that metadata
and its RECORD hash; it does not change the CUDA library or the separately
compiled PyTorch extension.

The build installs the vendored verl, runs `pip check`, loads the compiled
extensions, and checks Qwen3 MoE configuration with fused RoPE enabled. The
resolved package inventory is saved to `/opt/epic/environment.freeze.txt`.
Model weights, datasets, credentials and training outputs are excluded from
the build context.

## Check a GPU node

```bash
docker run --rm --gpus all --ipc=host epic:local \
  python scripts/check_training_env.py --require-gpu
```

This executes TE and FlashAttention forward/backward operations and compares
Apex's fused weight gradient with a matrix-multiplication reference. Run it on
every node before creating the training cluster. CPU tests alone do not check
CUDA extensions, model loading or distributed execution.

## Prepare model and data

Mount a shared directory at the same absolute path on every node. For example,
use `/shared` for model weights, processed Parquet files and experiment outputs.
Follow [DATA.md](DATA.md) to prepare all five datasets and download the model.
Data can also be prepared using the image's Python:

```bash
docker run --rm -v /shared:/shared epic:local \
  python scripts/prepare_data.py --output /shared/data

docker run --rm -v /shared:/shared epic:local \
  python scripts/check_data.py /shared/data --check-counts
```

The preparation output directory must be new. Preserve its
`preparation_manifest.json` alongside experiment outputs.

## Distributed setup

Use the same built image on every node. Transfer the image using your preferred
registry or `docker save` / `docker load`; no particular registry is required.
The default configuration expects 32 A100 GPUs across four nodes.

Start an interactive training container on each node with GPUs, host networking
and the shared directory:

```bash
docker run --rm -it --gpus all --ipc=host --network=host \
  -v /shared:/shared epic:local bash
```

Choose the head node's routable address as `HEAD_IP`, and each node's own
routable address as `NODE_IP`. The nodes must be able to communicate over Ray
and distributed-training ports. Run the following commands inside the
containers, setting the variables for your network.

On the head node:

```bash
ray start --head --node-ip-address="$HEAD_IP" --port=6379 \
  --num-gpus=8 --disable-usage-stats
```

On each of the other three nodes:

```bash
ray start --address="$HEAD_IP:6379" --node-ip-address="$NODE_IP" \
  --num-gpus=8 --disable-usage-stats
```

On the head node, check `ray status`, then inspect the resolved EPIC
configuration before launching:

```bash
python scripts/run_experiment.py epic \
  --model /shared/model/Qwen3-30B-A3B-Base \
  --data /shared/data --output /shared/outputs/epic --ray-address auto
```

After checking the paths and resources, run the same command with `--execute`.
Use a new output directory. The launcher connects to Ray; it does not provision
machines or allocate GPUs. The configuration JSON files under `configs/` remain
the default 32-GPU experiment configurations.

## Validation scope

This public-source image passed `pip check`, all 21 repository unit tests and
the GPU environment check on A100. A startup run with public Qwen3-0.6B weights
completed one EPIC step through Megatron and vLLM, validation generation,
Hugging Face model export and distributed checkpoint saving. Short responses
produced zero rewards, so this verifies execution rather than learning quality.
The default Qwen3-30B-A3B-Base experiment across 32 GPUs has not been rerun with
this image; full-scale performance and checkpoint resume require separate
validation.
