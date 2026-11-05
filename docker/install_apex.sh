#!/usr/bin/env bash
set -euo pipefail
: "${APEX_COMMIT:?Set an immutable NVIDIA Apex commit}"
export APEX_CPP_EXT=1 APEX_CUDA_EXT=1
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:---threads 2}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
python -m pip install --no-deps --no-build-isolation \
    "https://github.com/NVIDIA/apex/archive/${APEX_COMMIT}.tar.gz"
python -c 'import apex, fused_weight_gradient_mlp_cuda; print("Apex CUDA extensions loaded")'
