#!/usr/bin/env bash
set -euo pipefail
# Compile the TE PyTorch bindings against the base image's existing torch.
# A CPU-only Docker builder must specify architectures explicitly.
export MAX_JOBS="${MAX_JOBS:-4}"
export NVTE_BUILD_THREADS_PER_JOB="${NVTE_BUILD_THREADS_PER_JOB:-2}"
export NVTE_PYTORCH_FORCE_BUILD=TRUE
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-80}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
# The public base has cuDNN through PyTorch's NVIDIA wheel, not system headers.
cudnn_root=$(python - <<'CUDNN'
from importlib.metadata import distribution
root = distribution('nvidia-cudnn-cu12').locate_file('nvidia/cudnn')
if not (root / 'include/cudnn.h').is_file():
    raise RuntimeError('The installed CUDA 12 cuDNN package must include cudnn.h')
print(root)
CUDNN
)
export CPATH="$cudnn_root/include${CPATH:+:$CPATH}"
export CPLUS_INCLUDE_PATH="$cudnn_root/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
export LIBRARY_PATH="$cudnn_root/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
constraints=$(mktemp)
trap 'rm -f "$constraints"' EXIT
python - "$constraints" <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import sys
protected = {'torch', 'torchvision', 'torchaudio', 'triton', 'vllm', 'flash-attn'}
lines = []
for dist in metadata.distributions():
    name = dist.metadata['Name'].lower().replace('_', '-')
    if name in protected or name.startswith('nvidia-'):
        lines.append(f'{name}=={dist.version}')
Path(sys.argv[1]).write_text('\n'.join(sorted(set(lines))) + '\n')
PY
python -m pip install --constraint "$constraints" \
    ninja==1.11.1.4 pybind11==2.13.6 setuptools==78.1.1 wheel==0.45.1
mkdir -p /opt/epic/docker/wheels
# An optional local wheel avoids a large download on restricted build networks.
for wheel in /opt/epic/docker/wheels/*.whl; do
    if [ -f "$wheel" ]; then
        python -m pip install --no-deps --constraint "$constraints" "$wheel"
    fi
done
python -m pip install --constraint "$constraints" -r /opt/epic/docker/requirements-runtime.txt
python -m pip install --no-deps --constraint "$constraints" \
    transformer-engine==2.9.0 transformer-engine-cu12==2.9.0
python /opt/epic/docker/build_te_extension.py --constraints "$constraints"
python -m pip install --no-build-isolation --constraint "$constraints" \
    -r /opt/epic/docker/requirements-training.txt

python /opt/epic/docker/fix_te_wheel_metadata.py
