#!/usr/bin/env bash
set -euo pipefail
# TE 2.9 supports FlashAttention <=2.8.1; the base's 2.8.3 is incompatible.
# Use an official wheel matching this image's Python, torch and C++ ABI.
wheel=$(python - <<'PY'
import sys
import torch
if torch.version.cuda.split('.')[0] != '12':
    raise RuntimeError('This profile requires CUDA 12')
python = f'cp{sys.version_info.major}{sys.version_info.minor}'
version = '.'.join(torch.__version__.split('.')[:2])
abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
print(f'flash_attn-2.8.1+cu12torch{version}cxx11abi{abi}-{python}-{python}-linux_x86_64.whl')
PY
)
source="/opt/epic/docker/flash-attn/$wheel"
if [ ! -f "$source" ]; then
    source="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/$wheel"
fi
python -m pip install --no-deps --force-reinstall "$source"
python -c 'import flash_attn; print("FlashAttention", flash_attn.__version__)'
