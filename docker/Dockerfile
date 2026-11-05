# A100 training environment on a pinned CUDA/PyTorch base.
# BASE_IMAGE may override the default with another compatible immutable image.
# All base images, package distributions and extension sources are public.
ARG BASE_IMAGE=vllm/vllm-openai@sha256:d8d39b59e909d2378ac4feeb191f7e7b6f1342477dc66b7c47cec89e9985ad8a
FROM ${BASE_IMAGE}
WORKDIR /opt/epic
RUN if ! command -v python >/dev/null; then ln -s /usr/bin/python3 /usr/local/bin/python; fi
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_CACHE_DIR=/root/.cache/pip
ARG NVTE_CUDA_ARCHS=80
ARG TORCH_CUDA_ARCH_LIST=8.0
COPY docker/requirements-runtime.txt docker/requirements-training.txt docker/install_training_deps.sh docker/fix_te_wheel_metadata.py docker/build_te_extension.py docker/
COPY docker/wheels/ docker/wheels/
RUN bash docker/install_training_deps.sh
COPY docker/install_apex.sh docker/
ARG APEX_COMMIT=4bdecd06b3c4b2c0a8fb6603829a8f9f05a42b49
RUN bash docker/install_apex.sh
COPY docker/install_flash_attention.sh docker/
COPY docker/flash-attn/ docker/flash-attn/
RUN bash docker/install_flash_attention.sh
# Satisfy the public base's system PyGObject dependency for a clean pip check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libcairo2-dev pkg-config \
    && python -m pip install pycairo==1.28.0 \
    && apt-get clean
# Reinstall the Ray entry points and verify the documented setup commands.
RUN python -m pip install --no-deps --force-reinstall ray==2.52.1 \
    && ray start --help >/dev/null \
    && hf download --help >/dev/null
COPY vendor/verl/ vendor/verl/
COPY configs/ configs/
COPY scripts/ scripts/
COPY epic/ epic/
COPY tests/ tests/
COPY LICENSE NOTICE README.md DATA.md ENVIRONMENT.md IMPLEMENTATION.md ./
RUN python -m pip install --no-deps --no-build-isolation -e vendor/verl \
    && python -m pip check \
    && python scripts/check_training_env.py \
    && python -m pip freeze > /opt/epic/environment.freeze.txt
COPY docs/images/ docs/images/
ENTRYPOINT []
CMD ["/bin/bash"]
