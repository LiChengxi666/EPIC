# Overlay for an already validated Linux/CUDA training image.
# Supply BASE_IMAGE as an immutable registry digest. This does not reconstruct
# the custom Ray/vLLM builds from the historical environment.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
WORKDIR /opt/epic
COPY vendor/verl/ vendor/verl/
COPY configs/ configs/
COPY scripts/ scripts/
COPY epic/ epic/
COPY tests/ tests/
COPY LICENSE NOTICE README.md DATA.md ENVIRONMENT.md IMPLEMENTATION.md ./
RUN python -m pip install --no-deps --no-build-isolation -e vendor/verl \
    && python -m pip check \
    && python -m pip freeze > /opt/epic/environment.freeze.txt
CMD ["/bin/bash"]
