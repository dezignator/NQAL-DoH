# =============================================================================
# NQAL-DoH Dockerfile
# Multi-stage build: python base → deps → application
# Final image ~2.5 GB (torch is heavy); use `--slim` target for CI.
# =============================================================================
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl ca-certificates libgomp1 graphviz \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# Deps
# -----------------------------------------------------------------------------
FROM base AS deps
WORKDIR /tmp
COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools \
 && pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Application (full image)
# -----------------------------------------------------------------------------
FROM deps AS app
WORKDIR /opt/nqal
COPY . .
RUN pip install -e . --no-deps
RUN mkdir -p artifacts/logs artifacts/checkpoints artifacts/results
ENV PYTHONPATH=/opt/nqal
EXPOSE 8501
CMD ["python", "-c", "from models.nqal_pipeline import NQALPipeline; print('NQAL-DoH ready')"]

# -----------------------------------------------------------------------------
# Slim image for CI (no GPU, no graphviz)
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS slim
WORKDIR /opt/nqal
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install numpy scipy pandas scikit-learn pyyaml tqdm py_ecc cryptography
COPY models/ models/
COPY utils/ utils/
COPY hardware/ hardware/
COPY config/ config/
RUN python -c "from models.nqal_pipeline import NQALPipeline; print('slim OK')"
