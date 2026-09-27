# sklearn-mastery: reproducible experiment image
#
#   docker build -t sklearn-mastery .
#   docker run --rm sklearn-mastery sklearn-mastery benchmark --quick
#   docker run --rm -p 8888:8888 -v "$PWD:/workspace" sklearn-mastery \
#       sklearn-mastery launch-notebooks --port 8888

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    SKLEARN_MASTERY_ROOT=/workspace

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first so source edits do not invalidate the layer.
COPY pyproject.toml README.md LICENSE ./
COPY sklearn_mastery ./sklearn_mastery
RUN pip install --upgrade pip && pip install ".[all]" jupyterlab

# Now the rest of the repository (notebooks, examples, docs, tests).
COPY . .
RUN pip install --no-deps -e . && mkdir -p results logs data

EXPOSE 8888

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sklearn_mastery" || exit 1

LABEL org.opencontainers.image.title="sklearn-mastery" \
      org.opencontainers.image.description="Research-grade toolkit for reproducible scikit-learn experiments" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.authors="Satvik Praveen <satvikpraveen707@gmail.com>" \
      org.opencontainers.image.source="https://github.com/SatvikPraveen/Sklearn-Mastery" \
      org.opencontainers.image.licenses="MIT"

CMD ["sklearn-mastery", "--help"]
