# =========================================================================
# Dockerfile: Production Container for Flight Delay Predictor API
# Base: python:3.14-slim | Non-root security user | Healthcheck enabled
# =========================================================================

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PROJECT_ROOT=/app \
    PYTHONPATH=/app

# Install system dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root system user and group
RUN groupadd -g 1000 appgroup && \
    useradd -u 1000 -g appgroup -s /bin/bash -m appuser

WORKDIR /app

# Install pinned Python dependencies
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy required application code and frozen serving assets
COPY src/ /app/src/
COPY api/ /app/api/
COPY artifacts/feature_pipeline_v2.joblib /app/artifacts/feature_pipeline_v2.joblib
COPY artifacts/models/champion_v2.joblib /app/artifacts/models/champion_v2.joblib
COPY artifacts/models/calibrator_v2.joblib /app/artifacts/models/calibrator_v2.joblib
COPY artifacts/models/model_metadata_v2.json /app/artifacts/models/model_metadata_v2.json
COPY artifacts/serving/ /app/artifacts/serving/
COPY reports/freeze_manifest.json /app/reports/freeze_manifest.json
COPY reports/test_evaluation_metrics.json /app/reports/test_evaluation_metrics.json
COPY reports/phase4_metrics.json /app/reports/phase4_metrics.json

# Install package in editable mode
RUN pip install --no-cache-dir -e .

# Create log directory and configure permissions for non-root user
RUN mkdir -p /app/api/logs && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

# Native container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
