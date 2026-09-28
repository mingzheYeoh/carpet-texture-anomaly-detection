FROM node:22-bookworm-slim AS frontend
WORKDIR /build
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-lock.txt ./
# Preserve verified scientific versions, replacing Windows CUDA wheels with CPU wheels.
RUN sed '/^torch==/d; /^torchvision==/d' requirements-lock.txt > /tmp/constraints.txt \
    && pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -c /tmp/constraints.txt anomalib fastapi uvicorn joblib PyYAML \
    && pip check
RUN useradd --create-home --uid 1000 appuser && chown appuser:appuser /app
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser app/api.py ./app/api.py
COPY --chown=appuser:appuser data/manifests/ ./data/manifests/
COPY --chown=appuser:appuser reports/tables/ ./reports/tables/
COPY --chown=appuser:appuser runs/ ./runs/
COPY --from=frontend --chown=appuser:appuser /build/dist/ ./frontend/dist/
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
USER appuser
EXPOSE 7860
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
