FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PATHSNAP_DATA_DIR=/data PATHSNAP_REFERENCE_DIR=/references \
    PATHSNAP_MODEL=/artifacts/model.joblib \
    PATHSNAP_MODEL_METADATA=/artifacts/model_metadata.json \
    PATHSNAP_MATCH_CONFIG=/artifacts/match_config.json \
    PATHSNAP_INFERENCE_CONCURRENCY=1
WORKDIR /app
COPY deploy/requirements.txt /app/deploy/requirements.txt
RUN pip install --no-cache-dir -r /app/deploy/requirements.txt \
    && useradd --uid 10001 --create-home pathsnap
COPY mapmatch /app/mapmatch
COPY backend /app/backend
COPY frontend /app/frontend
# No training inputs, route outputs, references, secrets or weights enter image.
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -B -c "import json,urllib.request; h=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=8)); raise SystemExit(0 if h.get('status')=='ready' and h.get('model_loaded') else 1)"
CMD ["python", "-B", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log", "--limit-concurrency", "16", "--timeout-keep-alive", "5"]
