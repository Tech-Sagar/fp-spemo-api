# Speech Emotion Recognition API + web UI (CPU). Build after running the notebook, which
# writes artifacts/model/ser_model.joblib.
#   docker build -t ser-api .
#   docker run -p 8000:8000 ser-api        ->  http://localhost:8000
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/tmp/hf SER_ENCODER_DIR=/app/artifacts/encoders

# CPU-only PyTorch keeps the image ~2 GB smaller than the default CUDA build.
RUN pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Bake the two frozen encoders into the image (no downloads when the container starts).
COPY ser/ ser/
COPY scripts/download_encoders.py scripts/
RUN python scripts/download_encoders.py && rm -rf /tmp/hf

COPY api/ api/
COPY artifacts/model/ser_model.joblib artifacts/model/

EXPOSE 8000
HEALTHCHECK --interval=30s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
