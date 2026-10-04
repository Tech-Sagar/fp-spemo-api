"""REST API for the speech-emotion model.

    GET  /                 small web UI: record a voice memo or upload a file
    GET  /health           model + encoder status
    POST /predict          multipart upload:  curl -F "file=@memo.wav" localhost:8000/predict
    POST /predict/raw      raw bytes in the body (any audio container), or headerless 16-bit mono
                           PCM with ?sample_rate=16000
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from ser import EmotionModel
from ser.audio import load_bytes
from ser.config import MODEL_PATH
from ser.encoders import load_encoder

MODEL = EmotionModel.load(MODEL_PATH)
UI = Path(__file__).parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app):
    for alias in MODEL.layers:                     # load encoders once, not on the first request
        load_encoder(alias)
    MODEL.predict(np.random.RandomState(0).randn(16_000).astype(np.float32) * 0.01)
    yield


app = FastAPI(title="Speech Emotion Recognition API", version="3.0", lifespan=lifespan)


def _predict(data: bytes, sample_rate: Optional[int] = None) -> dict:
    if not data:
        raise HTTPException(400, "Empty audio.")
    try:
        y = load_bytes(data, sample_rate)
    except Exception as e:
        raise HTTPException(415, f"Could not decode audio: {e}")
    if len(y) < 0.3 * 16_000:
        raise HTTPException(422, "Less than 0.3 s of speech after trimming silence - please record again.")
    t0 = time.time()
    out = MODEL.predict(y)
    out["latency_ms"] = int((time.time() - t0) * 1000)
    return out


@app.get("/")
def ui():
    return FileResponse(UI)


@app.get("/health")
def health():
    return {"status": "ok", "encoders": MODEL.layers, "classes": 8}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return {"file": file.filename, **_predict(await file.read())}


@app.post("/predict/raw")
async def predict_raw(request: Request, sample_rate: Optional[int] = None):
    return _predict(await request.body(), sample_rate)
