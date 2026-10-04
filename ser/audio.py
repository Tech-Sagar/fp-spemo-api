"""Audio in: files or raw bytes -> 16 kHz mono float32, silence trimmed."""
from __future__ import annotations

import io
import shutil
import subprocess

import librosa
import numpy as np
import soundfile as sf

from .config import SR

MAX_SECONDS = 30                           # Whisper's encoder window; longer audio is cut


def trim_silence(y: np.ndarray, top_db: float = 30.0, pad_s: float = 0.1) -> np.ndarray:
    """Drop leading/trailing silence (RAVDESS clips carry ~1 s of it), keep a short pad."""
    if len(y) == 0:
        return y
    _, (start, end) = librosa.effects.trim(y, top_db=top_db)
    pad = int(pad_s * SR)
    return y[max(0, start - pad): min(len(y), end + pad)]


def _finish(y: np.ndarray, sr: int) -> np.ndarray:
    y = y.mean(axis=1) if y.ndim > 1 else y                # stereo -> mono
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    y = trim_silence(y.astype(np.float32))
    return y[: MAX_SECONDS * SR]


def load_file(path) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    return _finish(y, sr)


def load_bytes(data: bytes, sample_rate: int | None = None) -> np.ndarray:
    """Decode any audio container (wav, flac, ogg, mp3; m4a/webm via ffmpeg).
    With `sample_rate`, `data` is treated as headerless 16-bit mono PCM."""
    if sample_rate:
        return _finish(np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768, sample_rate)
    try:
        y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return _finish(y, sr)
    except Exception:
        if not shutil.which("ffmpeg"):
            raise ValueError("Unsupported audio format (install ffmpeg for m4a/webm, or send WAV).")
        out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1",
                              "-ar", str(SR), "pipe:1"], input=data, capture_output=True, check=True).stdout
        return _finish(np.frombuffer(out, dtype=np.float32).copy(), SR)


def add_noise(y: np.ndarray, snr_db: float = 20, seed: int = 0) -> np.ndarray:
    """Training augmentation: white noise at 20 dB SNR, mimicking a phone / call-centre mic."""
    noise = np.random.RandomState(seed).randn(len(y))
    return (y + noise * np.sqrt(np.mean(y ** 2) / 10 ** (snr_db / 10))).astype(np.float32)
