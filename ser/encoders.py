"""Frozen pre-trained speech encoders -> one fixed-length vector per clip (transfer learning).

A clip goes through the encoder once. For every hidden layer we summarise the frames over time
with their mean and standard deviation ("statistics pooling"), so a clip of any length becomes
(n_layers, 2 * hidden) numbers. The classifier then uses the average of a few middle layers.
"""
from __future__ import annotations

import numpy as np
import torch

from .config import ENCODERS, ENCODER_DIR, SR

_LOADED: dict = {}


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_encoder(alias: str):
    """(feature_extractor, model, is_whisper). Uses a local copy in ENCODER_DIR when present."""
    if alias not in _LOADED:
        from transformers import AutoFeatureExtractor, AutoModel, WhisperModel
        from transformers.models.whisper.modeling_whisper import WhisperEncoder
        local = ENCODER_DIR / alias
        is_whisper = "whisper" in alias
        if local.is_dir():                         # saved by scripts/download_encoders.py (encoder part only)
            name = str(local)
            model = WhisperEncoder.from_pretrained(name) if is_whisper else AutoModel.from_pretrained(name)
        else:
            name = ENCODERS[alias]["hf_name"]
            model = WhisperModel.from_pretrained(name).encoder if is_whisper else AutoModel.from_pretrained(name)
        _LOADED[alias] = (AutoFeatureExtractor.from_pretrained(name), model.to(device()).eval(), is_whisper)
    return _LOADED[alias]


@torch.no_grad()
def embed_layers(y: np.ndarray, alias: str) -> np.ndarray:
    """16 kHz clip -> (n_layers, 2*hidden) float32: time-mean and time-std of every hidden layer."""
    fe, model, is_whisper = load_encoder(alias)
    dev = device()
    if is_whisper:
        x = fe(y, sampling_rate=SR, return_tensors="pt").input_features.to(dev)   # padded to 30 s
        hs = model(x, output_hidden_states=True).hidden_states
        n = max(1, int(np.ceil(len(y) / SR / 0.02)))                           # 20 ms frames; drop padding
        h = torch.stack([s[0, :n] for s in hs])
    else:
        x = fe(y, sampling_rate=SR, return_tensors="pt").input_values.to(dev)
        h = torch.stack([s[0] for s in model(x, output_hidden_states=True).hidden_states])
    h = h.float()
    return torch.cat([h.mean(1), h.std(1)], dim=-1).cpu().numpy()


def embed(y: np.ndarray, alias: str, layers: list[int]) -> np.ndarray:
    """The classifier input: average of the chosen layer window, shape (2*hidden,)."""
    return embed_layers(y, alias)[layers].mean(0)
