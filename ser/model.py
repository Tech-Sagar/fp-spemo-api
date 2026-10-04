"""The emotion model: one logistic-regression "probe" per frozen encoder, probabilities averaged.

Why this design (see the notebook for the evidence):
  * frozen encoders + a linear probe beat every hand-crafted-feature model by ~20 points,
  * a linear probe on ~1,100 training clips is hard to over-fit and easy to explain,
  * averaging two different encoders (WavLM = speech self-supervision, Whisper = speech
    recognition) is more stable than either one alone.
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import EMOTIONS, NEGATIVE
from .encoders import embed


def make_probe(c: float = 0.1):
    """Standardise, then L2-regularised multinomial logistic regression. Feed float64:
    with float32 input the lbfgs solver is ~100x slower on Apple Accelerate."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))


class EmotionModel:
    def __init__(self, layers: dict[str, list[int]], c: float = 0.1):
        self.layers, self.c = dict(layers), c        # {encoder alias: layer window}
        self.probes: dict = {}

    # ---- training / evaluation on pre-computed features ------------------------------
    def fit(self, feats: dict[str, np.ndarray], y: np.ndarray):
        """feats: {encoder: (n_clips, dim)}; y: emotion ids."""
        for alias in self.layers:
            self.probes[alias] = make_probe(self.c).fit(np.asarray(feats[alias], np.float64), y)
        return self

    def predict_proba(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        return np.mean([self.probes[a].predict_proba(np.asarray(feats[a], np.float64)) for a in self.layers], axis=0)

    # ---- inference on raw audio ------------------------------------------------------
    def featurize(self, y: np.ndarray) -> dict[str, np.ndarray]:
        return {a: embed(y, a, layers)[None] for a, layers in self.layers.items()}

    def predict(self, y: np.ndarray) -> dict:
        """16 kHz waveform -> emotion, confidence, escalation score, all 8 probabilities."""
        p = self.predict_proba(self.featurize(y))[0]
        neg = float(sum(p[EMOTIONS.index(e)] for e in NEGATIVE))
        return {
            "emotion": EMOTIONS[int(p.argmax())],
            "confidence": round(float(p.max()), 4),
            "negative_score": round(neg, 4),
            "probabilities": {e: round(float(v), 4) for e, v in zip(EMOTIONS, p)},
            "duration_s": round(len(y) / 16_000, 2),
        }

    # ---- persistence -----------------------------------------------------------------
    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"layers": self.layers, "c": self.c, "probes": self.probes, "emotions": EMOTIONS}, path)

    @classmethod
    def load(cls, path) -> "EmotionModel":
        d = joblib.load(path)
        m = cls(d["layers"], d["c"])
        m.probes = d["probes"]
        return m
