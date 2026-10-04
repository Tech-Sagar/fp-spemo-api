"""Everything configurable in one place: paths, labels, speaker split, encoders."""
from pathlib import Path
import os

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FIG_DIR = ARTIFACTS / "figures"
CACHE_DIR = ARTIFACTS / "cache"            # embeddings of every clip (regenerable)
MODEL_PATH = Path(os.environ.get("SER_MODEL_PATH", ARTIFACTS / "model" / "ser_model.joblib"))
# Local folder with pre-downloaded encoders (used in Docker); falls back to the Hugging Face hub.
ENCODER_DIR = Path(os.environ.get("SER_ENCODER_DIR", ARTIFACTS / "encoders"))

SEED = 42
SR = 16_000                                # the pre-trained encoders expect 16 kHz mono

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
NEGATIVE = ["sad", "angry", "fearful", "disgust"]      # used for the "escalate?" business flag

# Speaker-independent split, identical to V1 and V2 so results are comparable:
# these 5 speakers are never used for training, tuning or model selection.
TEST_SPEAKERS = ["Person01", "Person09", "Person12", "Person17", "Person19"]
ALL_SPEAKERS = [f"Person{i:02d}" for i in range(1, 25)]
DEV_SPEAKERS = [s for s in ALL_SPEAKERS if s not in TEST_SPEAKERS]

# Frozen pre-trained encoders (transfer learning). Which middle layers to use is decided by
# the layer scan in the notebook and stored inside the saved model.
ENCODERS = {
    "wavlm_large": {"hf_name": "microsoft/wavlm-large"},
    "whisper_medium": {"hf_name": "openai/whisper-medium"},
}


def find_data_dir() -> Path:
    """Folder containing Person01..Person24 (set SER_DATA_DIR to override)."""
    candidates = [os.environ.get("SER_DATA_DIR"), ROOT / "data", ROOT.parent / "V1_project" / "data"]
    for c in candidates:
        if c and (Path(c) / "Person01").is_dir():
            return Path(c)
    raise FileNotFoundError("RAVDESS audio not found: put Person01..Person24 in ./data or set SER_DATA_DIR.")


def rng() -> np.random.RandomState:
    return np.random.RandomState(SEED)
