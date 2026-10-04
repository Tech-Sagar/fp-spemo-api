"""Download the frozen encoders once and save them to artifacts/encoders (used by the Docker build).
For Whisper only the encoder half is kept - the decoder (text generation) is never used."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transformers import AutoFeatureExtractor, AutoModel, WhisperModel  # noqa: E402

from ser.config import ENCODERS, ENCODER_DIR  # noqa: E402

for alias, cfg in ENCODERS.items():
    out = ENCODER_DIR / alias
    if out.is_dir():
        print(f"{alias}: already in {out}")
        continue
    name = cfg["hf_name"]
    model = WhisperModel.from_pretrained(name).encoder if "whisper" in alias else AutoModel.from_pretrained(name)
    model.save_pretrained(out)
    AutoFeatureExtractor.from_pretrained(name).save_pretrained(out)
    print(f"{alias}: saved {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters to {out}")
