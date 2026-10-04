"""Speech Emotion Recognition (simplified V3): frozen speech encoders + logistic-regression probes."""
import warnings

# Apple's Accelerate BLAS emits harmless "encountered in matmul" warnings on macOS.
warnings.filterwarnings("ignore", message=".*encountered in matmul")

from .model import EmotionModel  # noqa: E402

__all__ = ["EmotionModel"]
