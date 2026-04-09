"""Model modules for deepfake detection."""

from src.models.lstm_model import DeepfakeLSTM
from src.models.trainer import Trainer

__all__ = ["DeepfakeLSTM", "Trainer"]
