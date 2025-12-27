"""Surprise module - prediction error based novelty detection."""

from .forward_model import ForwardModel
from .surprise_module import SurpriseModule
from .volatility_detector import VolatilityDetector

__all__ = ["ForwardModel", "SurpriseModule", "VolatilityDetector"]
