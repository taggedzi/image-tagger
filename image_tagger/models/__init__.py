"""Model registry and base classes for image analysis."""

from .base import AnalysisRequest, ModelCapability, ModelError, ModelInfo, TaggingModel
from .blip import BlipCaptioningModel
from .registry import ModelRegistry
from .vision_remote import OllamaVisionModel

__all__ = [
    "AnalysisRequest",
    "ModelCapability",
    "ModelInfo",
    "ModelError",
    "ModelRegistry",
    "TaggingModel",
    "BlipCaptioningModel",
    "OllamaVisionModel",
]
