"""Model registry and base classes for image analysis."""

from .base import AnalysisRequest, ModelCapability, ModelInfo, ModelError, TaggingModel
from .registry import ModelRegistry
from .blip import BlipCaptioningModel
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
