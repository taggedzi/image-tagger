"""Model registry and base classes for image analysis."""

from .base import AnalysisRequest, ModelCapability, ModelInfo, ModelError, TaggingModel
from .registry import ModelRegistry
from .builtin.simple import SimpleHeuristicModel
from .blip import BlipCaptioningModel
from .blip2 import Blip2CaptioningModel
from .openclip import OpenClipTaggingModel
from .vision_remote import OllamaVisionModel

__all__ = [
    "AnalysisRequest",
    "ModelCapability",
    "ModelInfo",
    "ModelError",
    "ModelRegistry",
    "TaggingModel",
    "SimpleHeuristicModel",
    "OpenClipTaggingModel",
    "BlipCaptioningModel",
    "Blip2CaptioningModel",
    "OllamaVisionModel",
]
