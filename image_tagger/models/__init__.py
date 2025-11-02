"""Model registry and base classes for image analysis."""

from .base import AnalysisRequest, ModelCapability, ModelInfo, ModelError, TaggingModel
from .registry import ModelRegistry
from .builtin.simple import SimpleHeuristicModel
from .openclip import OpenClipTaggingModel
from .blip import BlipCaptioningModel

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
]
