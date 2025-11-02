"""Model registry and base classes for image analysis."""

from .base import AnalysisRequest, ModelCapability, ModelInfo, TaggingModel
from .registry import ModelRegistry
from .builtin.simple import SimpleHeuristicModel

__all__ = [
    "AnalysisRequest",
    "ModelCapability",
    "ModelInfo",
    "ModelRegistry",
    "TaggingModel",
    "SimpleHeuristicModel",
]
