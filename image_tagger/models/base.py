"""Abstract interfaces for image tagging models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Protocol, Sequence

try:  # Pillow is optional for type checking until runtime.
    from PIL import Image
except Exception:  # pragma: no cover
    Image = object  # type: ignore[assignment]


class ModelCapability(str, Enum):
    """Kinds of outputs a model can provide."""

    CAPTION = "caption"
    TAGS = "tags"


@dataclass(slots=True)
class ModelTag:
    """Represents a single keyword produced by a model."""

    value: str
    confidence: float | None = None

    def as_dict(self) -> dict[str, float | str]:
        payload: dict[str, float | str] = {"value": self.value}
        if self.confidence is not None:
            payload["confidence"] = float(self.confidence)
        return payload


@dataclass(slots=True)
class ModelOutput:
    """Structured result coming back from a tagging model."""

    caption: str | None = None
    tags: list[ModelTag] = field(default_factory=list)
    extras: dict[str, object] = field(default_factory=dict)

    def truncated(self, *, max_tags: int) -> "ModelOutput":
        """Return a copy with at most ``max_tags`` items."""
        return ModelOutput(
            caption=self.caption,
            tags=self.tags[:max_tags],
            extras=self.extras,
        )


@dataclass(slots=True)
class ModelInfo:
    """Metadata describing an available model implementation."""

    identifier: str
    display_name: str
    description: str
    capabilities: tuple[ModelCapability, ...]
    tags: Sequence[str] = ()


@dataclass(slots=True)
class AnalysisRequest:
    """Normalized model inputs derived from the application config."""

    generate_captions: bool
    generate_tags: bool
    max_tags: int
    confidence_threshold: float
    locale: str | None = None


class ModelError(RuntimeError):
    """Raised when a model cannot produce output for a given request."""


class TaggingModel(Protocol):
    """Interface that all tagging models must satisfy."""

    def info(self) -> ModelInfo:
        """Return metadata describing the model."""

    def load(self) -> None:
        """Perform any expensive model initialisation."""

    def analyze(self, image: "Image.Image", request: AnalysisRequest) -> ModelOutput:
        """Generate tags and/or captions for the provided image."""

