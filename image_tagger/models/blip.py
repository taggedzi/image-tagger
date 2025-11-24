"""BLIP-based captioning model using HuggingFace transformers."""

from __future__ import annotations

import logging
import math
import re

from PIL import Image

from ..utils.devices import detect_torch_device
from .base import (
    AnalysisRequest,
    ModelCapability,
    ModelError,
    ModelInfo,
    ModelOutput,
    ModelTag,
    TaggingModel,
)
from .registry import ModelRegistry

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional dependency handling
    pipeline = None  # type: ignore[assignment]

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


STOP_WORDS = {
    "the",
    "and",
    "with",
    "from",
    "into",
    "over",
    "under",
    "behind",
    "beside",
    "a",
    "an",
    "of",
    "in",
    "on",
    "to",
    "by",
    "for",
    "at",
    "around",
    "between",
    "while",
    "during",
    "through",
    "among",
    "near",
    "its",
    "their",
    "his",
    "her",
    "there",
    "this",
    "that",
    "these",
    "those",
}


class BlipCaptioningModel(TaggingModel):
    """Caption generator built on Salesforce's BLIP model."""

    def __init__(
        self,
        *,
        model_id: str,
        identifier: str,
        display_name: str,
        description: str | None = None,
    ) -> None:
        self.model_id = model_id
        self._info = ModelInfo(
            identifier=identifier,
            display_name=display_name,
            description=description
            or f"Generates descriptive captions via the BLIP model ({model_id}).",
            capabilities=(ModelCapability.CAPTION, ModelCapability.TAGS),
            tags=("transformers", "torch", "caption"),
        )
        self._pipeline = None
        self._device_choice: str | int | None = None

    def info(self) -> ModelInfo:
        return self._info

    def load(self) -> None:
        if pipeline is None or torch is None:
            raise ModelError(
                "transformers, accelerate, and torch must be installed to use the BLIP model. "
                "Install with `pip install image-tagger[blip]`."
            )

        device_str, message = detect_torch_device()
        logger.info("[BLIP] %s", message)

        if device_str.startswith("cuda"):
            try:
                _, index_str = device_str.split(":", 1)
                device = int(index_str)
            except Exception:
                device = 0
        elif device_str == "mps":
            device = "mps"
        else:
            device = -1

        self._pipeline = pipeline(
            "image-to-text",
            model=self.model_id,
            device=device,
        )
        self._device_choice = device_str

    def analyze(self, image: Image.Image, request: AnalysisRequest) -> ModelOutput:
        assert self._pipeline is not None

        tags: list[ModelTag] = []
        caption: str | None = None

        if request.generate_captions:
            outputs = self._pipeline(image)
            if isinstance(outputs, list):
                if outputs:
                    first = outputs[0]
                    if isinstance(first, dict):
                        caption = first.get("generated_text", "").strip()
                    else:
                        caption = str(first).strip()
            elif isinstance(outputs, dict):
                caption = str(outputs.get("generated_text", "")).strip()
            elif outputs is not None:
                caption = str(outputs).strip()

        if caption:
            caption = caption.strip()
        if not caption:
            caption = None

        if request.generate_tags and caption:
            tags = self._extract_tags_from_caption(
                caption, max_count=request.max_tags, threshold=request.confidence_threshold
            )

        return ModelOutput(
            caption=caption,
            tags=tags,
            extras={
                "model_id": self.model_id,
                "device": self._device_choice,
            },
        )

    @staticmethod
    def _extract_tags_from_caption(
        caption: str,
        *,
        max_count: int,
        threshold: float,
    ) -> list[ModelTag]:
        words = re.findall(r"[a-zA-Z][a-zA-Z\\-]+", caption.lower())
        seen: set[str] = set()
        tags: list[ModelTag] = []
        base_confidence = max(0.55, min(0.85, 1.0 - math.log(max_count + 1, 10)))
        for word in words:
            if word in STOP_WORDS:
                continue
            normalized = word.strip("-")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            confidence = base_confidence
            if confidence < threshold:
                continue
            tags.append(ModelTag(normalized, confidence=confidence))
            if len(tags) >= max_count:
                break
        return tags


def _register() -> None:
    available_models = [
        (
            "caption.blip",
            "Salesforce/blip-image-captioning-large",
            "BLIP Captioner (Large)",
            "Highest quality BLIP checkpoint; requires more VRAM.",
        ),
        (
            "caption.blip-base",
            "Salesforce/blip-image-captioning-base",
            "BLIP Captioner (Base)",
            "Base BLIP checkpoint with balanced quality and performance.",
        ),
    ]

    for identifier, model_id, label, desc in available_models:

        def _factory(
            model_id=model_id,
            identifier=identifier,
            label=label,
            desc=desc,
        ) -> BlipCaptioningModel:
            return BlipCaptioningModel(
                model_id=model_id,
                identifier=identifier,
                display_name=label,
                description=desc,
            )

        ModelRegistry.register(identifier, _factory)


_register()
