"""CLIP-based tagging model implemented with OpenCLIP."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

from PIL import Image

from .base import (
    AnalysisRequest,
    ModelCapability,
    ModelInfo,
    ModelOutput,
    ModelTag,
    TaggingModel,
    ModelError,
)
from .registry import ModelRegistry
from ..utils.devices import detect_torch_device, torch_device

logger = logging.getLogger(__name__)

try:
    import torch
    import open_clip
except Exception:  # pragma: no cover - optional dependency handling
    torch = None  # type: ignore[assignment]
    open_clip = None  # type: ignore[assignment]


def _default_candidate_tags() -> list[str]:
    # General-purpose vocabulary drawn from common photographic subjects.
    return [
        "portrait",
        "landscape",
        "architecture",
        "nature",
        "mountain",
        "ocean",
        "beach",
        "forest",
        "city",
        "street",
        "car",
        "train",
        "airplane",
        "sunset",
        "sunrise",
        "night",
        "snow",
        "rain",
        "flower",
        "animal",
        "dog",
        "cat",
        "bird",
        "food",
        "dessert",
        "fruit",
        "building",
        "bridge",
        "monument",
        "statue",
        "people",
        "group",
        "festival",
        "concert",
        "sports",
        "basketball",
        "football",
        "tennis",
        "swimming",
        "technology",
        "computer",
        "robot",
        "painting",
        "sculpture",
        "interior",
        "macro",
        "texture",
        "abstract",
        "pattern",
        "fire",
        "smoke",
        "waterfall",
        "river",
        "lake",
        "forest trail",
        "desert",
        "canyon",
        "wildlife",
        "insect",
        "butterfly",
        "wedding",
        "fashion",
        "business",
        "office",
        "drone view",
        "aerial",
        "night sky",
        "stars",
        "milky way",
        "storm",
        "lightning",
        "clouds",
        "garden",
        "farm",
        "market",
        "statue",
        "castle",
        "temple",
        "island",
        "volcano",
        "forest animals",
        "jungle",
        "beverage",
        "coffee",
        "tea",
        "beer",
        "wine",
        "portrait closeup",
        "selfie",
        "family",
        "baby",
        "child",
        "elderly",
        "vacation",
        "travel",
        "road trip",
        "camping",
        "hiking",
        "technology hardware",
        "smartphone",
        "tablet",
        "gaming",
        "musical instrument",
        "guitar",
        "piano",
        "drums",
        "microphone",
    ]


@dataclass(slots=True)
class _TagScore:
    label: str
    confidence: float


class OpenClipTaggingModel(TaggingModel):
    """Tagging model backed by OpenCLIP embeddings."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        candidate_tags: Sequence[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.candidate_tags = list(candidate_tags or _default_candidate_tags())
        self._info = ModelInfo(
            identifier="clip.openclip",
            display_name="OpenCLIP Tagger",
            description="Generates tags via CLIP similarity scoring.",
            capabilities=(ModelCapability.TAGS, ModelCapability.CAPTION),
            tags=("clip", "torch", "ml"),
        )
        self._device = None
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_features = None

    def info(self) -> ModelInfo:
        return self._info

    def load(self) -> None:
        if torch is None or open_clip is None:
            raise ModelError(
                "open-clip-torch and torch must be installed to use the OpenCLIP model. "
                "Install with `pip install image-tagger[clip]`."
            )
        device_str, message = detect_torch_device()
        logger.info("[OpenCLIP] %s", message)

        if device_str.startswith("cuda"):
            create_device = "cuda"
            try:
                _, index_str = device_str.split(":", 1)
                torch.cuda.set_device(int(index_str))
            except Exception:  # pragma: no cover - fallback to default device
                logger.debug("Unable to set CUDA device to %s; using default.", device_str)
        elif device_str == "cpu":
            create_device = "cpu"
        else:
            logger.warning(
                "OpenCLIP does not support device '%s'; falling back to CPU.", device_str
            )
            device_str = "cpu"
            create_device = "cpu"

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=create_device,
        )
        tokenizer = open_clip.get_tokenizer(self.model_name)

        model.eval()
        model.to(torch_device(device_str))

        with torch.no_grad():
            tokens = tokenizer(self.candidate_tags).to(torch_device(device_str))
            text_features = model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        self._device = torch_device(device_str)
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = tokenizer
        self._text_features = text_features

    def analyze(self, image: Image.Image, request: AnalysisRequest) -> ModelOutput:
        assert self._model is not None and self._preprocess is not None
        assert self._text_features is not None and self._device is not None

        tags: list[ModelTag] = []
        caption: str | None = None

        with torch.no_grad():
            image_tensor = self._preprocess(image).unsqueeze(0).to(self._device)
            image_features = self._model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = (image_features @ self._text_features.T).squeeze(0)
            probabilities = logits.softmax(dim=-1)

        scores = [
            _TagScore(label=label, confidence=float(prob))
            for label, prob in zip(self.candidate_tags, probabilities.cpu().tolist())
        ]
        scores.sort(key=lambda item: item.confidence, reverse=True)

        if request.generate_tags:
            for score in scores[: request.max_tags * 2]:
                if score.confidence < request.confidence_threshold:
                    continue
                tags.append(ModelTag(score.label, confidence=score.confidence))
                if len(tags) >= request.max_tags:
                    break

        if request.generate_captions:
            top_tags = [tag.value for tag in tags[:3]] or [
                score.label for score in scores[:3]
            ]
            caption = self._build_caption(top_tags)

        extras = {"device": str(self._device) if self._device is not None else "cpu"}
        return ModelOutput(caption=caption, tags=tags, extras=extras)

    @staticmethod
    def _build_caption(tags: Iterable[str]) -> str:
        selected = [tag.replace("-", " ") for tag in tags]
        if not selected:
            return "Image content could not be determined."
        if len(selected) == 1:
            return f"A scene featuring {selected[0]}."
        if len(selected) == 2:
            return f"A scene featuring {selected[0]} and {selected[1]}."
        return f"A scene featuring {', '.join(selected[:-1])}, and {selected[-1]}."


def _register() -> None:
    ModelRegistry.register("clip.openclip", OpenClipTaggingModel)


_register()
