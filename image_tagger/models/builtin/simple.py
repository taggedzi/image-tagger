"""A lightweight, dependency-free heuristic model used as a baseline."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from PIL import Image, ImageStat

from ..base import (
    AnalysisRequest,
    ModelCapability,
    ModelInfo,
    ModelOutput,
    ModelTag,
    TaggingModel,
)
from ..registry import ModelRegistry


def _brightness_levels(rgb_means: tuple[float, float, float]) -> tuple[str, float]:
    brightness = sum(rgb_means) / (3 * 255)
    if brightness > 0.75:
        return "bright", brightness
    if brightness < 0.25:
        return "dark", 1.0 - brightness
    return "balanced-exposure", 0.5


def _dominant_colors(rgb_means: tuple[float, float, float]) -> list[tuple[str, float]]:
    r, g, b = rgb_means
    total = max(r + g + b, 1.0)
    contributions = {
        "red": r / total,
        "green": g / total,
        "blue": b / total,
    }
    sorted_items = sorted(contributions.items(), key=lambda item: item[1], reverse=True)
    return [(name, score) for name, score in sorted_items if score > 0.34]


def _orientation_tag(image: Image.Image) -> tuple[str, float]:
    width, height = image.size
    if width > height:
        return "landscape-orientation", width / max(height, 1)
    if height > width:
        return "portrait-orientation", height / max(width, 1)
    return "square-orientation", 1.0


def _dominant_palette(image: Image.Image) -> list[tuple[str, float]]:
    quantized = image.convert("RGB").resize((64, 64))
    colors = quantized.getcolors(64 * 64) or []
    if not colors:
        return []
    total = sum(count for count, _ in colors)
    counter: Counter[str] = Counter()
    for count, color in colors:
        label = _label_color(color)
        counter[label] += count
    return [(name, score / total) for name, score in counter.most_common(4)]


def _label_color(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    if max(rgb) < 40:
        return "very-dark"
    if min(rgb) > 215:
        return "very-light"
    if abs(r - g) < 15 and abs(g - b) < 15:
        if r > 200:
            return "light-neutral"
        if r < 60:
            return "dark-neutral"
        return "neutral"
    if r > g and r > b:
        return "red-toned"
    if g > r and g > b:
        return "green-toned"
    if b > r and b > g:
        return "blue-toned"
    if r > 180 and g > 180:
        return "yellow-toned"
    if r > 180 and b > 180:
        return "magenta-toned"
    if g > 180 and b > 180:
        return "cyan-toned"
    return "multicolor"


class SimpleHeuristicModel(TaggingModel):
    """A heuristic model that inspects image statistics."""

    def __init__(self) -> None:
        self._info = ModelInfo(
            identifier="builtin.simple",
            display_name="Simple Heuristic",
            description="Generates basic tags from image statistics without ML dependencies.",
            capabilities=(ModelCapability.CAPTION, ModelCapability.TAGS),
            tags=("lightweight", "no-internet", "cpu"),
        )

    def info(self) -> ModelInfo:
        return self._info

    def load(self) -> None:
        # Nothing to initialise for the heuristic model.
        return

    def analyze(self, image: Image.Image, request: AnalysisRequest) -> ModelOutput:
        image = image.convert("RGB")
        stat = ImageStat.Stat(image)
        mean_rgb = tuple(stat.mean[:3])  # type: ignore[assignment]

        tags: list[ModelTag] = []
        if request.generate_tags:
            brightness_tag, brightness_score = _brightness_levels(mean_rgb)  # type: ignore[arg-type]
            tags.append(ModelTag(brightness_tag, confidence=brightness_score))

            for color, score in _dominant_colors(mean_rgb):  # type: ignore[arg-type]
                tags.append(ModelTag(f"color-{color}", confidence=score))

            orient_label, orient_score = _orientation_tag(image)
            tags.append(ModelTag(orient_label, confidence=min(1.0, orient_score / 2)))

            for palette, score in _dominant_palette(image):
                tags.append(ModelTag(palette, confidence=score))

            tags = [
                tag
                for tag in tags
                if tag.confidence is None or tag.confidence >= request.confidence_threshold
            ]
            tags = tags[: request.max_tags]

        caption: str | None = None
        if request.generate_captions:
            caption = self._build_caption(image, tags)

        return ModelOutput(caption=caption, tags=tags)

    def _build_caption(self, image: Image.Image, tags: Iterable[ModelTag]) -> str:
        width, height = image.size
        orientation, _ = _orientation_tag(image)
        orientation_phrase = orientation.replace("-", " ")
        palette = ", ".join({tag.value for tag in tags if "color-" in tag.value})
        subject_hint = "abstract scene"
        if palette:
            subject_hint = f"{palette.replace('color-', '')} tones"
        return f"{orientation_phrase} featuring {subject_hint} ({width}x{height})"


def _register() -> None:
    ModelRegistry.register("builtin.simple", SimpleHeuristicModel)


_register()

