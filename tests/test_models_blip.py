"""Tests for the BLIP captioning model wrapper."""

from __future__ import annotations

import pytest

from image_tagger.models import blip
from image_tagger.models.base import AnalysisRequest, ModelError


def _make_model() -> blip.BlipCaptioningModel:
    return blip.BlipCaptioningModel(
        model_id="demo",
        identifier="caption.blip",
        display_name="Demo",
    )


def test_extract_tags_filters_stop_words():
    model = _make_model()
    tags = model._extract_tags_from_caption(  # pylint: disable=protected-access
        "The cat and dog stroll under a bridge near the river",
        max_count=3,
        threshold=0.0,
    )
    assert [tag.value for tag in tags] == ["cat", "dog", "stroll"]


def test_load_requires_dependencies(monkeypatch):
    monkeypatch.setattr(blip, "pipeline", None)
    monkeypatch.setattr(blip, "torch", None)
    model = _make_model()

    with pytest.raises(ModelError):
        model.load()


def test_analyze_uses_pipeline(monkeypatch):
    outputs = [{"generated_text": "A scenic bridge"}]

    class DummyPipeline:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, image):
            return outputs

    monkeypatch.setattr(blip, "pipeline", DummyPipeline)
    monkeypatch.setattr(blip, "torch", object())
    monkeypatch.setattr(blip, "detect_torch_device", lambda: ("cpu", "cpu"))

    model = _make_model()
    model.load()
    request = AnalysisRequest(
        generate_captions=True,
        generate_tags=True,
        max_tags=2,
        confidence_threshold=0.1,
        locale=None,
    )

    result = model.analyze(None, request)  # type: ignore[arg-type]
    assert result.caption == "A scenic bridge"
    assert result.tags
