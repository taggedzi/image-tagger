import pytest

from image_tagger.config import AppConfig
from image_tagger.models.base import AnalysisRequest
from image_tagger.models.vision_remote import OllamaVisionModel


def _analysis_request(max_tags: int = 5) -> AnalysisRequest:
    return AnalysisRequest(
        generate_captions=True,
        generate_tags=True,
        max_tags=max_tags,
        confidence_threshold=0.0,
        locale=None,
    )


def test_remote_base_url_is_normalised():
    config = AppConfig(remote_base_url="http://localhost:11434/")
    assert config.remote_base_url == "http://localhost:11434"


def test_remote_base_url_requires_scheme():
    with pytest.raises(ValueError):
        AppConfig(remote_base_url="localhost:11434")


def test_parse_json_response_handles_code_fences():
    model = OllamaVisionModel(AppConfig())
    payload = model._parse_json_response('```json\n{"caption":"A cat","tags":["cat"]}\n```')
    assert payload["caption"] == "A cat"
    assert payload["tags"] == ["cat"]


def test_normalise_tags_returns_unique_lowercase():
    model = OllamaVisionModel(AppConfig())
    tags = model._normalize_tags("Cat,  dog,cat\nBIRD", _analysis_request())
    assert [tag.value for tag in tags] == ["cat", "dog", "bird"]


def test_parse_json_response_merges_multiple_objects():
    model = OllamaVisionModel(AppConfig())
    raw = '{"caption":"one"},\n{"tags":["alpha","beta"]}'
    payload = model._parse_json_response(raw)
    assert payload == {"caption": "one", "tags": ["alpha", "beta"]}


def test_is_vision_candidate_by_keyword():
    model = OllamaVisionModel(AppConfig())
    assert model._is_vision_candidate("llava:13b", {})
    assert not model._is_vision_candidate("llama2:7b", {})


def test_discover_remote_models_filters(monkeypatch):
    model = OllamaVisionModel(AppConfig())

    def fake_fetch(self):
        return [
            ("llava:13b", {"families": ["vision"]}),
            ("llama2:7b", {"families": ["text"]}),
        ]

    monkeypatch.setattr(OllamaVisionModel, "_fetch_remote_model_metadata", fake_fetch)
    discovered = model.discover_remote_models()
    assert discovered == ["llava:13b"]
