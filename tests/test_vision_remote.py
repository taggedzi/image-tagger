import json
from types import SimpleNamespace

import pytest
from image_tagger.config import AppConfig
from image_tagger.models.base import AnalysisRequest, ModelError
from image_tagger.models.vision_remote import (
    BaseRemoteVisionModel,
    OllamaVisionModel,
    _encode_image,
    _unique,
)
from PIL import Image


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


def test_headers_include_api_key():
    model = OllamaVisionModel(AppConfig(remote_api_key="secret"))
    assert model._headers()["Authorization"] == "Bearer secret"


def test_session_post_handles_timeout(monkeypatch):
    class TimeoutSession:
        @staticmethod
        def post(*args, **kwargs):
            raise TimeoutError()

    config = AppConfig()
    model = OllamaVisionModel(config)
    model._session = TimeoutSession()
    fake_requests = SimpleNamespace(
        exceptions=SimpleNamespace(Timeout=TimeoutError),
    )
    monkeypatch.setattr("image_tagger.models.vision_remote.requests", fake_requests)

    with pytest.raises(ModelError):
        model._session_post("http://example", {}, timeout=1)


def test_session_post_raises_on_http_error(monkeypatch):
    class DummyResponse:
        status_code = 500
        text = "boom"

    model = OllamaVisionModel(AppConfig())
    model._session = SimpleNamespace(post=lambda *args, **kwargs: DummyResponse())
    fake_requests = SimpleNamespace(
        exceptions=SimpleNamespace(Timeout=TimeoutError),
    )
    monkeypatch.setattr("image_tagger.models.vision_remote.requests", fake_requests)

    with pytest.raises(ModelError):
        model._session_post("http://example", {}, timeout=1)


class DummyRemoteModel(BaseRemoteVisionModel):
    def __init__(self, response: str):
        super().__init__(
            identifier="remote.dummy",
            display_name="Dummy Remote",
            description="",
            backend="dummy",
            config=AppConfig(),
            tags=("dummy",),
        )
        self._response = response
        self.prompts: list[str] = []

    def _call_backend(self, encoded_image: str, prompt: str, request: AnalysisRequest) -> str:
        self.prompts.append(prompt)
        return self._response


def test_encode_image_and_unique_helpers():
    image = Image.new("RGB", (4, 4), color=(10, 20, 30))
    payload = _encode_image(image)
    assert isinstance(payload, str)
    assert len(payload) > 0
    assert _unique(["a", "b", "a", "c"]) == ["a", "b", "c"]


def test_base_remote_analyze_parses_backend(tmp_path):
    model = DummyRemoteModel(json.dumps({"caption": "Hi", "tags": ["Tree", "tree", "Sky"]}))
    request = _analysis_request(max_tags=3)
    image = Image.new("RGB", (2, 2))

    result = model.analyze(image, request)

    assert result.caption == "Hi"
    assert [tag.value for tag in result.tags] == ["tree", "sky"]
    assert model.prompts


def test_base_remote_parses_filename():
    model = DummyRemoteModel(json.dumps({"caption": "Hi", "tags": [], "filename": "sunset"}))
    request = _analysis_request(max_tags=3)
    image = Image.new("RGB", (2, 2))

    result = model.analyze(image, request)

    assert result.filename == "sunset"


def test_base_remote_prompt_varies_with_flags():
    payload = json.dumps({"caption": None, "tags": []})
    model = DummyRemoteModel(payload)
    request = AnalysisRequest(
        generate_captions=False,
        generate_tags=False,
        max_tags=1,
        confidence_threshold=0.0,
        locale=None,
    )
    image = Image.new("RGB", (1, 1))

    model.analyze(image, request)

    prompt = model.prompts[0]
    assert "Set the caption field to null" in prompt
    assert "Return an empty array for tags" in prompt


def test_session_helpers_success_paths(monkeypatch):
    class DummyResponse:
        status_code = 200
        text = "{}"

        def json(self):
            return {}

    model = DummyRemoteModel(json.dumps({"caption": "ok", "tags": []}))
    model._session = SimpleNamespace(
        post=lambda *args, **kwargs: DummyResponse(),
        get=lambda *args, **kwargs: DummyResponse(),
    )

    assert model._session_post("http://example", {}) is not None
    assert model._session_get("http://example") is not None


def test_discover_remote_models_handles_errors(monkeypatch):
    class BrokenRemote(DummyRemoteModel):
        def _fetch_remote_model_metadata(self):
            raise ModelError("boom")

    model = BrokenRemote("{}")
    assert model.discover_remote_models() == []


def test_fetch_remote_model_metadata(monkeypatch):
    model = OllamaVisionModel(AppConfig())

    class DummyResponse:
        status_code = 200
        text = ""

        def json(self):
            return {"models": [{"model": "llava:13b", "details": {}}]}

    fake_session = SimpleNamespace(get=lambda *args, **kwargs: DummyResponse())
    model._session = fake_session
    fake_requests = SimpleNamespace(
        exceptions=SimpleNamespace(Timeout=TimeoutError),
    )
    monkeypatch.setattr("image_tagger.models.vision_remote.requests", fake_requests)

    result = model._fetch_remote_model_metadata()
    assert result[0][0] == "llava:13b"
