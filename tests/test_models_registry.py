"""Tests for the dynamic model registry."""

from __future__ import annotations

import pytest

from image_tagger.config import AppConfig
from image_tagger.models.base import ModelCapability, ModelInfo, TaggingModel
from image_tagger.models.registry import ModelRegistry


class DummyModel(TaggingModel):
    def __init__(self, *, called_with: list[AppConfig | None]) -> None:
        self.called_with = called_with

    def info(self) -> ModelInfo:
        return ModelInfo(
            identifier="dummy",
            display_name="Dummy",
            description="",
            capabilities=(ModelCapability.CAPTION,),
        )

    def load(self) -> None:  # pragma: no cover - nothing to load
        return None

    def analyze(self, image, request):  # pragma: no cover - not needed
        raise NotImplementedError


@pytest.fixture(autouse=True)
def reset_registry():
    original = ModelRegistry._factories.copy()
    original_bootstrapped = ModelRegistry._bootstrap_complete
    yield
    ModelRegistry._factories = original
    ModelRegistry._bootstrap_complete = original_bootstrapped


def test_register_and_list_models(monkeypatch):
    ModelRegistry._factories.clear()
    ModelRegistry._bootstrap_complete = True

    def factory():
        return DummyModel(called_with=[])

    ModelRegistry.register("demo", factory)
    infos = ModelRegistry.list_model_infos()
    assert infos[0].identifier == "dummy"


def test_get_passes_config_and_handles_missing(monkeypatch):
    ModelRegistry._factories.clear()
    ModelRegistry._bootstrap_complete = True
    captured: list[AppConfig | None] = []

    def factory(config: AppConfig | None = None):
        captured.append(config)
        return DummyModel(called_with=captured)

    ModelRegistry.register("demo", factory)

    config = AppConfig(model_name="demo")
    instance = ModelRegistry.get("demo", config=config)
    assert captured[0] == config
    assert isinstance(instance, DummyModel)

    with pytest.raises(KeyError):
        ModelRegistry.get("missing")


def test_get_handles_factories_without_config():
    ModelRegistry._factories.clear()
    ModelRegistry._bootstrap_complete = True
    called = []

    def factory():
        called.append("ok")
        return DummyModel(called_with=[])

    ModelRegistry.register("demo", factory)
    ModelRegistry.get("demo", config=AppConfig())
    assert called == ["ok"]


def test_ensure_bootstrapped_imports_modules(monkeypatch):
    called = []

    def fake_import(name):
        called.append(name)

    monkeypatch.setattr("image_tagger.models.registry.import_module", fake_import)
    ModelRegistry._bootstrap_complete = False
    ModelRegistry._factories.clear()

    ModelRegistry.ensure_bootstrapped()

    assert "image_tagger.models.blip" in called
    assert ModelRegistry._bootstrap_complete is True


def test_unregister_and_list(monkeypatch):
    ModelRegistry._factories.clear()
    ModelRegistry._bootstrap_complete = True
    ModelRegistry.register("demo", lambda: DummyModel(called_with=[]))
    ModelRegistry.unregister("demo")
    assert ModelRegistry._factories == {}


def test_instantiate_factory_handles_import_error(monkeypatch):
    def fake_import(name):
        raise ImportError("boom")

    monkeypatch.setattr("image_tagger.models.registry.import_module", fake_import)
    ModelRegistry._bootstrap_complete = False
    ModelRegistry.ensure_bootstrapped()
    assert ModelRegistry._bootstrap_complete is True


def test_instantiate_factory_fallback(monkeypatch):
    called = []

    def factory():
        called.append("ok")
        return DummyModel(called_with=[])

    instance = ModelRegistry._instantiate_factory(factory, config=AppConfig())
    assert isinstance(instance, DummyModel)
    assert called == ["ok"]
