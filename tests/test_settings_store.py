"""Tests for the settings store helpers."""

from __future__ import annotations

from types import SimpleNamespace

from image_tagger.config import AppConfig
from image_tagger.settings_store import SettingsStore, default_settings_path


def _fake_os(name: str, **env: str) -> SimpleNamespace:
    def getenv(key: str, default=None):
        return env.get(key, default)

    return SimpleNamespace(name=name, getenv=getenv)


def test_default_settings_path_respects_xdg(monkeypatch, tmp_path):
    config_root = tmp_path / "xdg"
    fake_os = _fake_os("posix", XDG_CONFIG_HOME=str(config_root))
    monkeypatch.setattr("image_tagger.settings_store.os", fake_os)

    resolved = default_settings_path()

    assert resolved == config_root / "image_tagger" / "settings.yaml"


def test_default_settings_path_windows(monkeypatch, tmp_path):
    appdata = tmp_path / "AppData" / "Roaming"
    fake_os = _fake_os("nt", APPDATA=str(appdata))
    monkeypatch.setattr("image_tagger.settings_store.os", fake_os)

    resolved = default_settings_path()

    assert resolved == appdata / "image_tagger" / "settings.yaml"


def test_settings_store_round_trip(tmp_path):
    target_path = tmp_path / "settings.yaml"
    store = SettingsStore(path=target_path)
    original = AppConfig(model_name="demo.model", recursive=False, include_hidden=True)

    store.save(original)
    loaded = store.load()

    assert loaded.model_name == "demo.model"
    assert loaded.recursive is False
    assert loaded.include_hidden is True
    assert target_path.exists()


def test_settings_store_loads_defaults_when_missing(tmp_path):
    store = SettingsStore(path=tmp_path / "missing.yaml")

    config = store.load()

    assert isinstance(config, AppConfig)
    assert config == AppConfig()
