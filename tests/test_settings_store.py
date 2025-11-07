"""Tests for the settings store helpers."""

from __future__ import annotations

import os

import pytest
from image_tagger.config import AppConfig
from image_tagger.settings_store import SettingsStore, default_settings_path


def test_default_settings_path_respects_xdg(monkeypatch, tmp_path):
    if os.name == "nt":
        pytest.skip("Posix-only test; Windows path semantics differ.")
    monkeypatch.setattr("image_tagger.settings_store.os.name", "posix")
    config_root = tmp_path / "xdg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_root))

    resolved = default_settings_path()

    assert resolved == config_root / "image_tagger" / "settings.yaml"


def test_default_settings_path_windows(monkeypatch, tmp_path):
    monkeypatch.setattr("image_tagger.settings_store.os.name", "nt")
    appdata = tmp_path / "AppData" / "Roaming"
    monkeypatch.setenv("APPDATA", str(appdata))

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
