"""Additional tests for AppConfig validation."""

from __future__ import annotations

import json

import pytest
from image_tagger.config import AppConfig, OutputMode


def test_sidecar_extension_is_normalised():
    config = AppConfig(sidecar_extension=".Yml")
    assert config.sidecar_extension == "Yml".lstrip(".")


def test_sidecar_extension_requires_value():
    with pytest.raises(ValueError):
        AppConfig(output_mode=OutputMode.SIDECAR, sidecar_extension="")


def test_remote_base_url_is_trimmed():
    config = AppConfig(remote_base_url=" http://example.com/base/ ")
    assert config.remote_base_url == "http://example.com/base"


def test_load_and_save_round_trip(tmp_path):
    path = tmp_path / "config.yaml"
    original = AppConfig(
        model_name="custom",
        remote_base_url="http://localhost:1234/",
        sidecar_extension="json",
        output_mode=OutputMode.SIDECAR,
    )
    original.save(path)

    loaded = AppConfig.load(path)
    assert loaded.model_name == "custom"
    assert loaded.remote_base_url == "http://localhost:1234"
    assert loaded.sidecar_extension == "json"


def test_load_rejects_invalid_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"max_tags": -1}), encoding="utf-8")

    with pytest.raises(ValueError):
        AppConfig.load(path)
