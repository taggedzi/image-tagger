"""Tests for the CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from image_tagger import OutputMode
from image_tagger.__main__ import main as cli_main
from image_tagger.config import AppConfig
from image_tagger.models.base import ModelCapability, ModelInfo


def test_cli_lists_models(monkeypatch, capsys):
    model_info = ModelInfo(
        identifier="demo",
        display_name="Demo",
        description="Example",
        capabilities=(ModelCapability.CAPTION, ModelCapability.TAGS),
        tags=("demo",),
    )
    monkeypatch.setattr(
        "image_tagger.__main__.ModelRegistry.list_model_infos",
        lambda: [model_info],
    )

    cli_main(["--list-models"])

    captured = capsys.readouterr().out.strip()
    payload = json.loads(captured)
    assert payload[0]["identifier"] == "demo"
    assert payload[0]["capabilities"] == ["caption", "tags"]


def test_cli_requires_input_in_headless_mode():
    with pytest.raises(SystemExit):
        cli_main(["--headless"])


def test_cli_runs_headless_job(monkeypatch, tmp_path, capsys):
    image_path = tmp_path / "image.jpg"
    image_path.write_text("fake", encoding="utf-8")

    class DummyStore:
        def __init__(self) -> None:
            self.loaded = False

        def load(self) -> AppConfig:
            self.loaded = True
            return AppConfig()

    class DummyAnalyzer:
        def __init__(self, config: AppConfig) -> None:
            self.config = config

        def analyze_target(self, path: Path):
            return [
                SimpleNamespace(
                    image_path=path,
                    caption="A caption",
                    tags=["alpha", "beta"],
                    embedded=True,
                    sidecar_path=None,
                    error_message=None,
                )
            ]

    monkeypatch.setattr("image_tagger.__main__.SettingsStore", DummyStore)
    monkeypatch.setattr("image_tagger.__main__.ImageAnalyzer", DummyAnalyzer)

    cli_main(["--headless", "--input", str(image_path), "--output-mode", OutputMode.SIDECAR.value])

    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["path"] == str(image_path)
    assert payload[0]["embedded"] is True
