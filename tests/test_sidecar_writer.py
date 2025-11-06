from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from image_tagger.io.yaml_sidecar import YamlSidecarWriter


@pytest.fixture()
def sample_metadata() -> dict[str, object]:
    return {
        "image": "/tmp/example.jpg",
        "model": "demo",
        "caption": "A test caption",
        "tags": [{"value": "one"}, {"value": "two"}],
        "extras": {"foo": "bar"},
    }


def test_writes_yaml_sidecar(tmp_path: Path, sample_metadata: dict[str, object]) -> None:
    image_path = tmp_path / "image.png"
    writer = YamlSidecarWriter(extension="yaml")

    output_path = writer.write(image_path, sample_metadata)

    assert output_path.suffix == ".yaml"
    loaded = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert loaded == sample_metadata


def test_writes_json_sidecar(tmp_path: Path, sample_metadata: dict[str, object]) -> None:
    image_path = tmp_path / "image.png"
    writer = YamlSidecarWriter(extension="json")

    output_path = writer.write(image_path, sample_metadata)

    assert output_path.suffix == ".json"
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == sample_metadata
