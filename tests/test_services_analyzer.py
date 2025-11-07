"""Tests for the high-level analyzer workflow."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from image_tagger.config import AppConfig, OutputMode
from image_tagger.io.metadata import UnsupportedFormatError
from image_tagger.models.base import (
    ModelCapability,
    ModelError,
    ModelInfo,
    ModelOutput,
    ModelTag,
)
from image_tagger.services.analyzer import ImageAnalyzer


class DummySidecarWriter:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.calls: list[dict[str, object]] = []

    def write(
        self,
        image_path: Path,
        metadata: dict[str, object],
        *,
        output_directory: Path | None = None,
    ):
        target_dir = output_directory or self.base_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / f"{image_path.stem}.yaml"
        output_path.write_text("{}", encoding="utf-8")
        self.calls.append({"path": image_path, "metadata": metadata})
        return output_path


class DummyModel:
    def __init__(self) -> None:
        self._info = ModelInfo(
            identifier="dummy.model",
            display_name="Dummy",
            description="Test double",
            capabilities=(ModelCapability.CAPTION, ModelCapability.TAGS),
        )

    def info(self) -> ModelInfo:
        return self._info

    def load(self) -> None:  # pragma: no cover - nothing to do
        return None

    def analyze(self, image, request):
        tag = ModelTag(value=image.filename.split(".")[0], confidence=0.9)
        return ModelOutput(caption="caption", tags=[tag], extras={"seen": image.filename})


def _create_image(path: Path) -> None:
    Image.new("RGB", (4, 4), color=(123, 222, 111)).save(path)


def test_analyze_paths_runs_model_and_writes_sidecars(tmp_path, monkeypatch):
    img_dir = tmp_path / "inputs"
    img_dir.mkdir()
    a_path = img_dir / "b_image.jpg"
    b_path = img_dir / "a_image.jpg"
    _create_image(a_path)
    _create_image(b_path)

    config = AppConfig(
        model_name="dummy.model",
        output_mode=OutputMode.SIDECAR,
        embed_metadata=False,
        max_concurrency=1,
        output_directory=tmp_path / "sidecars",
    )
    analyzer = ImageAnalyzer(config)

    dummy_model = DummyModel()
    monkeypatch.setattr(
        "image_tagger.services.analyzer.ModelRegistry.get",
        lambda name, config: dummy_model,
    )
    analyzer.metadata_writer = SimpleNamespace(write=lambda *args, **kwargs: True)
    dummy_writer = DummySidecarWriter(config.output_directory)
    analyzer.sidecar_writer = dummy_writer

    progress_calls: list[tuple[int, int, Path]] = []
    results = analyzer.analyze_paths(
        [a_path, b_path],
        progress_callback=lambda current, total, path: progress_calls.append(
            (current, total, path)
        ),
    )

    # Results are sorted by path regardless of submission order.
    assert [res.image_path.name for res in results] == ["a_image.jpg", "b_image.jpg"]
    assert all(res.sidecar_path is not None for res in results)
    assert all(res.extras["model"] == "dummy.model" for res in results)

    # One sidecar written per image with the expected payload.
    assert len(dummy_writer.calls) == 2
    call_images = {call["path"].name for call in dummy_writer.calls}
    assert call_images == {"a_image.jpg", "b_image.jpg"}
    for call in dummy_writer.calls:
        metadata = call["metadata"]
        assert metadata["caption"] == "caption"
        assert metadata["model"] == "dummy.model"
        assert metadata["tags"]

    # Progress callback invoked once per image with sorted paths.
    assert [(c, t, p.name) for c, t, p in progress_calls] == [
        (1, 2, "a_image.jpg"),
        (2, 2, "b_image.jpg"),
    ]


def test_analyze_paths_embed_mode_falls_back_to_sidecar(tmp_path, monkeypatch):
    img_path = tmp_path / "img.jpg"
    _create_image(img_path)
    config = AppConfig(
        model_name="dummy.model",
        output_mode=OutputMode.EMBED,
        embed_metadata=True,
        max_concurrency=1,
    )
    analyzer = ImageAnalyzer(config)
    dummy_model = DummyModel()
    monkeypatch.setattr(
        "image_tagger.services.analyzer.ModelRegistry.get",
        lambda name, config: dummy_model,
    )
    analyzer.metadata_writer = SimpleNamespace(write=lambda *args, **kwargs: False)
    dummy_writer = DummySidecarWriter(tmp_path)
    analyzer.sidecar_writer = dummy_writer

    results = analyzer.analyze_paths([img_path])

    assert results[0].embedded is False
    assert dummy_writer.calls


def test_analyze_paths_empty_returns_empty(tmp_path):
    analyzer = ImageAnalyzer(AppConfig(max_concurrency=1))
    assert analyzer.analyze_paths([]) == []


def test_analyze_target_uses_resolver(monkeypatch, tmp_path):
    analyzer = ImageAnalyzer(AppConfig())
    expected = [tmp_path / "one.jpg"]
    monkeypatch.setattr(
        "image_tagger.services.analyzer.resolve_image_paths",
        lambda **kwargs: expected,
    )

    captured = {}

    def fake_analyze(paths, progress_callback=None):
        captured["paths"] = paths
        return ["ok"]

    monkeypatch.setattr(analyzer, "analyze_paths", fake_analyze)

    result = analyzer.analyze_target(tmp_path / "start")

    assert result == ["ok"]
    assert captured["paths"] == expected


def test_process_single_handles_model_error(tmp_path, monkeypatch):
    img_path = tmp_path / "err.jpg"
    _create_image(img_path)
    analyzer = ImageAnalyzer(AppConfig(model_name="dummy"))

    class ErrorModel(DummyModel):
        def analyze(self, image, request):
            raise ModelError("boom")

    error_model = ErrorModel()
    monkeypatch.setattr(ImageAnalyzer, "_get_model", lambda self: error_model)

    result = analyzer._process_single(img_path)

    assert result.error_message == "boom"
    assert result.tags == []


def test_try_embed_handles_exceptions(tmp_path):
    analyzer = ImageAnalyzer(AppConfig())

    def raise_unsupported(*args, **kwargs):
        raise UnsupportedFormatError("nope")

    analyzer.metadata_writer = SimpleNamespace(write=raise_unsupported)
    assert (
        analyzer._try_embed(tmp_path / "file.jpg", {"caption": "hi", "tags": ("one", "two")})
        is False
    )

    def raise_runtime(*args, **kwargs):
        raise RuntimeError("bad")

    analyzer.metadata_writer = SimpleNamespace(write=raise_runtime)
    assert analyzer._try_embed(tmp_path / "file.jpg", {"caption": None, "tags": []}) is False
