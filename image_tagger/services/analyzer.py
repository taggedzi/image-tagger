"""Core service orchestrating image analysis and metadata persistence."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

from PIL import Image

from ..config import AppConfig, OutputMode
from ..io.metadata import MetadataWriter, UnsupportedFormatError
from ..io.yaml_sidecar import YamlSidecarWriter
from ..models.base import AnalysisRequest, ModelError, ModelOutput, TaggingModel
from ..models.registry import ModelRegistry
from ..utils.paths import resolve_image_paths
from ..utils.text import slugify_filename

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, Path], None]


@dataclass(slots=True)
class AnalyzerResult:
    """Summary of processing work for a single image."""

    image_path: Path
    suggested_filename: str | None = None
    applied_filename: str | None = None
    caption: str | None = None
    tags: list[str] = field(default_factory=list)
    embedded: bool = False
    sidecar_path: Path | None = None
    extras: dict[str, object] = field(default_factory=dict)
    error_message: str | None = None


class ImageAnalyzer:
    """High-level orchestration for the tagging workflow."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._model: TaggingModel | None = None
        self._model_lock = Lock()
        self.metadata_writer = MetadataWriter()
        self.sidecar_writer = YamlSidecarWriter(extension=config.sidecar_extension)

    def analyze_target(
        self,
        target: Path,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> list[AnalyzerResult]:
        """Process a single image or an entire directory tree."""
        image_paths = resolve_image_paths(
            start=target,
            recursive=self.config.recursive,
            include_hidden=self.config.include_hidden,
        )
        return self.analyze_paths(image_paths, progress_callback=progress_callback)

    def analyze_paths(
        self,
        image_paths: Sequence[Path],
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> list[AnalyzerResult]:
        if not image_paths:
            return []

        sorted_paths = sorted(image_paths)
        total = len(sorted_paths)
        results: list[AnalyzerResult] = []

        def _worker(path: Path) -> AnalyzerResult:
            try:
                return self._process_single(path)
            except Exception as exc:  # pragma: no cover - defensive error handling
                logger.exception("Failed to analyze %s", path)
                return AnalyzerResult(
                    image_path=path,
                    suggested_filename=None,
                    applied_filename=None,
                    caption=None,
                    tags=[],
                    error_message=str(exc),
                )

        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
            futures = {executor.submit(_worker, path): path for path in sorted_paths}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                if progress_callback:
                    progress_callback(index, total, futures[future])

        results.sort(key=lambda item: item.image_path)
        return results

    def _build_request(self) -> AnalysisRequest:
        return AnalysisRequest(
            generate_captions=self.config.generate_captions,
            generate_tags=self.config.generate_tags,
            max_tags=self.config.max_tags,
            confidence_threshold=self.config.confidence_threshold,
            locale=self.config.localization,
        )

    def _process_single(self, image_path: Path) -> AnalyzerResult:
        logger.debug("Analyzing %s", image_path)
        request = self._build_request()

        model = self._get_model()

        with Image.open(image_path) as img:
            try:
                output = model.analyze(img, request)
            except ModelError as exc:
                identifier = model.info().identifier
                logger.warning("Model '%s' could not analyze %s: %s", identifier, image_path, exc)
                return AnalyzerResult(
                    image_path=image_path,
                    suggested_filename=None,
                    applied_filename=None,
                    caption=None,
                    tags=[],
                    embedded=False,
                    sidecar_path=None,
                    extras={"model": identifier},
                    error_message=str(exc),
                )

        prepared = self._prepare_output(output)
        model_identifier = model.info().identifier

        rename_result = self._apply_filename_strategy(image_path, prepared)
        final_path = rename_result["path"]
        suggested_filename = rename_result["suggested"]
        applied_filename = rename_result["applied"]

        embedded = False
        sidecar_path: Path | None = None

        if self.config.output_mode == OutputMode.EMBED:
            embedded = self._try_embed(final_path, prepared)
            if not embedded:
                sidecar_path = self._write_sidecar(final_path, prepared, model_identifier)
        else:
            sidecar_path = self._write_sidecar(final_path, prepared, model_identifier)
            if self.config.embed_metadata:
                embedded = self._try_embed(final_path, prepared)

        return AnalyzerResult(
            image_path=final_path,
            suggested_filename=suggested_filename,
            applied_filename=applied_filename,
            caption=prepared["caption"],
            tags=list(prepared["tags"]),
            embedded=embedded,
            sidecar_path=sidecar_path,
            extras={
                **prepared.get("extras", {}),
                "model": model_identifier,
                "suggested_filename": suggested_filename,
                "applied_filename": applied_filename,
                "original_path": str(image_path),
            },
        )

    def _prepare_output(self, output: ModelOutput) -> dict[str, object]:
        trimmed = output.truncated(max_tags=self.config.max_tags)
        suggested = None
        raw_suggested = None
        if self.config.suggest_filenames and trimmed.filename:
            raw_suggested = trimmed.filename
            suggested = slugify_filename(trimmed.filename)
        data = {
            "caption": trimmed.caption,
            "tags": [tag.value for tag in trimmed.tags],
            "tag_details": [tag.as_dict() for tag in trimmed.tags],
            "suggested_filename": suggested,
            "raw_suggested_filename": raw_suggested,
            "extras": trimmed.extras,
        }
        return data

    def _apply_filename_strategy(
        self, image_path: Path, payload: dict[str, object]
    ) -> dict[str, object]:
        suggested = payload.get("suggested_filename")
        raw_suggested = payload.get("raw_suggested_filename")
        if not self.config.suggest_filenames or not isinstance(suggested, str):
            return {"path": image_path, "suggested": None, "applied": None}

        if image_path.stem == suggested:
            target = image_path
        else:
            target = self._dedupe_target_path(image_path, suggested)
        if target == image_path or not self.config.auto_rename_files:
            return {
                "path": image_path,
                "suggested": suggested,
                "applied": None,
            }

        try:
            image_path.rename(target)
            return {"path": target, "suggested": suggested, "applied": target.name}
        except Exception as exc:  # pragma: no cover - filesystem issues are surfaced to users
            logger.warning("Failed to rename %s to %s: %s", image_path, target, exc)
            return {
                "path": image_path,
                "suggested": suggested,
                "applied": None,
                "error": str(exc),
                "raw_suggested": raw_suggested,
            }

    @staticmethod
    def _dedupe_target_path(image_path: Path, stem: str) -> Path:
        suffix = image_path.suffix
        base = image_path.with_name(stem + suffix)
        if not base.exists():
            return base

        counter = 1
        while True:
            candidate = image_path.with_name(f"{stem}-{counter}{suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    def _try_embed(self, image_path: Path, payload: dict[str, object]) -> bool:
        caption = payload.get("caption")
        tags = payload.get("tags", [])
        if not isinstance(tags, list):
            tags = list(tags)

        try:
            return self.metadata_writer.write(
                image_path,
                caption=caption if isinstance(caption, str) else None,
                tags=[str(tag) for tag in tags],
                overwrite_existing=self.config.overwrite_embedded_metadata,
            )
        except UnsupportedFormatError as exc:
            logger.info("%s; falling back to sidecar for %s", exc, image_path)
            return False
        except Exception:
            logger.exception("Failed to embed metadata into %s", image_path)
            return False

    def _write_sidecar(self, image_path: Path, payload: dict[str, object], model_id: str) -> Path:
        metadata = {
            "image": str(image_path),
            "model": model_id,
            "caption": payload.get("caption"),
            "tags": payload.get("tag_details"),
            "extras": payload.get("extras"),
        }
        return self.sidecar_writer.write(
            image_path,
            metadata,
            output_directory=self.config.output_directory,
        )

    def _get_model(self) -> TaggingModel:
        with self._model_lock:
            if self._model is None or self._model.info().identifier != self.config.model_name:
                logger.info("Loading model '%s'...", self.config.model_name)
                self._model = ModelRegistry.get(self.config.model_name, config=self.config)
                logger.info("Model '%s' ready.", self.config.model_name)
        return self._model
