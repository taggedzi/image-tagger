"""Write metadata to YAML or JSON sidecar files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


class YamlSidecarWriter:
    """Generate human-readable sidecar files next to the image."""

    def __init__(self, *, extension: str = "yaml") -> None:
        self.extension = extension.lstrip(".") or "yaml"
        self._format = self.extension.lower()

    def write(
        self,
        image_path: Path,
        metadata: dict[str, Any],
        *,
        output_directory: Path | None = None,
    ) -> Path:
        target_dir = output_directory or image_path.parent
        target_dir.mkdir(parents=True, exist_ok=True)

        target_path = target_dir / f"{image_path.stem}.{self.extension}"
        if self._format == "json":
            payload = json.dumps(metadata, indent=2, ensure_ascii=True) + "\n"
        else:
            payload = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=False)
        target_path.write_text(payload, encoding="utf-8")
        return target_path
