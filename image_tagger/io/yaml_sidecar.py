"""Write metadata to YAML sidecar files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class YamlSidecarWriter:
    """Generate human-readable YAML files next to the image."""

    def __init__(self, *, extension: str = "yaml") -> None:
        self.extension = extension.lstrip(".") or "yaml"

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
        with target_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(metadata, handle, sort_keys=False, allow_unicode=False)
        return target_path

