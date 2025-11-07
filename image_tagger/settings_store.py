"""Persistence helpers for user configuration."""

from __future__ import annotations

import os
from pathlib import Path

from .config import AppConfig


class SettingsStore:
    """Load and save application settings to a well-known path."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or default_settings_path()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> AppConfig:
        if not self._path.exists():
            return AppConfig()
        return AppConfig.load(self._path)

    def save(self, config: AppConfig) -> None:
        config.save(self._path)


def default_settings_path() -> Path:
    if os.name == "nt":
        base = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base.expanduser() / "image_tagger" / "settings.yaml"
