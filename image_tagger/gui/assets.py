"""Static asset helpers for the GUI layer."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QIcon

APP_ICON_PATH = Path(__file__).resolve().parents[2] / "resources" / "image-tagger-icon.png"


def load_app_icon() -> QIcon | None:
    """Return the application icon if the resource exists."""
    if APP_ICON_PATH.exists():
        return QIcon(str(APP_ICON_PATH))
    return None
