"""Application bootstrap for the Qt-based GUI."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .assets import load_app_icon
from .main_window import MainWindow


def run_app() -> None:
    """Launch the GUI application."""
    app = QApplication.instance() or QApplication(sys.argv)
    icon = load_app_icon()
    if icon is not None:
        app.setWindowIcon(icon)
    window = MainWindow()
    if icon is not None:
        window.setWindowIcon(icon)
    window.show()
    app.exec()
