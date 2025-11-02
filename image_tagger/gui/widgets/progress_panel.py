"""Reusable widget showing progress and status text."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget


class ProgressPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)

        self._status = QLabel("Idle")

        layout.addWidget(self._progress)
        layout.addWidget(self._status)

    def reset(self) -> None:
        self._progress.setValue(0)
        self._status.setText("Idle")

    def set_busy(self, busy: bool) -> None:
        self._progress.setRange(0, 0 if busy else 100)

    def update_progress(self, current: int, total: int, path: str) -> None:
        if total <= 0:
            self._progress.setRange(0, 0)
            self._status.setText("Working…")
            return
        self._progress.setRange(0, total)
        self._progress.setValue(current)
        self._status.setText(f"Processing {path}")

