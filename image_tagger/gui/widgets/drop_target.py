"""Widget that accepts files/directories via drag & drop."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDragLeaveEvent,
    QDropEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class DropTargetWidget(QWidget):
    files_dropped = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("DropTarget")
        self._active = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)

        self._label = QLabel("Drag and drop images or folders here")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        self._label.setStyleSheet("color: #f5f5f5;")
        layout.addWidget(self._label)

        self.setToolTip("Drop files or directories to queue them for tagging.")

    def paintEvent(self, event: QPaintEvent | None = None) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        background = QColor(18, 18, 22, 220)
        painter.fillRect(rect, background)

        border_color = QColor(111, 118, 134) if not self._active else QColor(70, 112, 187)
        pen = QPen(border_color, 2, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, 16, 16)

        super().paintEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._active = True
            self.update()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = [url for url in event.mimeData().urls() if url.isLocalFile()]
        if not urls:
            event.ignore()
            return
        paths = [Path(url.toLocalFile()) for url in urls]
        self.files_dropped.emit(paths)
        event.acceptProposedAction()
        self._active = False
        self.update()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        if self._active:
            self._active = False
            self.update()
