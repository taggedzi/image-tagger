"""Qt worker objects used to run analysis off the main thread."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from PySide6.QtCore import QObject, QRunnable, Signal

from ..services.analyzer import AnalyzerResult, ImageAnalyzer


class WorkerSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)
    error = Signal(str)


class ProcessingWorker(QRunnable):
    """Runs the heavy lifting on a background worker thread."""

    def __init__(self, analyzer: ImageAnalyzer, paths: Sequence[Path]) -> None:
        super().__init__()
        self.analyzer = analyzer
        self.paths = tuple(paths)
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            results = self.analyzer.analyze_paths(
                self.paths,
                progress_callback=self._emit_progress,
            )
            self.signals.finished.emit(results)
        except Exception as exc:  # pragma: no cover - safety net for GUI worker
            self.signals.error.emit(str(exc))

    def _emit_progress(self, index: int, total: int, path: Path) -> None:
        self.signals.progress.emit(index, total, str(path))

