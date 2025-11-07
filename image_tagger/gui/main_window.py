"""Main Qt window implementing the user interface."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PySide6 import QtCore
from PySide6.QtCore import QThreadPool
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import AppConfig
from ..services.analyzer import AnalyzerResult, ImageAnalyzer
from ..settings_store import SettingsStore
from ..utils.paths import is_image_file, resolve_image_paths
from .widgets.about_dialog import AboutDialog
from .widgets.drop_target import DropTargetWidget
from .widgets.progress_panel import ProgressPanel
from .widgets.settings_form import SettingsDialog
from .workers import ProcessingWorker


class MainWindow(QMainWindow):
    """Primary application window."""

    def __init__(self, settings_store: SettingsStore | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Image Tagger")
        self.resize(1080, 720)

        self.settings_store = settings_store or SettingsStore()
        self.config: AppConfig = self.settings_store.load()
        self.analyzer = ImageAnalyzer(self.config)
        self.thread_pool = QThreadPool()
        self._current_worker: ProcessingWorker | None = None
        self._collected_paths: list[Path] = []
        self._cursor_busy = False

        self._build_ui()
        self._rebuild_status_bar()
        self._build_menus()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)
        self.setCentralWidget(central)

        self.info_label = QLabel(
            "Select images or folders, or drop them below to begin tagging."
        )
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)

        self.open_files_btn = QPushButton("Select Images…")
        self.open_files_btn.clicked.connect(self._choose_files)
        self.open_files_btn.setToolTip("Choose one or more image files to process.")

        self.open_folder_btn = QPushButton("Select Folder…")
        self.open_folder_btn.clicked.connect(self._choose_directory)
        self.open_folder_btn.setToolTip("Choose a folder to scan for supported images.")

        self.settings_btn = QPushButton("Settings…")
        self.settings_btn.clicked.connect(self._open_settings)

        self.clear_btn = QPushButton("Clear Results")
        self.clear_btn.clicked.connect(self._clear_results)

        button_row.addWidget(self.open_files_btn)
        button_row.addWidget(self.open_folder_btn)
        button_row.addStretch()
        button_row.addWidget(self.settings_btn)
        button_row.addWidget(self.clear_btn)

        layout.addLayout(button_row)

        self.drop_target = DropTargetWidget()
        self.drop_target.files_dropped.connect(self._handle_dropped_paths)
        layout.addWidget(self.drop_target)

        self.progress_panel = ProgressPanel()
        layout.addWidget(self.progress_panel)

        self.results_view = QTreeWidget()
        self.results_view.setColumnCount(4)
        self.results_view.setHeaderLabels(["Image", "Caption", "Tags", "Output"])
        self.results_view.setRootIsDecorated(False)
        self.results_view.setAlternatingRowColors(True)
        layout.addWidget(self.results_view, stretch=1)

        status = QStatusBar()
        self.setStatusBar(status)

    def _build_menus(self) -> None:
        menu_bar = self.menuBar()
        help_menu = menu_bar.addMenu("&Help")
        about_action = help_menu.addAction("About Image Tagger…")
        about_action.triggered.connect(self._show_about_dialog)

    def _rebuild_status_bar(self) -> None:
        status = self.statusBar()
        status.showMessage(
            f"Model: {self.config.model_name} • Output: {self.config.output_mode.value}"
        )

    # --- Event handlers -------------------------------------------------

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select images",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tiff)",
        )
        if paths:
            self._queue_paths(Path(path) for path in paths)

    def _choose_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select folder")
        if directory:
            self._queue_paths([Path(directory)])

    def _handle_dropped_paths(self, paths: list[Path]) -> None:
        self._queue_paths(paths)

    def _queue_paths(self, paths: Iterable[Path]) -> None:
        candidates: list[Path] = []
        for path in paths:
            if path.is_dir():
                try:
                    discovered = resolve_image_paths(
                        path,
                        recursive=self.config.recursive,
                        include_hidden=self.config.include_hidden,
                    )
                except FileNotFoundError:
                    continue
                candidates.extend(discovered)
            elif path.is_file():
                if is_image_file(path):
                    candidates.append(path)

        if not candidates:
            QMessageBox.information(self, "No images found", "Nothing to process.")
            return

        self._start_analysis(candidates)

    def _start_analysis(self, paths: list[Path]) -> None:
        if self._current_worker is not None:
            QMessageBox.warning(
                self,
                "Processing already in progress",
                "Please wait for the current batch to finish.",
            )
            return

        self._collected_paths = paths
        self.progress_panel.set_busy(True)
        self.progress_panel.update_progress(0, len(paths), "")
        if self.config.model_name.startswith("remote."):
            self.progress_panel.show_hint(
                "Contacting remote model… the first run may take a minute while the model loads."
            )
        self._set_busy_cursor(True)
        self._toggle_controls(False)

        worker = ProcessingWorker(self.analyzer, paths)
        worker.signals.progress.connect(self._on_worker_progress)
        worker.signals.finished.connect(self._on_worker_finished)
        worker.signals.error.connect(self._on_worker_error)

        self._current_worker = worker
        self.thread_pool.start(worker)

    def _toggle_controls(self, enabled: bool) -> None:
        self.open_files_btn.setEnabled(enabled)
        self.open_folder_btn.setEnabled(enabled)
        self.settings_btn.setEnabled(enabled)
        self.clear_btn.setEnabled(enabled)

    def _on_worker_progress(self, current: int, total: int, path: str) -> None:
        self.progress_panel.update_progress(current, total, path)

    def _on_worker_error(self, message: str) -> None:
        QMessageBox.critical(self, "Processing error", message)
        self._finalise_worker([])

    def _on_worker_finished(self, results: list[AnalyzerResult]) -> None:
        self._finalise_worker(results)

    def _finalise_worker(self, results: list[AnalyzerResult]) -> None:
        self._current_worker = None
        self.progress_panel.set_busy(False)
        self._set_busy_cursor(False)
        self._toggle_controls(True)
        if results:
            self._render_results(results)
            self.progress_panel.update_progress(len(results), len(results), "Done")
        else:
            self.progress_panel.reset()

    def _set_busy_cursor(self, active: bool) -> None:
        app = QApplication.instance()
        if app is None:
            return
        if active and not self._cursor_busy:
            QGuiApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            self._cursor_busy = True
        elif not active and self._cursor_busy:
            QGuiApplication.restoreOverrideCursor()
            self._cursor_busy = False

    def _render_results(self, results: list[AnalyzerResult]) -> None:
        self.results_view.clear()
        for result in results:
            tags = ", ".join(result.tags)
            output = "Embedded" if result.embedded else ""
            if result.sidecar_path:
                if output:
                    output += " + "
                output += str(result.sidecar_path)
            item = QTreeWidgetItem(
                [
                    str(result.image_path),
                    result.caption or "",
                    tags,
                    output or "None",
                ]
            )
            if result.error_message:
                item.setToolTip(0, result.error_message)
            self.results_view.addTopLevelItem(item)
        self.results_view.resizeColumnToContents(0)

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        new_config = dialog.config()
        self._apply_new_config(new_config)

    def _apply_new_config(self, config: AppConfig) -> None:
        self.config = config
        self.analyzer = ImageAnalyzer(config)
        self.settings_store.save(config)
        self._rebuild_status_bar()

    def _clear_results(self) -> None:
        self.results_view.clear()
        self.progress_panel.reset()

    def _show_about_dialog(self) -> None:
        dialog = AboutDialog(self.config, self)
        dialog.exec()
