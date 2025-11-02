"""Dialog that exposes application settings with validation."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QWidget,
)

from ...config import AppConfig, OutputMode
from ...models.registry import ModelRegistry


class SettingsDialog(QDialog):
    """Shows a validated form for editing application configuration."""

    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self._original_config = config
        self._config: AppConfig | None = None

        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.model_combo = QComboBox()
        for info in ModelRegistry.list_model_infos():
            self.model_combo.addItem(info.display_name, info.identifier)
        idx = self.model_combo.findData(config.model_name)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

        self.output_mode_combo = QComboBox()
        for mode in OutputMode:
            self.output_mode_combo.addItem(mode.name.title(), mode.value)
        mode_index = self.output_mode_combo.findData(config.output_mode.value)
        self.output_mode_combo.setCurrentIndex(max(mode_index, 0))

        self.recursive_check = QCheckBox("Recurse into sub-folders")
        self.recursive_check.setChecked(config.recursive)

        self.hidden_check = QCheckBox("Include hidden files and folders")
        self.hidden_check.setChecked(config.include_hidden)

        self.captions_check = QCheckBox("Generate captions")
        self.captions_check.setChecked(config.generate_captions)

        self.tags_check = QCheckBox("Generate tags")
        self.tags_check.setChecked(config.generate_tags)

        self.max_tags_spin = QSpinBox()
        self.max_tags_spin.setRange(1, 128)
        self.max_tags_spin.setValue(config.max_tags)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setValue(config.confidence_threshold)

        self.concurrency_spin = QSpinBox()
        self.concurrency_spin.setRange(1, 32)
        self.concurrency_spin.setValue(config.max_concurrency)

        self.sidecar_line = QLineEdit(config.sidecar_extension)
        self.embed_check = QCheckBox("Attempt to embed metadata when possible")
        self.embed_check.setChecked(config.embed_metadata)

        self.output_dir_edit = QLineEdit(
            str(config.output_directory) if config.output_directory else ""
        )
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._select_output_dir)

        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(browse_button)

        self.locale_edit = QLineEdit(config.localization or "")

        form.addRow("Model", self.model_combo)
        form.addRow("Output mode", self.output_mode_combo)
        form.addRow("", self.recursive_check)
        form.addRow("", self.hidden_check)
        form.addRow("", self.captions_check)
        form.addRow("", self.tags_check)
        form.addRow("Max tags", self.max_tags_spin)
        form.addRow("Confidence threshold", self.confidence_spin)
        form.addRow("Workers", self.concurrency_spin)
        form.addRow("Sidecar extension", self.sidecar_line)
        form.addRow("", self.embed_check)
        form.addRow("Output directory", output_dir_layout)
        form.addRow("Locale", self.locale_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        self.setMinimumWidth(420)

    def _select_output_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select output folder")
        if directory:
            self.output_dir_edit.setText(directory)

    def _on_accept(self) -> None:
        data = {
            "model_name": self.model_combo.currentData(),
            "output_mode": self.output_mode_combo.currentData(),
            "recursive": self.recursive_check.isChecked(),
            "include_hidden": self.hidden_check.isChecked(),
            "generate_captions": self.captions_check.isChecked(),
            "generate_tags": self.tags_check.isChecked(),
            "max_tags": self.max_tags_spin.value(),
            "confidence_threshold": self.confidence_spin.value(),
            "max_concurrency": self.concurrency_spin.value(),
            "sidecar_extension": self.sidecar_line.text(),
            "embed_metadata": self.embed_check.isChecked(),
            "output_directory": self._parse_optional_path(self.output_dir_edit.text()),
            "localization": self.locale_edit.text() or None,
        }
        try:
            self._config = AppConfig.model_validate(data)
        except Exception as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return
        self.accept()

    @staticmethod
    def _parse_optional_path(text: str) -> Path | None:
        text = text.strip()
        return Path(text) if text else None

    def config(self) -> AppConfig:
        return self._config or self._original_config

