"""Dialog that exposes application settings with validation."""

from __future__ import annotations

from pathlib import Path

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

        self.sidecar_type_combo = QComboBox()
        self.sidecar_type_combo.addItem("YAML (.yaml)", "yaml")
        self.sidecar_type_combo.addItem("JSON (.json)", "json")
        current_extension = (config.sidecar_extension or "yaml").lower()
        current_extension = "yaml" if current_extension == "yml" else current_extension
        idx = self.sidecar_type_combo.findData(current_extension)
        self.sidecar_type_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.embed_check = QCheckBox("Attempt to embed metadata when possible")
        self.embed_check.setChecked(config.embed_metadata)

        self.overwrite_metadata_check = QCheckBox(
            "Overwrite embedded caption/tag fields when present"
        )
        self.overwrite_metadata_check.setChecked(config.overwrite_embedded_metadata)

        self.output_dir_edit = QLineEdit(
            str(config.output_directory) if config.output_directory else ""
        )
        browse_button = QPushButton("Browse…")
        browse_button.clicked.connect(self._select_output_dir)

        output_dir_layout = QHBoxLayout()
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(browse_button)

        self.locale_edit = QLineEdit(config.localization or "")

        self.remote_base_url_edit = QLineEdit(config.remote_base_url)
        self.remote_model_combo = QComboBox()
        self.remote_model_combo.setEditable(True)
        self.remote_model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.remote_model_combo.setEditText(config.remote_model)
        self.remote_refresh_button = QPushButton("Refresh list")
        self.remote_refresh_button.clicked.connect(self._refresh_remote_models)
        remote_model_layout = QHBoxLayout()
        remote_model_layout.addWidget(self.remote_model_combo)
        remote_model_layout.addWidget(self.remote_refresh_button)

        self.remote_api_key_edit = QLineEdit(config.remote_api_key or "")
        self.remote_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)

        self.remote_temperature_spin = QDoubleSpinBox()
        self.remote_temperature_spin.setRange(0.0, 2.0)
        self.remote_temperature_spin.setSingleStep(0.05)
        self.remote_temperature_spin.setDecimals(2)
        self.remote_temperature_spin.setValue(config.remote_temperature)

        self.remote_max_tokens_spin = QSpinBox()
        self.remote_max_tokens_spin.setRange(64, 8192)
        self.remote_max_tokens_spin.setSingleStep(32)
        self.remote_max_tokens_spin.setValue(config.remote_max_tokens)

        self.remote_timeout_spin = QDoubleSpinBox()
        self.remote_timeout_spin.setRange(1.0, 600.0)
        self.remote_timeout_spin.setSingleStep(1.0)
        self.remote_timeout_spin.setDecimals(1)
        self.remote_timeout_spin.setValue(config.remote_timeout)

        form.addRow("Model", self.model_combo)
        form.addRow("Output mode", self.output_mode_combo)
        form.addRow("", self.recursive_check)
        form.addRow("", self.hidden_check)
        form.addRow("", self.captions_check)
        form.addRow("", self.tags_check)
        form.addRow("Max tags", self.max_tags_spin)
        form.addRow("Confidence threshold", self.confidence_spin)
        form.addRow("Workers", self.concurrency_spin)
        form.addRow("Sidecar file type", self.sidecar_type_combo)
        form.addRow("", self.embed_check)
        form.addRow("", self.overwrite_metadata_check)
        form.addRow("Output directory", output_dir_layout)
        form.addRow("Locale", self.locale_edit)
        form.addRow("Remote base URL", self.remote_base_url_edit)
        form.addRow("Remote model id", remote_model_layout)
        form.addRow("Remote API key", self.remote_api_key_edit)
        form.addRow("Remote temperature", self.remote_temperature_spin)
        form.addRow("Remote max tokens", self.remote_max_tokens_spin)
        form.addRow("Remote timeout (s)", self.remote_timeout_spin)

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
        data = self._collect_form_data()
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

    def _collect_form_data(self) -> dict[str, object]:
        return {
            "model_name": self.model_combo.currentData(),
            "output_mode": self.output_mode_combo.currentData(),
            "recursive": self.recursive_check.isChecked(),
            "include_hidden": self.hidden_check.isChecked(),
            "generate_captions": self.captions_check.isChecked(),
            "generate_tags": self.tags_check.isChecked(),
            "max_tags": self.max_tags_spin.value(),
            "confidence_threshold": self.confidence_spin.value(),
            "max_concurrency": self.concurrency_spin.value(),
            "sidecar_extension": self.sidecar_type_combo.currentData(),
            "embed_metadata": self.embed_check.isChecked(),
            "overwrite_embedded_metadata": self.overwrite_metadata_check.isChecked(),
            "output_directory": self._parse_optional_path(self.output_dir_edit.text()),
            "localization": self.locale_edit.text() or None,
            "remote_base_url": self.remote_base_url_edit.text(),
            "remote_model": self.remote_model_combo.currentText(),
            "remote_api_key": self.remote_api_key_edit.text() or None,
            "remote_temperature": self.remote_temperature_spin.value(),
            "remote_max_tokens": self.remote_max_tokens_spin.value(),
            "remote_timeout": self.remote_timeout_spin.value(),
        }

    def _refresh_remote_models(self) -> None:
        model_id = self.model_combo.currentData()
        if model_id != "remote.ollama":
            QMessageBox.information(
                self,
                "Remote discovery",
                "Select the Ollama model before refreshing the remote list.",
            )
            return

        try:
            config = AppConfig.model_validate(self._collect_form_data())
        except Exception as exc:
            QMessageBox.critical(self, "Invalid settings", str(exc))
            return

        try:
            remote_model = ModelRegistry.get(model_id, config=config)
        except Exception as exc:
            QMessageBox.critical(self, "Connection error", str(exc))
            return

        models: list[str] = []
        try:
            discover = getattr(remote_model, "discover_remote_models", None)
            if callable(discover):
                models = discover()
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Discovery failed",
                f"Unable to retrieve remote models: {exc}",
            )
            models = []
        finally:
            session = getattr(remote_model, "_session", None)
            close = getattr(session, "close", None)
            if callable(close):
                close()

        if not models:
            QMessageBox.information(
                self,
                "Remote models",
                "No vision-capable models were reported by the remote backend.",
            )
            return

        current = self.remote_model_combo.currentText()
        self.remote_model_combo.blockSignals(True)
        self.remote_model_combo.clear()
        for name in models:
            self.remote_model_combo.addItem(name)
        if current in models:
            self.remote_model_combo.setCurrentText(current)
        else:
            self.remote_model_combo.setCurrentText(models[0])
        self.remote_model_combo.blockSignals(False)

        QMessageBox.information(
            self,
            "Remote models",
            f"Discovered {len(models)} model(s) from the remote backend.",
        )

    def config(self) -> AppConfig:
        return self._config or self._original_config
