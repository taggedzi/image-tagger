"""Dialog that exposes application settings with validation."""

from __future__ import annotations

from pathlib import Path
from functools import partial

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QToolButton,
    QVBoxLayout,
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
        self._field_min_width = 320

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        self.model_combo = QComboBox()
        for info in ModelRegistry.list_model_infos():
            self.model_combo.addItem(info.display_name, info.identifier)
        idx = self.model_combo.findData(config.model_name)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self._normalise_width(self.model_combo)

        self.output_mode_combo = QComboBox()
        for mode in OutputMode:
            self.output_mode_combo.addItem(mode.name.title(), mode.value)
        mode_index = self.output_mode_combo.findData(config.output_mode.value)
        self.output_mode_combo.setCurrentIndex(max(mode_index, 0))
        self._normalise_width(self.output_mode_combo)

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
        self._normalise_width(self.max_tags_spin)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setValue(config.confidence_threshold)
        self._normalise_width(self.confidence_spin)

        self.concurrency_spin = QSpinBox()
        self.concurrency_spin.setRange(1, 32)
        self.concurrency_spin.setValue(config.max_concurrency)
        self._normalise_width(self.concurrency_spin)

        self.sidecar_type_combo = QComboBox()
        self.sidecar_type_combo.addItem("YAML (.yaml)", "yaml")
        self.sidecar_type_combo.addItem("JSON (.json)", "json")
        current_extension = (config.sidecar_extension or "yaml").lower()
        current_extension = "yaml" if current_extension == "yml" else current_extension
        idx = self.sidecar_type_combo.findData(current_extension)
        self.sidecar_type_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self._normalise_width(self.sidecar_type_combo)
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

        output_dir_container = QWidget()
        output_dir_layout = QHBoxLayout(output_dir_container)
        output_dir_layout.setContentsMargins(0, 0, 0, 0)
        output_dir_layout.addWidget(self.output_dir_edit)
        output_dir_layout.addWidget(browse_button)
        self._normalise_width(output_dir_container)

        self.locale_edit = QLineEdit(config.localization or "")
        self._normalise_width(self.locale_edit)

        self.remote_base_url_edit = QLineEdit(config.remote_base_url)
        self._normalise_width(self.remote_base_url_edit)
        self.remote_model_combo = QComboBox()
        self.remote_model_combo.setEditable(True)
        self.remote_model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.remote_model_combo.setEditText(config.remote_model)
        self.remote_refresh_button = QPushButton("Refresh list")
        self.remote_refresh_button.clicked.connect(self._refresh_remote_models)
        remote_model_container = QWidget()
        remote_model_layout = QHBoxLayout(remote_model_container)
        remote_model_layout.setContentsMargins(0, 0, 0, 0)
        remote_model_layout.addWidget(self.remote_model_combo)
        remote_model_layout.addWidget(self.remote_refresh_button)
        self._normalise_width(remote_model_container)

        self.remote_api_key_edit = QLineEdit(config.remote_api_key or "")
        self.remote_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._normalise_width(self.remote_api_key_edit)

        self.remote_temperature_spin = QDoubleSpinBox()
        self.remote_temperature_spin.setRange(0.0, 2.0)
        self.remote_temperature_spin.setSingleStep(0.05)
        self.remote_temperature_spin.setDecimals(2)
        self.remote_temperature_spin.setValue(config.remote_temperature)
        self._normalise_width(self.remote_temperature_spin)

        self.remote_max_tokens_spin = QSpinBox()
        self.remote_max_tokens_spin.setRange(64, 8192)
        self.remote_max_tokens_spin.setSingleStep(32)
        self.remote_max_tokens_spin.setValue(config.remote_max_tokens)
        self._normalise_width(self.remote_max_tokens_spin)

        self.remote_timeout_spin = QDoubleSpinBox()
        self.remote_timeout_spin.setRange(1.0, 600.0)
        self.remote_timeout_spin.setSingleStep(1.0)
        self.remote_timeout_spin.setDecimals(1)
        self.remote_timeout_spin.setValue(config.remote_timeout)
        self._normalise_width(self.remote_timeout_spin)

        model_group = QGroupBox("Model & Analysis")
        model_form = self._create_form_layout()
        model_form.addRow(
            "Model",
            self._with_help(
                self.model_combo,
                "Model",
                "Choose which captioning backend to run. BLIP checkpoints execute locally; "
                "the Ollama option sends each image to a running Ollama server over HTTP.",
            ),
        )
        model_form.addRow(
            "",
            self._with_help(
                self.captions_check,
                "Generate captions",
                "Controls whether the selected model should return natural-language descriptions.",
            ),
        )
        model_form.addRow(
            "",
            self._with_help(
                self.tags_check,
                "Generate tags",
                "Controls whether keyword tags should be produced from the caption/model output.",
            ),
        )
        model_form.addRow(
            "Max tags",
            self._with_help(
                self.max_tags_spin,
                "Maximum tags",
                "Upper limit on how many tags are kept per image. Lower numbers keep the output concise.",
            ),
        )
        model_form.addRow(
            "Confidence threshold",
            self._with_help(
                self.confidence_spin,
                "Confidence threshold",
                "Minimum confidence a generated tag must have before it is kept. "
                "Lower this to accept more speculative tags.",
            ),
        )
        model_form.addRow(
            "Locale",
            self._with_help(
                self.locale_edit,
                "Locale",
                "Optional hint that nudges compatible models to respond in the specified language (e.g. 'en-US' or 'fr').",
            ),
        )
        model_group.setLayout(model_form)
        main_layout.addWidget(model_group)

        processing_group = QGroupBox("Batch Processing")
        processing_form = self._create_form_layout()
        processing_form.addRow(
            "",
            self._with_help(
                self.recursive_check,
                "Sub-folders",
                "Enable this to walk through any sub-directories found under the chosen folder.",
            ),
        )
        processing_form.addRow(
            "",
            self._with_help(
                self.hidden_check,
                "Hidden files",
                "Toggle whether dot-prefixed files and folders should be processed.",
            ),
        )
        processing_form.addRow(
            "Workers",
            self._with_help(
                self.concurrency_spin,
                "Workers",
                "How many images to process at the same time. Increase to speed up large batches, "
                "but note that heavy models will consume more CPU/GPU memory.",
            ),
        )
        processing_group.setLayout(processing_form)
        main_layout.addWidget(processing_group)

        output_group = QGroupBox("Metadata Output")
        output_form = self._create_form_layout()
        output_form.addRow(
            "Output mode",
            self._with_help(
                self.output_mode_combo,
                "Output mode",
                "Select how metadata is persisted. 'Embed' writes captions/tags into supported "
                "image formats, while 'Sidecar' emits YAML/JSON files next to each image.",
            ),
        )
        output_form.addRow(
            "Sidecar file type",
            self._with_help(
                self.sidecar_type_combo,
                "Sidecar file type",
                "Choose YAML or JSON for generated sidecar files when sidecar mode is active.",
            ),
        )
        output_form.addRow(
            "",
            self._with_help(
                self.embed_check,
                "Embed metadata",
                "When enabled, the app attempts to write captions and tags directly into the image metadata.",
            ),
        )
        output_form.addRow(
            "",
            self._with_help(
                self.overwrite_metadata_check,
                "Overwrite metadata",
                "Enable this if you want embedded captions/tags to replace any existing values in the file.",
            ),
        )
        output_form.addRow(
            "Output directory",
            self._with_help(
                output_dir_container,
                "Output directory",
                "Optional override folder for generated sidecars. Leave blank to write files next to each image.",
            ),
        )
        output_group.setLayout(output_form)
        main_layout.addWidget(output_group)

        remote_group = QGroupBox("Ollama Remote Settings")
        remote_form = self._create_form_layout()
        remote_form.addRow(
            "Remote base URL",
            self._with_help(
                self.remote_base_url_edit,
                "Remote base URL",
                "HTTP address of your Ollama server, typically http://localhost:11434 when running locally.",
            ),
        )
        remote_form.addRow(
            "Remote model id",
            self._with_help(
                remote_model_container,
                "Remote model id",
                "Name of the Ollama model to invoke (e.g. 'llava:13b'). Use the Refresh button to query the server.",
            ),
        )
        remote_form.addRow(
            "Remote API key",
            self._with_help(
                self.remote_api_key_edit,
                "Remote API key",
                "Optional bearer token sent with every remote request. Leave empty if your server is unsecured.",
            ),
        )
        remote_form.addRow(
            "Remote temperature",
            self._with_help(
                self.remote_temperature_spin,
                "Remote temperature",
                "Controls response randomness for remote models. Smaller numbers yield more deterministic captions.",
            ),
        )
        remote_form.addRow(
            "Remote max tokens",
            self._with_help(
                self.remote_max_tokens_spin,
                "Remote max tokens",
                "Upper bound on how many tokens the remote backend may return for each request.",
            ),
        )
        remote_form.addRow(
            "Remote timeout (s)",
            self._with_help(
                self.remote_timeout_spin,
                "Remote timeout",
                "How long to wait for remote HTTP responses. Increase this if large models take longer to start.",
            ),
        )
        remote_group.setLayout(remote_form)
        self.remote_group = remote_group
        main_layout.addWidget(remote_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons, alignment=Qt.AlignmentFlag.AlignRight)

        self.model_combo.currentIndexChanged.connect(self._on_model_selection_changed)
        self._on_model_selection_changed()

        self.setMinimumWidth(560)

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

    def _with_help(self, widget: QWidget, title: str, message: str) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        layout.addStretch()
        layout.addWidget(self._make_help_button(title, message))
        self._normalise_width(container)
        return container

    def _make_help_button(self, title: str, message: str) -> QToolButton:
        button = QToolButton(self)
        button.setText("?")
        button.setAutoRaise(True)
        button.setFixedSize(24, 24)
        button.clicked.connect(
            partial(QMessageBox.information, self, title, message)
        )
        return button

    def _normalise_width(self, widget: QWidget) -> None:
        widget.setMinimumWidth(self._field_min_width)

    def _create_form_layout(self) -> QFormLayout:
        layout = QFormLayout()
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return layout

    def _on_model_selection_changed(self) -> None:
        model_id = self.model_combo.currentData()
        show_remote = model_id == "remote.ollama"
        self.remote_group.setVisible(show_remote)
