"""About dialog presenting metadata and environment information."""

from __future__ import annotations

import platform
import sys
from importlib import metadata

from PySide6 import __version__ as PYSIDE_VERSION
from PySide6.QtCore import Qt, qVersion
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from ...config import AppConfig
from ...utils.devices import detect_torch_device
from ..assets import load_app_icon


def _get_package_metadata() -> tuple[str, str, str]:
    """Return (version, homepage, summary) from package metadata."""
    try:
        pkg_version = metadata.version("image-tagger")
        pkg_meta = metadata.metadata("image-tagger")
        homepage = pkg_meta.get("Home-page") or pkg_meta.get("Project-URL", "")
        summary = pkg_meta.get("Summary", "")
        return pkg_version, homepage, summary
    except metadata.PackageNotFoundError:
        return "0.1.0-dev", "", ""


def _torch_info() -> str:
    device_str, message = detect_torch_device()
    try:
        import torch  # noqa: WPS433
    except Exception:
        torch_version = "not installed"
    else:
        torch_version = torch.__version__
    return f"PyTorch: {torch_version} • {message}"


def _ollama_info(base_url: str) -> str:
    try:
        import requests  # noqa: WPS433
    except Exception:
        return f"Ollama backend: requests not installed (configured {base_url})"

    endpoint = f"{base_url.rstrip('/')}/api/version"
    try:
        response = requests.get(endpoint, timeout=1.5)
    except Exception as exc:
        return f"Ollama backend unreachable ({base_url}): {exc}"

    if not response.ok:
        return f"Ollama backend error ({base_url}): HTTP {response.status_code}"

    try:
        payload = response.json()
    except ValueError:
        payload = {}
    version = payload.get("version") if isinstance(payload, dict) else None
    if isinstance(version, str) and version:
        return f"Ollama backend reachable ({base_url}) • version {version}"
    return f"Ollama backend reachable ({base_url})"


def _requests_info() -> str:
    try:
        import requests  # noqa: WPS433
    except Exception:
        return "Requests: not installed"
    return f"Requests: {requests.__version__}"


def _gather_environment(config: AppConfig) -> str:
    lines = [
        f"Python: {platform.python_version()} ({sys.executable})",
        f"Platform: {platform.platform()}",
        f"PySide6: {PYSIDE_VERSION} / Qt: {qVersion()}",
    ]

    lines.append(_torch_info())
    lines.append(_requests_info())
    lines.append(_ollama_info(config.remote_base_url))
    lines.append(f"Configured model: {config.model_name}")
    lines.append(f"Remote model id: {config.remote_model}")
    lines.append(f"Output mode: {config.output_mode.value}")
    lines.append(f"Max concurrency: {config.max_concurrency}")
    return "\n".join(lines)


class AboutDialog(QDialog):
    """Simple about dialog with project metadata and env info."""

    def __init__(self, config: AppConfig, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About Image Tagger")
        icon = load_app_icon()
        if icon is not None:
            self.setWindowIcon(icon)

        version, homepage, summary = _get_package_metadata()
        description = summary or (
            "Cross-platform desktop app that captions and tags images using BLIP "
            "or remote Ollama models."
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title_label = QLabel("<h2>Image Tagger</h2>")
        title_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(title_label)

        version_label = QLabel(f"Version {version}")
        layout.addWidget(version_label)

        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        if homepage:
            link_label = QLabel(
                f'<a href="{homepage}">{homepage}</a>'
            )
        else:
            link_label = QLabel(
                '<a href="https://github.com/your-org/image-tagger">Project repository</a>'
            )
        link_label.setOpenExternalLinks(True)
        layout.addWidget(link_label)

        env_label = QLabel("<b>Environment</b>")
        env_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(env_label)

        env_text = QPlainTextEdit()
        env_text.setReadOnly(True)
        env_text.setPlainText(_gather_environment(config))
        env_text.setMinimumHeight(180)
        layout.addWidget(env_text)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
