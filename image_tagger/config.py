"""Application-wide configuration models and persistence helpers."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator


class OutputMode(str, Enum):
    """Supported ways to persist generated metadata."""

    EMBED = "embed"
    SIDECAR = "sidecar"


class AppConfig(BaseModel):
    """Validates and stores runtime settings for the application."""

    model_name: str = Field(
        default="remote.ollama",
        description="Identifier of the selected tagging model.",
    )
    recursive: bool = Field(
        default=True,
        description="If true, traverse sub-directories when processing folders.",
    )
    include_hidden: bool = Field(
        default=False,
        description="If true, include files and directories that start with a dot.",
    )
    generate_captions: bool = Field(
        default=True,
        description="Controls whether caption text should be produced.",
    )
    generate_tags: bool = Field(
        default=True,
        description="Controls whether keyword tags should be produced.",
    )
    max_tags: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Maximum number of tags to keep per image.",
    )
    confidence_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum confidence value for a generated tag to be included.",
    )
    output_mode: OutputMode = Field(
        default=OutputMode.SIDECAR, description="How metadata is written out."
    )
    sidecar_extension: str = Field(
        default="yaml",
        description="File extension to use when saving sidecar files.",
    )
    embed_metadata: bool = Field(
        default=True,
        description=("When true and supported, embed generated metadata directly in the image."),
    )
    overwrite_embedded_metadata: bool = Field(
        default=False,
        description=(
            "When embedding metadata, overwrite existing caption/tag fields inside the image."
        ),
    )
    suggest_filenames: bool = Field(
        default=False,
        description="Attempt to propose safe filenames for processed images.",
    )
    auto_rename_files: bool = Field(
        default=False,
        description="When true, apply suggested filenames and rename images on disk.",
    )
    output_directory: Path | None = Field(
        default=None,
        description="Optional override directory for generated sidecars.",
    )
    max_concurrency: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Number of worker threads used during batch processing.",
    )
    localization: str | None = Field(
        default=None,
        description="Optional locale hint for models that support multilingual output.",
    )
    remote_base_url: str = Field(
        default="http://127.0.0.1:11434",
        description="Base URL for the Ollama vision backend.",
    )
    remote_model: str = Field(
        default="llava",
        description="Model identifier served by the remote vision backend.",
    )
    remote_api_key: str | None = Field(
        default=None,
        description="Optional bearer token for remote vision backends that require authentication.",
    )
    remote_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature passed to remote vision backends.",
    )
    remote_max_tokens: int = Field(
        default=768,
        ge=64,
        le=8192,
        description="Maximum number of tokens requested from remote vision backends.",
    )
    remote_timeout: float = Field(
        default=90.0,
        ge=1.0,
        le=600.0,
        description="Timeout (seconds) for HTTP calls to remote vision services.",
    )

    @model_validator(mode="after")
    def _validate_sidecar_extension(self) -> AppConfig:
        if self.output_mode == OutputMode.SIDECAR and not self.sidecar_extension:
            raise ValueError("A sidecar extension must be configured for sidecar mode.")
        if self.sidecar_extension.startswith("."):
            self.sidecar_extension = self.sidecar_extension.lstrip(".")
        return self

    @model_validator(mode="after")
    def _normalise_remote_settings(self) -> AppConfig:
        base = self.remote_base_url.strip()
        if not base:
            raise ValueError("Remote base URL must not be empty.")
        if "://" not in base:
            raise ValueError(
                "Remote base URL must include a scheme such as http://localhost:11434."
            )
        self.remote_base_url = base.rstrip("/")
        return self

    @model_validator(mode="after")
    def _validate_filename_preferences(self) -> AppConfig:
        if self.auto_rename_files and not self.suggest_filenames:
            self.suggest_filenames = True
        return self

    def as_dict(self) -> dict[str, Any]:
        """Serialize the configuration to primitive Python types."""
        payload = self.model_dump(mode="json")
        if self.output_directory is not None:
            payload["output_directory"] = str(self.output_directory)
        return payload

    @classmethod
    def load(cls, path: Path) -> AppConfig:
        """Load configuration from a YAML or JSON file."""
        data = _read_config_file(path)
        try:
            return cls.model_validate(data)
        except ValidationError as exc:  # pragma: no cover - pass through details
            raise ValueError(f"Invalid configuration file at {path}: {exc}") from exc

    def save(self, path: Path) -> None:
        """Persist configuration to a YAML file."""
        _write_config_file(path, self.as_dict())


def _read_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _write_config_file(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".yaml", ".yml"}:
        yaml_text = yaml.safe_dump(
            data,
            allow_unicode=False,
            sort_keys=False,
        )
        path.write_text(yaml_text, encoding="utf-8")
    else:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
