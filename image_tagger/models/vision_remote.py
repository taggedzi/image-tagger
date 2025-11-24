"""Vision-language model integration via Ollama."""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from collections.abc import Iterable, Sequence
from typing import Any

from PIL import Image

from ..config import AppConfig
from .base import (
    AnalysisRequest,
    ModelCapability,
    ModelError,
    ModelInfo,
    ModelOutput,
    ModelTag,
    TaggingModel,
)
from .registry import ModelRegistry

logger = logging.getLogger(__name__)

try:  # Requests is an optional dependency until configured.
    import requests
    from requests import Response, Session
except Exception:  # pragma: no cover - handled at runtime with an explicit error
    requests = None  # type: ignore[assignment]
    Session = None  # type: ignore[assignment]
    Response = None  # type: ignore[assignment]


_JSON_OBJECT_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)
_VISION_KEYWORDS = {
    "vision",
    "multimodal",
    "vl",
    "llava",
    "minicpm",
    "paligemma",
    "gemma",
    "qwen",
    "moondream",
    "phi3-vision",
    "pali",
    "pixtral",
    "idefics",
    "cogvlm",
    "omni",
    "image",
}


def _encode_image(image: Image.Image) -> str:
    """Encode an image as a base64 JPEG payload suitable for HTTP APIs."""
    buffer = io.BytesIO()
    converted = image.convert("RGB")
    converted.save(buffer, format="JPEG", quality=92, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return encoded


def _unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


class BaseRemoteVisionModel(TaggingModel):
    """Common functionality for remote multimodal models."""

    def __init__(
        self,
        *,
        identifier: str,
        display_name: str,
        description: str,
        backend: str,
        config: AppConfig | None,
        tags: Sequence[str],
    ) -> None:
        self._backend = backend
        self._config = config or AppConfig()
        self._info = ModelInfo(
            identifier=identifier,
            display_name=display_name,
            description=description,
            capabilities=(ModelCapability.CAPTION, ModelCapability.TAGS),
            tags=tuple(tags),
        )
        self._prompt_version = "vision_remote/v1"
        self._session: Session | None = None

    def info(self) -> ModelInfo:
        return self._info

    def load(self) -> None:
        if requests is None:
            raise ModelError(
                "The 'requests' package is required for remote vision models. "
                "Install it with `pip install requests`."
            )
        self._session = requests.Session()

    def analyze(self, image: Image.Image, request: AnalysisRequest) -> ModelOutput:
        prompt = self._build_prompt(request)
        encoded_image = _encode_image(image)
        raw_text = self._call_backend(encoded_image, prompt, request)
        payload = self._parse_json_response(raw_text)

        caption: str | None = None
        if request.generate_captions:
            caption_raw = payload.get("caption")
            if isinstance(caption_raw, str):
                caption = caption_raw.strip() or None
        filename: str | None = None
        filename_raw = payload.get("filename")
        if isinstance(filename_raw, str):
            filename = filename_raw.strip() or None
        tags: list[ModelTag] = []
        if request.generate_tags:
            tags_payload = payload.get("tags", [])
            tags = self._normalize_tags(tags_payload, request)

        return ModelOutput(
            caption=caption,
            tags=tags[: request.max_tags],
            filename=filename,
            extras={
                "backend": self._backend,
                "remote_model": self._config.remote_model,
                "temperature": self._config.remote_temperature,
                "prompt_version": self._prompt_version,
            },
        )

    # ----- Prompt creation -------------------------------------------------

    def _build_prompt(self, request: AnalysisRequest) -> str:
        locale_hint = request.locale or "English"
        instructions: list[str] = [
            "You are an assistant that analyses a single image and returns strictly valid JSON.",
            "Respond with minified JSON that matches this schema:",
            '{"caption": string|null, "tags": string[], "filename": string|null}',
            f"Use {locale_hint} for all text.",
            "Never wrap the JSON in backticks or additional commentary.",
        ]

        if request.generate_captions:
            instructions.append(
                "Write an accessible, richly detailed description suitable for alt text."
            )
            instructions.append(
                "Explain the overall scene, key objects, people or characters, their "
                "attributes, actions, relationships, and the dominant colours or lighting."
            )
            instructions.append(
                "Use 2-3 sentences and keep the caption under 420 characters while "
                "remaining vivid and clear."
            )
        else:
            instructions.append("Set the caption field to null.")

        if request.generate_tags:
            instructions.append(
                f"Populate the tags array with up to {request.max_tags} short keywords "
                "sorted by relevance. Each tag should be lowercase and omit punctuation."
            )
        else:
            instructions.append("Return an empty array for tags.")

        instructions.append(
            "Propose a concise, URL-safe filename stem in the filename field, without any file "
            "extension. Use lowercase ASCII letters, numbers, and hyphens only, avoid spaces, and "
            "prefer 3-6 short words. Omit the filename if you cannot infer one."
        )

        instructions.append(
            "If you cannot understand the image, return "
            '{"caption": null, "tags": [], "filename": null}.'
        )

        return " ".join(instructions)

    # ----- Backend dispatch ------------------------------------------------

    def _call_backend(
        self,
        encoded_image: str,
        prompt: str,
        request: AnalysisRequest,
    ) -> str:
        raise NotImplementedError

    # ----- Response handling -----------------------------------------------

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = self._strip_markdown(cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as err:
            matches = list(_JSON_OBJECT_PATTERN.finditer(cleaned))
            if not matches:
                raise ModelError(
                    f"{self._info.display_name} returned non-JSON output: {cleaned!r}"
                ) from err
            merged: dict[str, Any] = {}
            errors: list[str] = []
            for match in matches:
                candidate = match.group(0)
                try:
                    fragment = json.loads(candidate)
                    if isinstance(fragment, dict):
                        merged.update(fragment)
                except Exception:  # pragma: no cover - defensive fallback
                    errors.append(candidate)
            if merged:
                return merged
            if errors:
                raise ModelError(
                    f"{self._info.display_name} produced invalid JSON fragments: {errors[0]}"
                ) from None
            raise ModelError(f"{self._info.display_name} produced invalid JSON: {cleaned}") from err

    @staticmethod
    def _strip_markdown(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped
        parts = stripped.split("```")
        # The second segment typically contains the JSON payload (possibly with a language tag).
        if len(parts) < 3:
            return stripped
        candidate = parts[1]
        if "\n" in candidate:
            _, remainder = candidate.split("\n", 1)
            return remainder.strip()
        return parts[-1].strip()

    def _normalize_tags(self, tags_payload: Any, request: AnalysisRequest) -> list[ModelTag]:
        if isinstance(tags_payload, str):
            raw_items = re.split(r"[,\n;]+", tags_payload)
        elif isinstance(tags_payload, Sequence):
            raw_items = [str(item) for item in tags_payload]
        else:
            raw_items = []

        processed = []
        for item in raw_items:
            value = item.strip().lower()
            if not value:
                continue
            processed.append(value)

        deduped = _unique(processed)[: request.max_tags]
        return [ModelTag(value=tag) for tag in deduped]

    # ----- HTTP helpers ----------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._config.remote_api_key:
            headers["Authorization"] = f"Bearer {self._config.remote_api_key}"
        return headers

    def _session_post(
        self,
        url: str,
        payload: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> Response:
        if self._session is None:
            raise ModelError("HTTP session not initialised.")
        try:
            effective_timeout = timeout or self._config.remote_timeout
            response = self._session.post(
                url,
                json=payload,
                headers=self._headers(),
                timeout=effective_timeout,
            )
        except Exception as exc:  # pragma: no cover - network failures are surfaced to users
            if requests is not None and isinstance(exc, requests.exceptions.Timeout):
                raise ModelError(
                    f"{self._backend} request timed out after {effective_timeout}s. "
                    "Increase the remote timeout or ensure the model is loaded."
                ) from exc
            raise ModelError(f"Failed to contact {self._backend} backend: {exc}") from exc
        if response.status_code >= 400:
            raise ModelError(
                f"{self._backend} backend returned HTTP {response.status_code}: {response.text}"
            )
        return response

    def _session_get(self, url: str, *, timeout: float | None = None) -> Response:
        if self._session is None:
            raise ModelError("HTTP session not initialised.")
        try:
            effective_timeout = timeout or self._config.remote_timeout
            response = self._session.get(
                url,
                headers=self._headers(),
                timeout=effective_timeout,
            )
        except Exception as exc:  # pragma: no cover - network failures are surfaced to users
            if requests is not None and isinstance(exc, requests.exceptions.Timeout):
                raise ModelError(
                    f"{self._backend} request timed out after {effective_timeout}s. "
                    "Increase the remote timeout or ensure the model is loaded."
                ) from exc
            raise ModelError(f"Failed to contact {self._backend} backend: {exc}") from exc
        if response.status_code >= 400:
            raise ModelError(
                f"{self._backend} backend returned HTTP {response.status_code}: {response.text}"
            )
        return response

    # ----- Discovery helpers ----------------------------------------------

    def discover_remote_models(self) -> list[str]:
        """Return remote models that appear to support image analysis."""
        try:
            models = self._fetch_remote_model_metadata()
        except ModelError as exc:
            logger.info("Unable to query %s backend for models: %s", self._backend, exc)
            return []

        discovered: list[str] = []
        for name, metadata in models:
            if self._is_vision_candidate(name, metadata):
                discovered.append(name)
        return discovered

    def _fetch_remote_model_metadata(self) -> list[tuple[str, dict[str, Any]]]:
        raise NotImplementedError

    def _is_vision_candidate(self, name: str, metadata: dict[str, Any]) -> bool:
        text = f"{name} {json.dumps(metadata, ensure_ascii=False)}".lower()

        modalities = metadata.get("modalities") or metadata.get("modality")
        if isinstance(modalities, str):
            if any(keyword in modalities.lower() for keyword in _VISION_KEYWORDS):
                return True
        elif isinstance(modalities, (list, tuple)):
            lowered = " ".join(str(item).lower() for item in modalities)
            if any(keyword in lowered for keyword in _VISION_KEYWORDS):
                return True

        families = metadata.get("families")
        if isinstance(families, (list, tuple)):
            lowered = " ".join(str(item).lower() for item in families)
            if any(keyword in lowered for keyword in _VISION_KEYWORDS):
                return True

        for keyword in _VISION_KEYWORDS:
            if keyword in text:
                return True
        return False


class OllamaVisionModel(BaseRemoteVisionModel):
    """Vision-language integration using the Ollama HTTP API."""

    def __init__(self, config: AppConfig | None = None) -> None:
        super().__init__(
            identifier="remote.ollama",
            display_name="Ollama Vision",
            description=(
                "Leverages Ollama-hosted multimodal models such as LLaVA, Qwen2.5-VL, or Gemma."
            ),
            backend="ollama",
            config=config,
            tags=("remote", "ollama", "vision", "http"),
        )

    def _call_backend(
        self,
        encoded_image: str,
        prompt: str,
        request: AnalysisRequest,
    ) -> str:
        endpoint = f"{self._config.remote_base_url}/api/generate"
        payload = {
            "model": self._config.remote_model,
            "prompt": prompt,
            "images": [encoded_image],
            "stream": False,
            "options": {
                "temperature": self._config.remote_temperature,
                "num_predict": self._config.remote_max_tokens,
            },
        }
        try:
            response = self._session_post(endpoint, payload)
        except ModelError as exc:
            message = str(exc).lower()
            if "timed out" not in message:
                raise
            extended_timeout = self._config.remote_timeout + 60.0
            logger.info(
                "Ollama request exceeded %.1fs; retrying once with %.1fs to allow "
                "the model to load.",
                self._config.remote_timeout,
                extended_timeout,
            )
            response = self._session_post(endpoint, payload, timeout=extended_timeout)
        data = response.json()
        if "error" in data:
            raise ModelError(f"Ollama backend error: {data['error']}")
        text = data.get("response")
        if not isinstance(text, str):
            raise ModelError("Ollama backend returned an unexpected payload.")
        return text

    def _fetch_remote_model_metadata(self) -> list[tuple[str, dict[str, Any]]]:
        endpoint = f"{self._config.remote_base_url}/api/tags"
        response = self._session_get(endpoint)
        payload = response.json()
        models = payload.get("models")
        if not isinstance(models, list):
            return []
        results: list[tuple[str, dict[str, Any]]] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            name = item.get("model") or item.get("name")
            if not isinstance(name, str):
                continue
            details = item.get("details")
            if not isinstance(details, dict):
                details = {}
            results.append((name, details))
        return results


def _register() -> None:
    ModelRegistry.register("remote.ollama", lambda config=None: OllamaVisionModel(config=config))


_register()
