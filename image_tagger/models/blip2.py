"""BLIP-2 captioning models with optional quantization."""

from __future__ import annotations

import logging
import math
import re
import sys
from PIL import Image

from .base import (
    AnalysisRequest,
    ModelCapability,
    ModelInfo,
    ModelOutput,
    ModelTag,
    ModelError,
    TaggingModel,
)
from .registry import ModelRegistry
from ..utils.devices import detect_torch_device, torch_device

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from transformers import (
        BitsAndBytesConfig,
        Blip2ForConditionalGeneration,
        Blip2Processor,
    )
except Exception:  # pragma: no cover - optional dependency handling
    BitsAndBytesConfig = None  # type: ignore[assignment]
    Blip2ForConditionalGeneration = None  # type: ignore[assignment]
    Blip2Processor = None  # type: ignore[assignment]

try:  # bitsandbytes is only required for int8/int4 modes
    import bitsandbytes as bnb  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover
    bnb = None  # type: ignore[assignment]


STOP_WORDS = {
    "the",
    "and",
    "with",
    "from",
    "into",
    "over",
    "under",
    "behind",
    "beside",
    "a",
    "an",
    "of",
    "in",
    "on",
    "to",
    "by",
    "for",
    "at",
    "around",
    "between",
    "while",
    "during",
    "through",
    "among",
    "near",
    "its",
    "their",
    "his",
    "her",
    "there",
    "this",
    "that",
    "these",
    "those",
}


class Blip2CaptioningModel(TaggingModel):
    """Caption generator that wraps the BLIP-2 family of models."""

    def __init__(
        self,
        *,
        model_id: str,
        identifier: str,
        display_name: str,
        description: str,
        precision: str,
        max_new_tokens: int = 64,
    ) -> None:
        self.model_id = model_id
        self.identifier = identifier
        self.precision = precision.lower()
        self.max_new_tokens = max_new_tokens
        self._info = ModelInfo(
            identifier=identifier,
            display_name=display_name,
            description=description,
            capabilities=(ModelCapability.CAPTION, ModelCapability.TAGS),
            tags=("transformers", "torch", "caption", "blip2", self.precision),
        )
        self._processor = None
        self._model = None
        self._device = None
        self._is_quantized = False
        self._device_string: str | None = None

    def info(self) -> ModelInfo:
        return self._info

    def load(self) -> None:
        if torch is None or Blip2Processor is None or Blip2ForConditionalGeneration is None:
            raise ModelError(
                "transformers>=4.39 and torch>=2.1 are required for BLIP-2 models. "
                "Install with `pip install image-tagger[blip2]`."
            )

        device_str, message = detect_torch_device()
        logger.info("[BLIP-2] %s", message)
        self._device = torch_device(device_str)
        self._device_string = device_str

        if device_str == "cpu":
            raise ModelError(
                "BLIP-2 models require a CUDA-capable GPU. Select a standard BLIP model "
                "or install/configure a GPU-enabled PyTorch build."
            )

        quant_config: BitsAndBytesConfig | None = None
        torch_dtype = None
        precision = self.precision

        if precision in {"fp32", "float32"}:
            torch_dtype = torch.float32
        elif precision in {"fp16", "float16"}:
            if device_str.startswith("cuda"):
                torch_dtype = torch.float16
            elif device_str == "mps":
                torch_dtype = torch.float16
            else:
                logger.warning(
                    "FP16 requested but GPU not available; falling back to float32 on CPU."
                )
                torch_dtype = torch.float32
        elif precision in {"bf16", "bfloat16"}:
            if not hasattr(torch, "bfloat16"):
                raise ModelError("bfloat16 precision is not supported by this build of PyTorch.")
            torch_dtype = torch.bfloat16  # type: ignore[attr-defined]
        elif precision in {"int8", "nf4-int8"}:
            if BitsAndBytesConfig is None or bnb is None:
                raise ModelError(
                    "bitsandbytes must be installed for 8-bit quantisation. "
                    "Install with `pip install image-tagger[blip2]`."
                )
            if sys.platform == "win32":
                raise ModelError(
                    "bitsandbytes-based 8-bit quantisation is not supported on Windows. "
                    "Use the fp16 precision variant or run under WSL/Linux."
                )
            if not torch.cuda.is_available():
                raise ModelError("8-bit quantisation requires a CUDA-capable GPU.")
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
            self._is_quantized = True
        elif precision in {"int4", "nf4"}:
            if BitsAndBytesConfig is None or bnb is None:
                raise ModelError(
                    "bitsandbytes must be installed for 4-bit quantisation. "
                    "Install with `pip install image-tagger[blip2]`."
                )
            if sys.platform == "win32":
                raise ModelError(
                    "bitsandbytes-based 4-bit quantisation is not supported on Windows. "
                    "Use the fp16 precision variant or run under WSL/Linux."
                )
            if not torch.cuda.is_available():
                raise ModelError("4-bit quantisation requires a CUDA-capable GPU.")
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)
            self._is_quantized = True
        else:
            raise ModelError(f"Unsupported precision '{precision}'.")

        model_kwargs = {}
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
            model_kwargs["device_map"] = "auto"
        else:
            if torch_dtype is not None:
                model_kwargs["dtype"] = torch_dtype

        try:
            self._processor = Blip2Processor.from_pretrained(self.model_id)
            try:
                self._model = Blip2ForConditionalGeneration.from_pretrained(
                    self.model_id,
                    **model_kwargs,
                )
            except TypeError as type_err:
                if "unexpected keyword argument 'dtype'" in str(type_err):
                    # Older versions of transformers expect torch_dtype
                    model_kwargs.pop("dtype", None)
                    if torch_dtype is not None:
                        model_kwargs["torch_dtype"] = torch_dtype
                    self._model = Blip2ForConditionalGeneration.from_pretrained(
                        self.model_id,
                        **model_kwargs,
                    )
                else:
                    raise
        except Exception as exc:  # pragma: no cover - propagate context
            raise ModelError(f"Failed to load BLIP-2 model '{self.model_id}': {exc}") from exc

        if not self._is_quantized:
            assert self._model is not None
            self._model.to(self._device)

        self._model.eval()

    def analyze(self, image: Image.Image, request: AnalysisRequest) -> ModelOutput:
        assert self._processor is not None and self._model is not None
        assert torch is not None

        tags: list[ModelTag] = []
        caption: str | None = None

        inputs = self._processor(images=image, return_tensors="pt")

        if self._is_quantized:
            target_device = torch_device("cuda")
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "num_beams": 3,
        }

        with torch.inference_mode():
            generated_ids = self._model.generate(**inputs, **generation_kwargs)

        if hasattr(self._processor, "tokenizer"):
            caption = self._processor.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
        else:  # pragma: no cover - fallback path
            caption = str(generated_ids)

        if request.generate_tags and caption:
            tags = self._extract_tags_from_caption(
                caption, max_count=request.max_tags, threshold=request.confidence_threshold
            )

        if request.generate_captions:
            caption = caption or None
        else:
            caption = None

        return ModelOutput(
            caption=caption,
            tags=tags,
            extras={
                "model_id": self.model_id,
                "precision": self.precision,
                "device": self._device_string,
            },
        )

    @staticmethod
    def _extract_tags_from_caption(
        caption: str,
        *,
        max_count: int,
        threshold: float,
    ) -> list[ModelTag]:
        words = re.findall(r"[a-zA-Z][a-zA-Z\\-]+", caption.lower())
        seen: set[str] = set()
        tags: list[ModelTag] = []
        base_confidence = max(0.55, min(0.9, 1.0 - math.log(max_count + 1, 10)))
        for word in words:
            if word in STOP_WORDS:
                continue
            normalized = word.strip("-")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            confidence = base_confidence
            if confidence < threshold:
                continue
            tags.append(ModelTag(normalized, confidence=confidence))
            if len(tags) >= max_count:
                break
        return tags


def _register() -> None:
    variants = [
        (
            "caption.blip2-opt-6.7b-fp16",
            "Salesforce/blip2-opt-6.7b",
            "BLIP-2 OPT 6.7B (FP16)",
            "Large BLIP-2 OPT model with 6.7B parameters. Requires significant GPU memory.",
            "fp16",
        ),
        (
            "caption.blip2-opt-6.7b-int8",
            "Salesforce/blip2-opt-6.7b",
            "BLIP-2 OPT 6.7B (INT8)",
            "6.7B BLIP-2 OPT model quantised to 8-bit for reduced VRAM usage.",
            "int8",
        ),
        (
            "caption.blip2-opt-6.7b-int4",
            "Salesforce/blip2-opt-6.7b",
            "BLIP-2 OPT 6.7B (INT4)",
            "6.7B BLIP-2 OPT model quantised to 4-bit for aggressive memory savings.",
            "int4",
        ),
        (
            "caption.blip2-opt-2.7b-fp16",
            "Salesforce/blip2-opt-2.7b",
            "BLIP-2 OPT 2.7B (FP16)",
            "Mid-sized BLIP-2 OPT model balancing quality and performance.",
            "fp16",
        ),
        (
            "caption.blip2-opt-2.7b-int8",
            "Salesforce/blip2-opt-2.7b",
            "BLIP-2 OPT 2.7B (INT8)",
            "2.7B BLIP-2 OPT model quantised to 8-bit.",
            "int8",
        ),
        (
            "caption.blip2-opt-2.7b-int4",
            "Salesforce/blip2-opt-2.7b",
            "BLIP-2 OPT 2.7B (INT4)",
            "2.7B BLIP-2 OPT model quantised to 4-bit.",
            "int4",
        ),
        (
            "caption.blip2-opt-2.7b-coco-fp16",
            "Salesforce/blip2-opt-2.7b-coco",
            "BLIP-2 OPT 2.7B COCO (FP16)",
            "COCO-finetuned BLIP-2 OPT 2.7B model for photographic captions.",
            "fp16",
        ),
        (
            "caption.blip2-opt-2.7b-coco-int8",
            "Salesforce/blip2-opt-2.7b-coco",
            "BLIP-2 OPT 2.7B COCO (INT8)",
            "COCO BLIP-2 OPT 2.7B quantised to 8-bit.",
            "int8",
        ),
        (
            "caption.blip2-opt-2.7b-coco-int4",
            "Salesforce/blip2-opt-2.7b-coco",
            "BLIP-2 OPT 2.7B COCO (INT4)",
            "COCO BLIP-2 OPT 2.7B quantised to 4-bit.",
            "int4",
        ),
        (
            "caption.blip2-flan-t5-xl-fp16",
            "Salesforce/blip2-flan-t5-xl",
            "BLIP-2 Flan-T5 XL (FP16)",
            "BLIP-2 model paired with Flan-T5 XL decoder.",
            "fp16",
        ),
        (
            "caption.blip2-flan-t5-xl-int8",
            "Salesforce/blip2-flan-t5-xl",
            "BLIP-2 Flan-T5 XL (INT8)",
            "Flan-T5 XL BLIP-2 quantised to 8-bit.",
            "int8",
        ),
    ]

    for identifier, model_id, label, desc, precision in variants:
        ModelRegistry.register(
            identifier,
            lambda model_id=model_id, identifier=identifier, label=label, desc=desc, precision=precision: Blip2CaptioningModel(
                model_id=model_id,
                identifier=identifier,
                display_name=label,
                description=desc,
                precision=precision,
            ),
        )


_register()
