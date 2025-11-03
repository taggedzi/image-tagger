"""Helpers for selecting accelerator devices when optional libraries are present."""

from __future__ import annotations

import logging
from typing import Tuple

logger = logging.getLogger(__name__)

try:  # Torch is optional; fall back gracefully when unavailable.
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


def detect_torch_device(preference: str = "auto") -> Tuple[str, str]:
    """Determine the most appropriate torch device string.

    Returns
    -------
    tuple[str, str]
        A pair of ``(device_string, message)`` describing the selected device
        and a human-readable explanation.
    """

    if torch is None:
        return "cpu", "PyTorch is not installed; using CPU execution."

    pref = (preference or "auto").lower()

    def wants(target: str) -> bool:
        return pref == target or pref == "auto"

    if wants("cuda") and torch.cuda.is_available():
        index = torch.cuda.current_device()
        name = torch.cuda.get_device_name(index)
        return f"cuda:{index}", f"CUDA device detected: {name}"

    mps_backend = getattr(torch.backends, "mps", None)
    if wants("mps") and mps_backend and mps_backend.is_available():
        return "mps", "Apple MPS backend detected."

    if wants("xpu") and hasattr(torch, "xpu") and torch.xpu.is_available():  # pragma: no cover
        return "xpu", "Intel XPU backend detected."

    return "cpu", "No GPU accelerator detected; using CPU."


def torch_device(device_str: str):
    """Convert a device string to a torch.device when torch is available."""
    if torch is None:
        return device_str
    return torch.device(device_str)

