"""String helpers for filename sanitisation."""

from __future__ import annotations

import re
import unicodedata

_SEP_PATTERN = re.compile(r"[^a-z0-9]+")


def slugify_filename(
    text: str,
    *,
    max_length: int = 64,
    default: str = "image",
) -> str | None:
    """Return a lowercase, ASCII-safe slug for use as a filename stem."""
    ascii_only = "".join(ch for ch in text if ord(ch) < 128)
    normalized = ascii_only or unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode(
        "ascii"
    )
    lowered = normalized.lower()
    replaced = _SEP_PATTERN.sub("-", lowered)
    collapsed = re.sub(r"-{2,}", "-", replaced).strip("-")
    if not collapsed:
        collapsed = default.strip("-")
    if not collapsed:
        return None
    slug = collapsed[:max_length].strip("-")
    return slug or None
