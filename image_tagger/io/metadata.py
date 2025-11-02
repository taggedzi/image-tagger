"""Embed generated metadata directly into supported image formats."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from PIL import Image, PngImagePlugin

try:
    import piexif
except Exception:  # pragma: no cover
    piexif = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class UnsupportedFormatError(RuntimeError):
    """Raised when attempting to embed metadata into an unsupported image type."""


class MetadataWriter:
    """Persist captions and tags inside the image file where possible."""

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    def write(
        self,
        path: Path,
        *,
        caption: str | None,
        tags: Iterable[str],
    ) -> bool:
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(f"{ext} does not support embedded metadata.")

        if ext in {".jpg", ".jpeg"}:
            return self._write_jpeg(path, caption=caption, tags=list(tags))

        if ext == ".png":
            return self._write_png(path, caption=caption, tags=list(tags))

        raise UnsupportedFormatError(ext)

    def _write_jpeg(self, path: Path, *, caption: str | None, tags: list[str]) -> bool:
        if piexif is None:
            logger.warning("piexif is not available; cannot embed metadata in %s", path)
            return False

        with Image.open(path) as image:
            exif_dict = piexif.load(image.info.get("exif", b""))

        if caption:
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = caption.encode("utf-8")
        if tags:
            keywords = ";".join(tags).encode("utf-16le")
            exif_dict["0th"][piexif.ImageIFD.XPKeywords] = keywords

        exif_bytes = piexif.dump(exif_dict)
        with Image.open(path) as image:
            image.save(path, exif=exif_bytes)
        return True

    def _write_png(self, path: Path, *, caption: str | None, tags: list[str]) -> bool:
        png_info = PngImagePlugin.PngInfo()
        if caption:
            png_info.add_text("Description", caption)
        if tags:
            png_info.add_text("Keywords", ",".join(tags))

        with Image.open(path) as image:
            image.save(path, pnginfo=png_info)
        return True
