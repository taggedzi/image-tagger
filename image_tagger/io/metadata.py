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


def _ensure_bytes(raw: object) -> bytes:
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, bytearray):
        return bytes(raw)
    if isinstance(raw, memoryview):
        return bytes(raw)
    if isinstance(raw, (tuple, list)):
        try:
            return bytes(int(item) & 0xFF for item in raw)
        except Exception:
            return b""
    return b""


def _decode_utf8(raw: object) -> str:
    data = _ensure_bytes(raw)
    if not data:
        return ""
    return data.decode("utf-8", errors="ignore").strip()


def _decode_xp_keywords(raw: object) -> list[str]:
    data = _ensure_bytes(raw)
    if not data:
        return []
    decoded = data.decode("utf-16le", errors="ignore").rstrip("\x00").split("\x00")
    return [item for item in decoded if item]


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
        overwrite_existing: bool = True,
    ) -> bool:
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(f"{ext} does not support embedded metadata.")

        if ext in {".jpg", ".jpeg"}:
            return self._write_jpeg(
                path, caption=caption, tags=list(tags), overwrite_existing=overwrite_existing
            )

        if ext == ".png":
            return self._write_png(
                path, caption=caption, tags=list(tags), overwrite_existing=overwrite_existing
            )

        raise UnsupportedFormatError(ext)

    def _write_jpeg(
        self,
        path: Path,
        *,
        caption: str | None,
        tags: list[str],
        overwrite_existing: bool,
    ) -> bool:
        if piexif is None:
            logger.warning("piexif is not available; cannot embed metadata in %s", path)
            return False

        with Image.open(path) as image:
            exif_blob = image.info.get("exif")
            if exif_blob:
                exif_dict = piexif.load(exif_blob)
            else:
                # Create a minimal EXIF structure so piexif.dump succeeds.
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "Interop": {}, "thumbnail": None}

        zeroth = exif_dict.setdefault("0th", {})

        existing_caption = _decode_utf8(zeroth.get(piexif.ImageIFD.ImageDescription, b""))
        xp_keywords_tag = getattr(piexif.ImageIFD, "XPKeywords", 0x9C9E)
        existing_keywords = _decode_xp_keywords(zeroth.get(xp_keywords_tag, b""))

        should_write_caption = bool(caption) and (overwrite_existing or not existing_caption)
        should_write_tags = bool(tags) and (overwrite_existing or not existing_keywords)

        if not should_write_caption and not should_write_tags:
            logger.info(
                "Skipping JPEG metadata for %s because caption/tags already populated.", path
            )
            return False

        if should_write_caption and caption:
            zeroth[piexif.ImageIFD.ImageDescription] = caption.encode("utf-8")
            utf16_caption = caption.encode("utf-16le") + b"\x00\x00"
            xp_title_tag = getattr(piexif.ImageIFD, "XPTitle", 0x9C9B)
            xp_subject_tag = getattr(piexif.ImageIFD, "XPSubject", 0x9C9F)
            zeroth[xp_title_tag] = utf16_caption
            zeroth[xp_subject_tag] = utf16_caption
        if should_write_tags and tags:
            joined = "\u0000".join(tags)
            utf16_keywords = joined.encode("utf-16le") + b"\x00\x00"
            zeroth[xp_keywords_tag] = utf16_keywords

        exif_bytes = piexif.dump(exif_dict)
        with Image.open(path) as image:
            image.save(path, exif=exif_bytes)
        return self._verify_jpeg_metadata(
            path,
            caption=caption if should_write_caption else None,
            tags=tags if should_write_tags else [],
        )

    def _write_png(
        self,
        path: Path,
        *,
        caption: str | None,
        tags: list[str],
        overwrite_existing: bool,
    ) -> bool:
        with Image.open(path) as image:
            existing_caption = (image.info.get("Description") or "").strip()
            existing_keywords = (image.info.get("Keywords") or "").strip()

        should_write_caption = bool(caption) and (overwrite_existing or not existing_caption)
        should_write_tags = bool(tags) and (overwrite_existing or not existing_keywords)

        if not should_write_caption and not should_write_tags:
            logger.info(
                "Skipping PNG metadata for %s because caption/tags already populated.", path
            )
            return False

        png_info = PngImagePlugin.PngInfo()
        if should_write_caption and caption:
            png_info.add_text("Description", caption)
        elif existing_caption:
            png_info.add_text("Description", existing_caption)

        if should_write_tags and tags:
            png_info.add_text("Keywords", ",".join(tags))
        elif existing_keywords:
            png_info.add_text("Keywords", existing_keywords)

        with Image.open(path) as image:
            image.save(path, pnginfo=png_info)
        return self._verify_png_metadata(
            path,
            caption=caption if should_write_caption else None,
            tags=tags if should_write_tags else [],
        )

    def _verify_jpeg_metadata(self, path: Path, *, caption: str | None, tags: list[str]) -> bool:
        if piexif is None:
            return False

        try:
            with Image.open(path) as image:
                exif_blob = image.info.get("exif")
        except Exception:
            logger.exception("Unable to reopen %s to verify EXIF write", path)
            return False

        if not exif_blob:
            logger.warning("JPEG %s reported success but no EXIF blob was present after saving.", path)
            return False

        try:
            exif_dict = piexif.load(exif_blob)
        except Exception as exc:
            logger.warning("Failed to parse EXIF when verifying %s: %s", path, exc)
            return False

        zeroth = exif_dict.get("0th", {})

        if caption:
            stored_caption = _decode_utf8(zeroth.get(piexif.ImageIFD.ImageDescription, b""))
            if not stored_caption:
                logger.warning("JPEG %s missing ImageDescription after embedding.", path)
                return False
            if stored_caption.strip() != caption.strip():
                logger.warning(
                    "JPEG %s caption mismatch after embedding. Expected %r, found %r",
                    path,
                    caption,
                    stored_caption,
                )
                return False

        if tags:
            xp_keywords_tag = getattr(piexif.ImageIFD, "XPKeywords", 0x9C9E)
            decoded = _decode_xp_keywords(zeroth.get(xp_keywords_tag, b""))
            if not decoded:
                logger.warning("JPEG %s missing XPKeywords after embedding.", path)
                return False
            missing = [tag for tag in tags if tag not in decoded]
            if missing:
                logger.warning(
                    "JPEG %s keywords mismatch after embedding. Missing: %s",
                    path,
                    ", ".join(missing),
                )
                return False

        return True

    def _verify_png_metadata(self, path: Path, *, caption: str | None, tags: list[str]) -> bool:
        if not caption and not tags:
            return True

        try:
            with Image.open(path) as image:
                info = dict(image.info)
        except Exception:
            logger.exception("Unable to reopen %s to verify PNG metadata", path)
            return False

        if caption:
            stored_caption = (info.get("Description") or "").strip()
            if not stored_caption:
                logger.warning("PNG %s missing Description after embedding.", path)
                return False
            if stored_caption != caption:
                logger.warning(
                    "PNG %s caption mismatch after embedding. Expected %r, found %r",
                    path,
                    caption,
                    stored_caption,
                )
                return False

        if tags:
            stored_keywords = (info.get("Keywords") or "").strip()
            if not stored_keywords:
                logger.warning("PNG %s missing Keywords after embedding.", path)
                return False
            stored_list = [item.strip() for item in stored_keywords.split(",") if item.strip()]
            missing = [tag for tag in tags if tag not in stored_list]
            if missing:
                logger.warning(
                    "PNG %s keywords mismatch after embedding. Missing: %s",
                    path,
                    ", ".join(missing),
                )
                return False

        return True
