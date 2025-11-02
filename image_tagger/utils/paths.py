"""Path helpers used across the application."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
    ".gif",
    ".heic",
}


def is_image_file(path: Path, *, extensions: Iterable[str] | None = None) -> bool:
    """Return True if the given path has a supported image extension."""
    exts = {ext.lower() for ext in (extensions or IMAGE_EXTENSIONS)}
    return path.suffix.lower() in exts


def resolve_image_paths(
    start: Path,
    *,
    recursive: bool = True,
    include_hidden: bool = False,
    extensions: Sequence[str] | None = None,
) -> list[Path]:
    """Collect image paths starting from ``start``.

    Parameters
    ----------
    start:
        File or directory to inspect.
    recursive:
        Whether to traverse sub-directories.
    include_hidden:
        Include files whose name starts with ``.`` when True.
    extensions:
        Optional whitelist of extensions to match.
    """

    start = start.expanduser()
    if not start.exists():
        raise FileNotFoundError(start)

    collected: list[Path] = []

    if start.is_file():
        if is_image_file(start, extensions=extensions):
            if include_hidden or not _is_hidden(start):
                collected.append(start)
        return collected

    walker: Iterator[Path]
    if recursive:
        walker = (path for path in start.rglob("*") if path.is_file())
    else:
        walker = (path for path in start.iterdir() if path.is_file())

    for path in walker:
        if not include_hidden and _is_hidden(path):
            continue
        if is_image_file(path, extensions=extensions):
            collected.append(path)

    collected.sort()
    return collected


def _is_hidden(path: Path) -> bool:
    name = path.name
    if name.startswith("."):
        return True
    # Windows compatibility: check attribute bit if available.
    try:
        import ctypes

        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
        if attrs == -1:
            return False
        return bool(attrs & 2)  # FILE_ATTRIBUTE_HIDDEN bit
    except (AttributeError, ImportError, OSError):
        return False

