import pytest
from PIL import Image

from image_tagger.io.metadata import (
    MetadataWriter,
    UnsupportedFormatError,
    _decode_utf8,
    _decode_xp_keywords,
    _ensure_bytes,
)

piexif = pytest.importorskip("piexif")
XP_KEYWORDS_TAG = getattr(piexif.ImageIFD, "XPKeywords", 0x9C9E)


def _as_bytes(raw):
    if isinstance(raw, bytes):
        return raw
    if isinstance(raw, bytearray):
        return bytes(raw)
    if isinstance(raw, memoryview):
        return raw.tobytes()
    if isinstance(raw, (tuple, list)):
        return bytes(int(item) & 0xFF for item in raw)
    return b""


def _decode_caption(raw) -> str:
    return _as_bytes(raw).decode("utf-8", errors="ignore") if raw else ""


def _decode_keywords(raw) -> list[str]:
    data = _as_bytes(raw)
    if not data:
        return []
    chunks = data.decode("utf-16le", errors="ignore").rstrip("\x00").split("\x00")
    return [item for item in chunks if item]


def _create_jpeg_with_metadata(
    path,
    *,
    caption: str | None = None,
    tags: list[str] | None = None,
) -> None:
    image = Image.new("RGB", (8, 8), color=(0, 128, 255))
    exif_dict = {
        "0th": {},
        "Exif": {},
        "GPS": {},
        "1st": {},
        "Interop": {},
        "thumbnail": None,
    }
    if caption:
        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = caption.encode("utf-8")
    if tags:
        payload = "\u0000".join(tags).encode("utf-16le") + b"\x00\x00"
        exif_dict["0th"][XP_KEYWORDS_TAG] = payload
    exif_bytes = piexif.dump(exif_dict)
    image.save(path, format="JPEG", exif=exif_bytes)


def test_write_jpeg_without_existing_exif(tmp_path):
    path = tmp_path / "sample.jpg"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(path, format="JPEG")

    writer = MetadataWriter()
    result = writer.write(
        path,
        caption="Test caption",
        tags=["alpha", "beta"],
        overwrite_existing=True,
    )

    assert result is True

    with Image.open(path) as image:
        exif_blob = image.info.get("exif")
    assert exif_blob

    exif_dict = piexif.load(exif_blob)
    description = _decode_caption(exif_dict["0th"].get(piexif.ImageIFD.ImageDescription, b""))
    keywords = _decode_keywords(exif_dict["0th"].get(piexif.ImageIFD.XPKeywords, b""))

    assert description == "Test caption"
    assert keywords == ["alpha", "beta"]


def test_skip_overwriting_existing_metadata(tmp_path):
    path = tmp_path / "existing.jpg"
    _create_jpeg_with_metadata(path, caption="Original caption", tags=["original"])

    writer = MetadataWriter()
    result = writer.write(
        path,
        caption="New caption",
        tags=["new-tag"],
        overwrite_existing=False,
    )

    assert result is False

    with Image.open(path) as image:
        exif_dict = piexif.load(image.info.get("exif"))
    stored_caption = _decode_caption(exif_dict["0th"].get(piexif.ImageIFD.ImageDescription, b""))
    decoded_keywords = _decode_keywords(exif_dict["0th"].get(XP_KEYWORDS_TAG, b""))

    assert stored_caption == "Original caption"
    assert decoded_keywords == ["original"]


def test_overwrite_existing_metadata_when_enabled(tmp_path):
    path = tmp_path / "overwrite.jpg"
    _create_jpeg_with_metadata(path, caption="Original caption", tags=["original"])

    writer = MetadataWriter()
    result = writer.write(
        path,
        caption="New caption",
        tags=["new-tag"],
        overwrite_existing=True,
    )

    assert result is True

    with Image.open(path) as image:
        exif_dict = piexif.load(image.info.get("exif"))
    stored_caption = _decode_caption(exif_dict["0th"].get(piexif.ImageIFD.ImageDescription, b""))
    decoded_keywords = _decode_keywords(exif_dict["0th"].get(XP_KEYWORDS_TAG, b""))

    assert stored_caption == "New caption"
    assert decoded_keywords == ["new-tag"]


def test_add_tags_when_missing_does_not_overwrite_caption(tmp_path):
    path = tmp_path / "add-tags.jpg"
    _create_jpeg_with_metadata(path, caption="Original caption", tags=None)

    writer = MetadataWriter()
    result = writer.write(
        path,
        caption=None,
        tags=["tag-a", "tag-b"],
        overwrite_existing=False,
    )

    assert result is True

    with Image.open(path) as image:
        exif_dict = piexif.load(image.info.get("exif"))
    stored_caption = _decode_caption(exif_dict["0th"].get(piexif.ImageIFD.ImageDescription, b""))
    decoded_keywords = _decode_keywords(exif_dict["0th"].get(XP_KEYWORDS_TAG, b""))

    assert stored_caption == "Original caption"
    assert decoded_keywords == ["tag-a", "tag-b"]


def test_tuple_encoded_keywords_do_not_break(tmp_path):
    path = tmp_path / "tuple-keywords.jpg"
    Image.new("RGB", (8, 8), color=(20, 20, 20)).save(path, format="JPEG")

    payload = "existing".encode("utf-16le") + b"\x00\x00"
    exif_dict = {
        "0th": {XP_KEYWORDS_TAG: tuple(payload)},
        "Exif": {},
        "GPS": {},
        "1st": {},
        "Interop": {},
        "thumbnail": None,
    }
    exif_bytes = piexif.dump(exif_dict)
    with Image.open(path) as image:
        image.save(path, exif=exif_bytes)

    writer = MetadataWriter()
    result = writer.write(
        path,
        caption="Caption",
        tags=["one", "two"],
        overwrite_existing=True,
    )

    assert result is True

    with Image.open(path) as image:
        exif_blob = image.info.get("exif")
    exif_dict = piexif.load(exif_blob)
    stored_caption = _decode_caption(exif_dict["0th"].get(piexif.ImageIFD.ImageDescription, b""))
    decoded_keywords = _decode_keywords(exif_dict["0th"].get(XP_KEYWORDS_TAG, b""))

    assert stored_caption == "Caption"
    assert decoded_keywords == ["one", "two"]


def test_write_png_metadata(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (10, 10), color=(0, 0, 0)).save(path, format="PNG")

    writer = MetadataWriter()
    result = writer.write(
        path,
        caption="PNG caption",
        tags=["one", "two"],
        overwrite_existing=True,
    )

    assert result is True
    with Image.open(path) as image:
        info = dict(image.info)
    assert info.get("Description") == "PNG caption"
    assert info.get("Keywords") == "one,two"


def test_write_rejects_unknown_extension(tmp_path):
    path = tmp_path / "image.bmp"
    Image.new("RGB", (4, 4)).save(path, format="BMP")
    writer = MetadataWriter()
    with pytest.raises(UnsupportedFormatError):
        writer.write(path, caption="x", tags=["y"])


def test_helper_conversions_cover_edge_cases():
    assert _ensure_bytes(bytearray(b"abc")) == b"abc"
    assert _ensure_bytes(memoryview(b"xyz")) == b"xyz"
    assert _ensure_bytes([255, 0, 1]) == b"\xff\x00\x01"
    assert _decode_utf8(b" hello ") == "hello"
    assert _decode_utf8(b"") == ""
    assert _decode_xp_keywords("a\x00b\x00".encode("utf-16le")) == ["a", "b"]
