"""Tests for filesystem helper utilities."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from image_tagger.utils import paths as paths_module
from image_tagger.utils.paths import is_image_file, resolve_image_paths


def test_is_image_file_with_custom_extensions(tmp_path):
    path = tmp_path / "sample.custom"
    path.write_text("data", encoding="utf-8")

    assert not is_image_file(path)
    assert is_image_file(path, extensions=[".custom"])


def test_resolve_image_paths_filters_hidden(tmp_path):
    root = tmp_path / "root"
    subdir = root / "nested"
    subdir.mkdir(parents=True)

    visible = root / "visible.jpg"
    hidden = root / ".secret.png"
    nested = subdir / "nested.webp"
    ignored = subdir / "notes.txt"
    for candidate in (visible, hidden, nested):
        candidate.write_text("placeholder", encoding="utf-8")
    ignored.write_text("text", encoding="utf-8")

    # Default is recursive with hidden files excluded.
    discovered = resolve_image_paths(root)
    assert discovered == [nested, visible]

    # Hidden files become visible when include_hidden=True.
    discovered_with_hidden = resolve_image_paths(
        root,
        include_hidden=True,
        extensions=[".jpg", ".png", ".webp"],
    )
    assert discovered_with_hidden == [hidden, nested, visible]

    # Non-recursive traversal should only list files at the top level.
    top_level_only = resolve_image_paths(
        root,
        recursive=False,
        include_hidden=True,
    )
    assert top_level_only == [hidden, visible]


def test_resolve_image_paths_missing_target(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_image_paths(tmp_path / "missing")


def test_resolve_image_paths_file_mode_hidden(monkeypatch, tmp_path):
    image_path = tmp_path / "file.jpg"
    image_path.write_text("data", encoding="utf-8")

    monkeypatch.setattr("image_tagger.utils.paths._is_hidden", lambda *_: True)
    assert resolve_image_paths(image_path, include_hidden=False) == []
    assert resolve_image_paths(image_path, include_hidden=True) == [image_path]


def test_is_hidden_windows_ctypes(monkeypatch, tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")

    class Kernel32:
        @staticmethod
        def GetFileAttributesW(path):
            return 2

    dummy_ctypes = SimpleNamespace(windll=SimpleNamespace(kernel32=Kernel32()))
    monkeypatch.setitem(sys.modules, "ctypes", dummy_ctypes)

    assert paths_module._is_hidden(target) is True


def test_is_hidden_windows_missing_attr(monkeypatch, tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")

    class Kernel32:
        @staticmethod
        def GetFileAttributesW(path):
            return -1

    dummy_ctypes = SimpleNamespace(windll=SimpleNamespace(kernel32=Kernel32()))
    monkeypatch.setitem(sys.modules, "ctypes", dummy_ctypes)

    assert paths_module._is_hidden(target) is False
