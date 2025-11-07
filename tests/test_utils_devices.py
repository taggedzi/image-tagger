"""Tests for hardware detection helpers."""

from __future__ import annotations

from types import SimpleNamespace

from image_tagger.utils import devices


def test_detect_torch_device_without_torch(monkeypatch):
    monkeypatch.setattr(devices, "torch", None)

    device, message = devices.detect_torch_device()

    assert device == "cpu"
    assert "PyTorch is not installed" in message


def test_detect_torch_device_prefers_cuda(monkeypatch):
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def current_device() -> int:
            return 0

        @staticmethod
        def get_device_name(index: int) -> str:
            return f"Fake GPU {index}"

    fake_torch = SimpleNamespace(
        cuda=FakeCuda(),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        xpu=SimpleNamespace(is_available=lambda: False),
        device=lambda name: f"torch-device:{name}",
    )
    monkeypatch.setattr(devices, "torch", fake_torch)

    device, message = devices.detect_torch_device()

    assert device == "cuda:0"
    assert "CUDA device detected" in message


def test_detect_torch_device_mps(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        xpu=SimpleNamespace(is_available=lambda: False),
        device=lambda name: f"torch-device:{name}",
    )
    monkeypatch.setattr(devices, "torch", fake_torch)

    device, message = devices.detect_torch_device("mps")

    assert device == "mps"
    assert "MPS backend" in message


def test_detect_torch_device_xpu(monkeypatch):
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
        xpu=SimpleNamespace(is_available=lambda: True),
        device=lambda name: f"torch-device:{name}",
    )
    monkeypatch.setattr(devices, "torch", fake_torch)

    device, message = devices.detect_torch_device("xpu")

    assert device == "xpu"
    assert "XPU backend" in message


def test_torch_device_with_and_without_torch(monkeypatch):
    monkeypatch.setattr(devices, "torch", None)
    assert devices.torch_device("cpu") == "cpu"

    monkeypatch.setattr(devices, "torch", SimpleNamespace(device=lambda name: f"dev:{name}"))
    assert devices.torch_device("xpu") == "dev:xpu"
