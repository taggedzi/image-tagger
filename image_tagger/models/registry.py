"""Registry for dynamically discovering tagging models."""

from __future__ import annotations

from importlib import import_module
from typing import Callable, Dict, Iterable, Optional

from .base import ModelInfo, TaggingModel


Factory = Callable[[], TaggingModel]


class ModelRegistry:
    """Tracks available model factories and lazily loads them on demand."""

    _factories: Dict[str, Factory] = {}
    _bootstrap_complete: bool = False

    @classmethod
    def register(cls, name: str, factory: Factory) -> None:
        """Register a model factory under the provided name."""
        cls._factories[name] = factory

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._factories.pop(name, None)

    @classmethod
    def ensure_bootstrapped(cls) -> None:
        if cls._bootstrap_complete:
            return
        import_module("image_tagger.models.builtin.simple")
        cls._bootstrap_complete = True

    @classmethod
    def list_model_infos(cls) -> list[ModelInfo]:
        """Return metadata for all registered models."""
        cls.ensure_bootstrapped()
        infos: list[ModelInfo] = []
        for factory in cls._factories.values():
            infos.append(factory().info())
        return infos

    @classmethod
    def get(cls, name: str) -> TaggingModel:
        cls.ensure_bootstrapped()
        try:
            factory = cls._factories[name]
        except KeyError as exc:
            available = ", ".join(sorted(cls._factories))
            raise KeyError(f"Unknown model '{name}'. Available: {available}") from exc
        instance = factory()
        instance.load()
        return instance

