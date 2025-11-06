"""Registry for dynamically discovering tagging models."""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Callable, Dict

from ..config import AppConfig
from .base import ModelInfo, TaggingModel


Factory = Callable[..., TaggingModel]
logger = logging.getLogger(__name__)


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
        modules = [
            "image_tagger.models.blip",
            "image_tagger.models.vision_remote",
        ]
        for module_name in modules:
            try:
                import_module(module_name)
            except ImportError as exc:  # pragma: no cover - optional dependency paths
                logger.debug("Optional model module %s could not be imported: %s", module_name, exc)
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
    def get(cls, name: str, *, config: AppConfig | None = None) -> TaggingModel:
        cls.ensure_bootstrapped()
        try:
            factory = cls._factories[name]
        except KeyError as exc:
            available = ", ".join(sorted(cls._factories))
            raise KeyError(f"Unknown model '{name}'. Available: {available}") from exc
        instance = cls._instantiate_factory(factory, config=config)
        instance.load()
        return instance

    @staticmethod
    def _instantiate_factory(
        factory: Callable[..., TaggingModel], *, config: AppConfig | None
    ) -> TaggingModel:
        if config is not None:
            try:
                return factory(config)
            except TypeError:
                logger.debug(
                    "Factory %s does not accept configuration parameter; instantiating without it.",
                    getattr(factory, "__name__", repr(factory)),
                )
        return factory()
