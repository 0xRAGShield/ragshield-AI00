"""Name -> engine factory. Add an engine by registering one."""

from __future__ import annotations

from typing import Callable

from ocr.engines.base import Engine
from ocr.engines.paddle import PaddleEngine
from ocr.engines.qwen import QwenEngine
from ocr.engines.surya import SuryaEngine

_FACTORIES: dict[str, Callable[..., Engine]] = {
    "qwen": QwenEngine,
    "paddleocr": PaddleEngine,
    "surya": SuryaEngine,
}


def register_engine(name: str, factory: Callable[..., Engine]) -> None:
    _FACTORIES[name] = factory


def engine_names() -> list[str]:
    return sorted(_FACTORIES)


def create_engine(name: str, **options) -> Engine:
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise ValueError(f"Unknown engine {name!r}. Available: {', '.join(engine_names())}") from None
    return factory(**options)


def engine_status() -> list[dict]:
    """Availability of every registered engine without loading any model."""
    rows = []
    for name in engine_names():
        engine = create_engine(name)
        usable, detail = engine.availability()
        rows.append({"name": name, "available": usable, "detail": detail,
                     "description": engine.description, "license": engine.license_note})
    return rows
