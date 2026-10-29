"""Pluggable page-level OCR engines."""

from ocr.engines.base import (
    STATUS_ERROR,
    STATUS_SUCCESS,
    STATUS_UNAVAILABLE,
    Engine,
    EngineLine,
    EngineResult,
)
from ocr.engines.registry import create_engine, engine_names, engine_status, register_engine

__all__ = [
    "STATUS_ERROR", "STATUS_SUCCESS", "STATUS_UNAVAILABLE",
    "Engine", "EngineLine", "EngineResult",
    "create_engine", "engine_names", "engine_status", "register_engine",
]
