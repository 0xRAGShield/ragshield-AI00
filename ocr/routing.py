"""Heuristic line routing. Not a calibrated language classifier."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

ENGINE_QWEN = "qwen"
ENGINE_TROCR = "trocr"
ENGINE_QWEN_FALLBACK = "qwen_fallback"
ENGINE_MANUAL_REVIEW = "manual_review"

_ARABIC = re.compile(r"[\u0600-\u06FF]")
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"\d")
_SAFETY = re.compile(
    r"(?i)\b(mg|mcg|ml|iu|tab|tablet|cap|capsule|drop|syrup|bid|tid|qid|"
    r"daily|hour|hrs|x\s*\d)\b|[\d]+(\.\d+)?\s*(mg|ml|mcg)|[×x]\s*\d"
)


@dataclass
class RoutingConfig:
    policy: str = "english_line_heuristic"
    enable_qwen_fallback: bool = True
    min_english_aspect: float = 3.0
    min_english_height: int = 10
    max_english_height: int = 96
    max_ink_ratio: float = 0.45
    min_ink_ratio: float = 0.012


@dataclass
class RoutingDecision:
    engine: str
    reason: str
    review_required: bool = True
    script_guess: str = "unknown"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "reason": self.reason,
            "review_required": self.review_required,
            "script_guess": self.script_guess,
            "warnings": list(self.warnings),
        }


def route_line(
    image: Image.Image,
    config: RoutingConfig | None = None,
) -> RoutingDecision:
    """Choose an engine from crop geometry and ink. Script is not known yet."""
    config = config or RoutingConfig()
    warnings = [
        "Routing is a heuristic. It is not a handwriting language classifier "
        "and is not proof of script identity."
    ]
    width, height = image.size
    ink = _ink_ratio(image)
    aspect = width / max(height, 1)

    if width < 8 or height < 8 or ink < config.min_ink_ratio:
        return RoutingDecision(
            engine=ENGINE_QWEN,
            reason="empty_or_tiny_crop; default Qwen and review",
            review_required=True,
            script_guess="unknown",
            warnings=warnings + ["crop may be empty or too small"],
        )

    if _looks_numeric_line(image, aspect, height):
        return RoutingDecision(
            engine=ENGINE_QWEN,
            reason="short/numeric-like geometry; conservative Qwen for dosages",
            review_required=True,
            script_guess="numeric_or_short",
            warnings=warnings,
        )

    if config.policy == "qwen_default":
        return RoutingDecision(
            engine=ENGINE_QWEN,
            reason="policy qwen_default; language unknown before OCR",
            review_required=True,
            script_guess="unknown",
            warnings=warnings,
        )

    if (
        config.policy == "english_line_heuristic"
        and aspect >= config.min_english_aspect
        and config.min_english_height <= height <= config.max_english_height
        and config.min_ink_ratio <= ink <= config.max_ink_ratio
    ):
        return RoutingDecision(
            engine=ENGINE_TROCR,
            reason=(
                "english_line_heuristic: wide short line with moderate ink. "
                "Not a validated English classifier. TrOCR is English-only."
            ),
            review_required=True,
            script_guess="possibly_latin_geometry",
            warnings=warnings,
        )

    return RoutingDecision(
        engine=ENGINE_QWEN,
        reason="unclassified geometry; Qwen is the mixed Arabic-English default",
        review_required=True,
        script_guess="unknown",
        warnings=warnings,
    )


def dominant_direction(texts) -> str:
    """"rtl" when most letters are Arabic, otherwise "ltr". Digits and punctuation are ignored."""
    arabic = latin = 0
    for text in texts:
        if text:
            arabic += len(_ARABIC.findall(text))
            latin += len(_LATIN.findall(text))
    if arabic + latin == 0:
        return "ltr"
    return "rtl" if arabic / (arabic + latin) >= 0.5 else "ltr"


def classify_recognized_script(text: str | None) -> str:
    if not text or not str(text).strip():
        return "empty"
    body = str(text)
    arabic = len(_ARABIC.findall(body))
    latin = len(_LATIN.findall(body))
    digits = len(_DIGIT.findall(body))
    if arabic and latin:
        return "mixed"
    if arabic:
        return "arabic"
    if latin and not arabic:
        return "latin"
    if digits and not arabic and not latin:
        return "numeric"
    return "unknown"


def looks_implausible(text: str | None) -> bool:
    if text is None or not str(text).strip():
        return True
    body = str(text).strip()
    if len(set(body.replace(" ", ""))) <= 1 and len(body) >= 6:
        return True
    return False


def needs_safety_review(text: str | None) -> bool:
    if not text:
        return True
    return bool(_SAFETY.search(text) or _DIGIT.search(text) or _ARABIC.search(text))


def should_fallback_to_qwen(
    text: str | None,
    status: str,
    config: RoutingConfig | None = None,
) -> bool:
    config = config or RoutingConfig()
    if not config.enable_qwen_fallback:
        return False
    if status != "success":
        return True
    script = classify_recognized_script(text)
    if script in {"empty", "arabic", "mixed"}:
        return True
    return looks_implausible(text)


def _ink_ratio(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    ink = gray < 200
    if ink.size == 0:
        return 0.0
    return float(ink.mean())


def _looks_numeric_line(image: Image.Image, aspect: float, height: int) -> bool:
    return aspect >= 1.2 and height <= 28 and image.width <= 140
