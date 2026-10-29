"""Character error rate, word error rate, latency, and failure rate.

Raw transcriptions are never rewritten by this module. Metrics use a separate
normalized copy of both the prediction and the reference.

Normalization (applied to both sides, and only for scoring):
- Unicode NFKC
- remove the byte-order mark and zero-width characters
- turn newlines and tabs into spaces
- collapse repeated spaces
- strip the ends

Case, punctuation, and Arabic letter forms are kept. Folding them would hide
errors in dosages, decimals, and dots. Line breaks therefore do not count as
their own characters after normalization; a word split only by a newline is
joined for WER.

CER is the Levenshtein distance divided by the number of reference characters.
WER is the same distance on whitespace-separated tokens, divided by the number
of reference tokens. When the reference is empty, the rate is 0 if the
prediction is also empty and 1 otherwise.

Three views of the error rate are reported per variant:

- ``cer`` / ``wer``: mean of per-item rates over successful items only.
- ``cer_micro`` / ``wer_micro``: total edits divided by total reference length,
  over successful items only. Long references weigh more, as they should.
- ``cer_all`` / ``wer_all``: like micro, but a failed recognition counts as a
  fully wrong transcription (every reference character or word is an error).
  This is the honest headline number: failing to read a page is not free.

Failed recognitions are also counted in the failure rate. Latency is averaged
over every attempt, including failures.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

NORMALIZATION = (
    "Unicode NFKC; drop BOM and zero-width characters; replace newlines and tabs "
    "with spaces; collapse repeated spaces; strip ends. Case and punctuation stay. "
    "Applied only when scoring, and applied to both the prediction and the reference."
)

_ZERO_WIDTH = {
    "\ufeff",
    "\u200b",
    "\u200c",
    "\u200d",
    "\u2060",
}


def normalize_for_metrics(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = "".join(ch for ch in normalized if ch not in _ZERO_WIDTH)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\n", " ").replace("\t", " ")
    return " ".join(normalized.split())


def levenshtein(reference: Iterable, hypothesis: Iterable) -> int:
    ref = list(reference)
    hyp = list(hypothesis)
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    previous = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, start=1):
        current = [i]
        for j, hyp_item in enumerate(hyp, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ref_item != hyp_item)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _rate(distance: int, reference_length: int, hypothesis_length: int) -> float:
    if reference_length == 0:
        return 0.0 if hypothesis_length == 0 else 1.0
    return distance / reference_length


def character_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_for_metrics(reference)
    hyp = normalize_for_metrics(hypothesis)
    return _rate(levenshtein(ref, hyp), len(ref), len(hyp))


def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = normalize_for_metrics(reference).split()
    hyp = normalize_for_metrics(hypothesis).split()
    return _rate(levenshtein(ref, hyp), len(ref), len(hyp))


@dataclass
class ItemScore:
    image_id: str
    variant: str
    status: str
    cer: float | None
    wer: float | None
    latency_ms: float | None
    error: str | None = None
    ref_chars: int = 0
    char_edits: int = 0
    ref_words: int = 0
    word_edits: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VariantScore:
    variant: str
    count: int
    failure_rate: float
    mean_latency_ms: float | None
    cer: float | None
    wer: float | None
    scored: int
    cer_micro: float | None = None
    wer_micro: float | None = None
    cer_all: float | None = None
    wer_all: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationReport:
    normalization: str
    variants: list[VariantScore]
    items: list[ItemScore] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "normalization": self.normalization,
            "variants": [variant.to_dict() for variant in self.variants],
            "items": [item.to_dict() for item in self.items],
        }


def _edit_stats(reference: str, hypothesis: str) -> tuple[int, int, int, int]:
    """Return (char edits, reference chars, word edits, reference words)."""
    ref = normalize_for_metrics(reference)
    hyp = normalize_for_metrics(hypothesis)
    ref_words, hyp_words = ref.split(), hyp.split()
    return (
        levenshtein(ref, hyp),
        len(ref),
        levenshtein(ref_words, hyp_words),
        len(ref_words),
    )


def score_predictions(samples: list[dict], predictions: list[dict]) -> EvaluationReport:
    """Score prediction records against dataset samples.

    Each sample needs ``id`` and ``text``. Each prediction needs ``image_id``,
    ``variant``, ``status``, and, on success, ``text``. ``latency_ms`` is optional.
    """
    references = {sample["id"]: sample["text"] for sample in samples}
    grouped: dict[str, list[ItemScore]] = {}
    items: list[ItemScore] = []
    for prediction in predictions:
        image_id = prediction["image_id"]
        if image_id not in references:
            raise KeyError(f"No ground truth for image id {image_id!r}")
        status = prediction.get("status", "success")
        latency = prediction.get("latency_ms")
        if status != "success" or prediction.get("text") is None:
            ref_norm = normalize_for_metrics(references[image_id])
            ref_word_count = len(ref_norm.split())
            item = ItemScore(
                image_id=image_id,
                variant=prediction["variant"],
                status="error",
                cer=None,
                wer=None,
                latency_ms=latency,
                error=prediction.get("error") or "recognition failed",
                ref_chars=len(ref_norm),
                char_edits=len(ref_norm),  # unread page: every character is an error
                ref_words=ref_word_count,
                word_edits=ref_word_count,
            )
        else:
            reference = references[image_id]
            hypothesis = prediction["text"]
            char_edits, ref_chars, word_edits, ref_words = _edit_stats(reference, hypothesis)
            item = ItemScore(
                image_id=image_id,
                variant=prediction["variant"],
                status="success",
                cer=round(character_error_rate(reference, hypothesis), 6),
                wer=round(word_error_rate(reference, hypothesis), 6),
                latency_ms=latency,
                error=None,
                ref_chars=ref_chars,
                char_edits=char_edits,
                ref_words=ref_words,
                word_edits=word_edits,
            )
        items.append(item)
        grouped.setdefault(item.variant, []).append(item)

    summaries = [_summarize(variant, scores) for variant, scores in grouped.items()]
    return EvaluationReport(normalization=NORMALIZATION, variants=summaries, items=items)


def _summarize(variant: str, scores: list[ItemScore]) -> VariantScore:
    failures = sum(1 for score in scores if score.status != "success")
    latencies = [score.latency_ms for score in scores if score.latency_ms is not None]
    scored = [score for score in scores if score.cer is not None and score.wer is not None]
    return VariantScore(
        variant=variant,
        count=len(scores),
        failure_rate=failures / len(scores) if scores else 0.0,
        mean_latency_ms=(sum(latencies) / len(latencies)) if latencies else None,
        cer=(sum(score.cer for score in scored) / len(scored)) if scored else None,
        wer=(sum(score.wer for score in scored) / len(scored)) if scored else None,
        scored=len(scored),
        cer_micro=_pooled(scored, "char_edits", "ref_chars"),
        wer_micro=_pooled(scored, "word_edits", "ref_words"),
        cer_all=_pooled(scores, "char_edits", "ref_chars"),
        wer_all=_pooled(scores, "word_edits", "ref_words"),
    )


def _pooled(scores: list[ItemScore], edits: str, length: str) -> float | None:
    """Total edits over total reference length. None when nothing was scored."""
    if not scores:
        return None
    total_length = sum(getattr(score, length) for score in scores)
    total_edits = sum(getattr(score, edits) for score in scores)
    if total_length == 0:
        return 0.0 if total_edits == 0 else 1.0
    return round(total_edits / total_length, 6)


def load_dataset(directory: str | Path) -> list[dict]:
    manifest = Path(directory) / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    samples = payload["samples"] if isinstance(payload, dict) else payload
    loaded = []
    for sample in samples:
        item = dict(sample)
        item["text"] = (Path(directory) / sample["text_file"]).read_text(encoding="utf-8")
        item["image_path"] = str(Path(directory) / sample["image"])
        loaded.append(item)
    return loaded
