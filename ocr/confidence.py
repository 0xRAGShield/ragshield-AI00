"""Token-level confidence from the recognizer itself.

While the model decodes, the probability of each chosen token is recorded. The
geometric mean of those probabilities is a cheap signal for "the model was
unsure here". It is NOT an accuracy: a model can be confidently wrong, and the
numbers are not calibrated. Use them to rank lines for human review, and pick a
threshold against a labelled set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

# A token the model gave less than a 50% chance counts as "low".
LOW_TOKEN_LOGPROB = math.log(0.5)


@dataclass
class TokenStats:
    count: int = 0
    total_logprob: float = 0.0
    min_logprob: float = 0.0
    low_count: int = 0

    def add(self, logprobs: Iterable[float]) -> "TokenStats":
        for value in logprobs:
            if self.count == 0 or value < self.min_logprob:
                self.min_logprob = value
            self.count += 1
            self.total_logprob += value
            if value < LOW_TOKEN_LOGPROB:
                self.low_count += 1
        return self

    def merge(self, other: "TokenStats") -> "TokenStats":
        if other.count:
            self.min_logprob = other.min_logprob if self.count == 0 else min(self.min_logprob, other.min_logprob)
            self.count += other.count
            self.total_logprob += other.total_logprob
            self.low_count += other.low_count
        return self

    @property
    def confidence(self) -> float | None:
        """Geometric-mean token probability in (0, 1], or None when nothing was recorded."""
        return math.exp(self.total_logprob / self.count) if self.count else None

    @property
    def min_token_prob(self) -> float | None:
        return math.exp(self.min_logprob) if self.count else None

    @property
    def low_fraction(self) -> float | None:
        return self.low_count / self.count if self.count else None


def stats_from_logprobs(values: Iterable[float]) -> TokenStats:
    return TokenStats().add(values)


class ConfidenceRecorder:
    """Logits processor that records the top token's log-probability at every step.

    With greedy decoding the top token is the chosen token. It never changes the scores.
    """

    def __init__(self) -> None:
        self._steps: list = []

    def __call__(self, input_ids, scores):
        import torch

        self._steps.append(torch.log_softmax(scores.float(), dim=-1).max(dim=-1).values.detach())
        return scores

    def stats(self) -> TokenStats:
        if not self._steps:
            return TokenStats()
        import torch

        values = torch.stack(self._steps)[:, 0].cpu().tolist()  # first (only) sequence
        return stats_from_logprobs(values)


def make_recorder():
    """Return (recorder, LogitsProcessorList) or (None, None) when torch/transformers are absent."""
    try:
        import torch  # noqa: F401
        from transformers import LogitsProcessor, LogitsProcessorList
    except ImportError:
        return None, None

    class _Recorder(ConfidenceRecorder, LogitsProcessor):
        pass

    recorder = _Recorder()
    return recorder, LogitsProcessorList([recorder])
