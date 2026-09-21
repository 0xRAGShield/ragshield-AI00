from __future__ import annotations

from typing import Sequence

from sentence_transformers import CrossEncoder


class CrossEncoderRerankerModel:
  
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:

        if not model_name.strip():
            raise ValueError(
                "model_name cannot be empty."
            )

        self._model_name = model_name

        self._model = CrossEncoder(
            model_name
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def score_pairs(
        self,
        query: str,
        texts: list[str],
    ) -> list[float | None]:

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        if not isinstance(texts, list):
            raise TypeError(
                "texts must be a list."
            )

        if not texts:
            return []

        pairs = [
            [query, text]
            for text in texts
        ]

        scores = self._model.predict(
            pairs
        )

        return [
            float(score)
            for score in scores
        ]