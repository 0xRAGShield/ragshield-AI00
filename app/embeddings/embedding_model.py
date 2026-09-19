from __future__ import annotations

from sentence_transformers import SentenceTransformer


class BGE_M3_Embedding:

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 8,
    ) -> None:

        if not model_name or not model_name.strip():
            raise ValueError(
                "model_name cannot be empty."
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        self.model_name = model_name
        self.batch_size = batch_size

        try:
            self._model = SentenceTransformer(
                model_name
            )
        except Exception as error:
            raise RuntimeError(
                f"Failed to load embedding model "
                f"'{model_name}'."
            ) from error

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not isinstance(texts, list):
            raise TypeError(
                "texts must be a list of strings."
            )

        if not texts:
            return []

        if any(
            not isinstance(text, str)
            for text in texts
        ):
            raise TypeError(
                "All items in texts must be strings."
            )

        if any(
            not text.strip()
            for text in texts
        ):
            raise ValueError(
                "texts cannot contain empty strings."
            )

        try:
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
                batch_size=self.batch_size,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to generate embeddings."
            ) from error

        return embeddings.tolist()

    def embed_query(
        self,
        query: str,
    ) -> list[float]:

        if not isinstance(query, str):
            raise TypeError(
                "query must be a string."
            )

        if not query.strip():
            raise ValueError(
                "query cannot be empty."
            )

        try:
            embedding = self._model.encode(
                query.strip(),
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to generate query embedding."
            ) from error

        return embedding.tolist()

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()