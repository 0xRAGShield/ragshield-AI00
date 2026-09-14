from __future__ import annotations

from sentence_transformers import SentenceTransformer


class BGE_M3_Embedding:

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
    ) -> None:

        self.model_name = model_name

        self._model = SentenceTransformer(
            model_name
        )

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=8,
        )

        return embeddings.tolist()

    def embed_query(
        self,
        query: str,
    ) -> list[float]:

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        embedding = self._model.encode(
            query.strip(),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return embedding.tolist()

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()