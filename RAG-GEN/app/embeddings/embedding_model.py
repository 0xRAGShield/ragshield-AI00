from __future__ import annotations

from sentence_transformers import SentenceTransformer


class BGE_M3_Embedding:
    """
    BGE-M3 embedding model wrapper.

    Responsibilities:
    - Load and own the embedding model.
    - Generate normalized document embeddings.
    - Generate normalized query embeddings.
    - Expose the embedding dimension.

    Forbidden responsibilities:
    - Chunking.
    - Cleaning or normalization of source text.
    - Vector-store operations.
    - Retrieval.
    - Reranking.
    - Business logic.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        batch_size: int = 8,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer.")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self._model_name = model_name.strip()
        self._batch_size = batch_size

        try:
            self._model = SentenceTransformer(self._model_name)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load embedding model '{self._model_name}'."
            ) from error

        try:
            dimension = self._model.get_embedding_dimension()
        except Exception as error:
            raise RuntimeError(
                "Failed to determine embedding dimension."
            ) from error

        if not isinstance(dimension, int) or dimension <= 0:
            raise RuntimeError(
                f"Invalid embedding dimension returned by model: {dimension!r}."
            )

        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Generate normalized embeddings for document/chunk texts.

        Output order is guaranteed to match input order.
        """
        self._validate_texts(texts)

        if not texts:
            return []

        try:
            embeddings = self._model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to generate document embeddings."
            ) from error

        return self._convert_embeddings(embeddings, len(texts))

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Generate one normalized embedding for a query.
        """
        if not isinstance(query, str):
            raise TypeError("query must be a string.")

        query = query.strip()

        if not query:
            raise ValueError("query cannot be empty.")

        try:
            embedding = self._model.encode(
                query,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as error:
            raise RuntimeError(
                "Failed to generate query embedding."
            ) from error

        converted = self._convert_embeddings(embedding, 1)

        return converted[0]

    def _validate_texts(
        self,
        texts: list[str],
    ) -> None:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings.")

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"texts[{index}] must be a string."
                )

            if not text.strip():
                raise ValueError(
                    f"texts[{index}] cannot be empty."
                )

    def _convert_embeddings(
        self,
        embeddings: object,
        expected_count: int,
    ) -> list[list[float]]:
        try:
            converted = embeddings.tolist()
        except AttributeError as error:
            raise RuntimeError(
                "Embedding model returned an unsupported output type."
            ) from error

        if expected_count == 1:
            if not converted:
                raise RuntimeError(
                    "Embedding model returned an empty embedding."
                )

            if not isinstance(converted[0], (int, float)):
                raise RuntimeError(
                    "Embedding model returned an invalid embedding shape."
                )

            result = [converted]
        else:
            result = converted

        if not isinstance(result, list):
            raise RuntimeError(
                "Embedding model returned an invalid embedding collection."
            )

        if len(result) != expected_count:
            raise RuntimeError(
                "Embedding count does not match input text count."
            )

        for index, vector in enumerate(result):
            if not isinstance(vector, list):
                raise RuntimeError(
                    f"Embedding at index {index} is not a list."
                )

            if len(vector) != self._dimension:
                raise RuntimeError(
                    f"Embedding at index {index} has dimension "
                    f"{len(vector)}; expected {self._dimension}."
                )

            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                for value in vector
            ):
                raise RuntimeError(
                    f"Embedding at index {index} contains non-numeric values."
                )

        return result