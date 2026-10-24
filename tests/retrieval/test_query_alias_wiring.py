import pytest

from app.core.settings import RetrievalSettings, RerankerSettings
from app.models.image_refs import expand_query_with_document_aliases
from app.models.rag_models import (
    RetrievalCandidate,
    RetrievalMetrics,
    RetrievalQuery,
    RetrievalResult,
)
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever


class RecordingEmbedder:
    dimension = 2
    model_name = "fake-embedder"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.1, 0.2]


class EmptyVectorStore:
    collection_name = "ragshield_test"
    vector_size = 2

    def search(self, query_vector, limit=5):
        return []


class RecordingRerankModel:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        self.queries.append(query)
        return [0.4 for _ in texts]


@pytest.mark.asyncio
async def test_retriever_embeds_expanded_query_and_keeps_original():
    embedder = RecordingEmbedder()
    retriever = Retriever(
        embedding_model=embedder,
        vector_store=EmptyVectorStore(),
        settings=RetrievalSettings(),
    )
    original = "عايزة روشتة عزة"

    try:
        result = await retriever.retrieve(RetrievalQuery(text=original))
    finally:
        retriever.close()

    assert result.query.text == original
    assert embedder.queries
    assert embedder.queries[0] == expand_query_with_document_aliases(original)
    assert embedder.queries[0] != original
    assert "prescription" in embedder.queries[0].casefold()


@pytest.mark.asyncio
async def test_reranker_scores_expanded_query_and_keeps_original():
    model = RecordingRerankModel()
    reranker = Reranker(
        model=model,
        settings=RerankerSettings(fallback_to_retrieval=False),
    )
    original = "عايزه روشته منار"
    retrieval = RetrievalResult(
        query=RetrievalQuery(text=original),
        candidates=[
            RetrievalCandidate(
                chunk_id="chunk-1",
                source_id="source-1",
                text=(
                    "Prescription image labeled Manar. "
                    "Document type aliases: prescription, روشتة."
                ),
                score=0.7,
                metadata={},
            )
        ],
        metrics=RetrievalMetrics(trace_id="trace-rerank"),
    )

    try:
        result = await reranker.rerank(retrieval)
    finally:
        reranker.close()

    assert result.query == original
    assert model.queries
    assert model.queries[0] == expand_query_with_document_aliases(original)
    assert "prescription" in model.queries[0].casefold()
