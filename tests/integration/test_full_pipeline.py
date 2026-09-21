import os

import pytest

from app.context.context_builder import (
    ContextBuilder,
)
from app.core.settings import get_settings
from app.embeddings.embedding_model import (
    BGE_M3_Embedding,
)
from app.generation.llm.llm import (
    build_llm_runtime,
)
from app.generation.prompts.prompt_builder import (
    PromptBuilder,
)
from app.pipeline import RAGPipeline
from app.retrieval.reranker import Reranker
from app.retrieval.ranker_model import (
    CrossEncoderRerankerModel,
)
from app.retrieval.retriever import Retriever
from app.vector_store.vector_store import (
    QdrantStore,
)


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.getenv("RUN_REAL_RAG") != "1",
    reason=(
        "Set RUN_REAL_RAG=1 to run the real "
        "Qdrant + BGE-M3 + CrossEncoder + Qwen test."
    ),
)
async def test_real_full_rag_pipeline():

    settings = get_settings()

    embedding_model = (
        BGE_M3_Embedding()
    )

    vector_store = QdrantStore(
        collection_name=(
            settings.retrieval.collection_name
        ),
        host="localhost",
        port=6333,
        vector_size=embedding_model.dimension,
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        settings=settings.retrieval,
    )

    reranker = Reranker(
        model=CrossEncoderRerankerModel(),
        settings=settings.reranker,
    )

    context_builder = ContextBuilder(
        settings=settings.context,
    )

    prompt_builder = PromptBuilder(
        settings=settings.prompt,
    )

    llm = build_llm_runtime(
        settings=settings.llm,
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm=llm,
    )

    await pipeline.startup()

    try:
        result = await pipeline.run(
            "What is diabetes?"
        )

        assert result.query

        assert result.retrieval is not None

        assert result.reranking is not None

        assert result.context is not None

        assert result.prompt is not None

        assert result.generation is not None

        assert isinstance(
            result.generation.text,
            str,
        )

    finally:
        await pipeline.shutdown()