from __future__ import annotations

import asyncio
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.context.context_builder import ContextBuilder
from app.core.settings import get_settings
from app.embeddings.embedding_model import BGE_M3_Embedding
from app.generation.llm.llm import build_llm_runtime
from app.generation.prompts.prompt_builder import PromptBuilder
from app.pipeline import RAGPipeline
from app.retrieval.ranker_model import CrossEncoderRerankerModel
from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever
from app.vector_store.vector_store import QdrantStore


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=100_000)
    top_k: int | None = Field(default=None, ge=1, le=100)
    top_n: int | None = Field(default=None, ge=1, le=10)
    max_new_tokens: int | None = Field(default=None, ge=1, le=1024)


class QueryResponse(BaseModel):
    answer: str
    trace_id: str
    degraded: bool
    retrieved_candidates: int
    reranked_candidates: int
    evidence_items: int


settings = get_settings()
pipeline: RAGPipeline | None = None


def build_pipeline() -> RAGPipeline:
    embedding_model = BGE_M3_Embedding()

    vector_store = QdrantStore(
        collection_name=settings.retrieval.collection_name,
        host=settings.retrieval.qdrant_host,
        port=settings.retrieval.qdrant_port,
        vector_size=embedding_model.dimension,
    )

    retriever = Retriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        settings=settings.retrieval,
    )

    reranker_model = CrossEncoderRerankerModel()

    reranker = Reranker(
        model=reranker_model,
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

    return RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm=llm,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline

    pipeline = build_pipeline()
    await pipeline.startup()

    try:
        yield
    finally:
        await pipeline.shutdown()


app = FastAPI(
    title="RAGShield Base RAG API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline is not ready.",
        )

    try:
        result = await pipeline.run(
            request.query,
            top_k=request.top_k,
            top_n=request.top_n,
            max_new_tokens=request.max_new_tokens,
        )

    except Exception as exc:
        print(
            f"RAG pipeline error: "
            f"{type(exc).__name__}: {exc}"
        )
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail="RAG pipeline execution failed.",
        ) from exc

    return QueryResponse(
        answer=result.generation.text,
        trace_id=result.trace_id,
        degraded=result.is_degraded,
        retrieved_candidates=len(result.retrieval.candidates),
        reranked_candidates=len(result.reranking.candidates),
        evidence_items=len(result.context.evidence),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "RAG_GEN:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )