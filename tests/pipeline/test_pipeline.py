import pytest

from app.models.rag_models import (
    AssembledContext,
    ContextBudgetStats,
    ContextBuilderMetrics,
    LLMPrompt,
    RetrievalCandidate,
    RetrievalMetrics,
    RetrievalResult,
    RerankedCandidate,
    RerankedResult,
)
from app.pipeline import RAGPipeline


class FakeRetriever:

    async def retrieve(self, query):
        return RetrievalResult(
            query=query,
            candidates=[
                RetrievalCandidate(
                    chunk_id="chunk-1",
                    source_id="source-1",
                    text="Medical evidence text.",
                    score=0.95,
                    metadata={
                        "file_name": "test.txt"
                    },
                )
            ],
            metrics=RetrievalMetrics(
                trace_id="trace-1",
                requested_top_k=5,
                returned_candidates=1,
            ),
        )

    def close(self):
        pass


class FakeReranker:

    async def rerank(
        self,
        retrieval_result,
        *,
        top_n=None,
    ):
        return RerankedResult(
            query=retrieval_result.query.text,
            candidates=[
                RerankedCandidate(
                    chunk_id="chunk-1",
                    source_id="source-1",
                    text="Medical evidence text.",
                    original_score=0.95,
                    rerank_score=0.99,
                    original_rank=0,
                    rank=0,
                    metadata={
                        "file_name": "test.txt"
                    },
                )
            ],
            trace_id="trace-1",
        )

    def close(self):
        pass


class FakeContextBuilder:

    def build(self, result):
        return AssembledContext(
            query=result.query,
            evidence=[],
            budget=ContextBudgetStats(
                max_context_tokens=1000,
            ),
            metrics=ContextBuilderMetrics(
                input_candidate_count=1,
            ),
            trace_id=result.trace_id,
        )

    def close(self):
        pass


class FakePromptBuilder:

    def build(
        self,
        query,
        context,
        *,
        request=None,
    ):
        return LLMPrompt(
            messages=[
                {
                    "role": "system",
                    "content": "Answer using evidence.",
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
            trace_id="trace-1",
        )


class FakeGeneration:

    text = "Test answer"
    finish_reason = "stop"
    output_token_count = 3
    latency_ms = 1.0
    model_id = "fake"
    backend_id = "fake"
    quantization = "none"
    modality = "text"
    trace_id = "trace-1"
    streamed = False
    model_fingerprint = "fake"


class FakeLLM:

    async def startup(self):
        pass

    async def shutdown(self):
        pass

    async def generate(self, request):
        return FakeGeneration()


@pytest.mark.asyncio
async def test_full_base_rag_pipeline():

    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        context_builder=FakeContextBuilder(),
        prompt_builder=FakePromptBuilder(),
        llm=FakeLLM(),
    )

    await pipeline.startup()

    result = await pipeline.run(
        "What is diabetes?"
    )

    assert result.query == "What is diabetes?"

    assert len(
        result.retrieval.candidates
    ) == 1

    assert len(
        result.reranking.candidates
    ) == 1

    assert result.generation.text == (
        "Test answer"
    )

    assert result.trace_id