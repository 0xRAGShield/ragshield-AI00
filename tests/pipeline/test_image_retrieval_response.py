import pytest

from app.models.image_refs import public_query_answer
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


PRESCRIPTION_URL = "/media/images/azza_prescription.jpeg"
PRESCRIPTION_TEXT = (
    "Prescription image labeled Azza "
    "(from filename, not verified identity). "
    "Document type aliases: prescription, medical prescription, روشتة."
)


def _image_metadata(url: str = PRESCRIPTION_URL) -> dict:
    return {
        "file_name": "azza_prescription.jpeg",
        "file_type": "image",
        "media_type": "image/jpeg",
        "image_url": "images/azza_prescription.jpeg",
        "image_path": "images/azza_prescription.jpeg",
        "image_description": PRESCRIPTION_TEXT,
        "images": [
            {
                "url": "images/azza_prescription.jpeg",
                "source": "azza_prescription.jpeg",
                "source_id": "source-azza",
                "document_id": "source-azza",
                "media_type": "image/jpeg",
                "image_description": PRESCRIPTION_TEXT,
            }
        ],
    }


class ImageRetriever:
    def __init__(self, candidates: list[RetrievalCandidate]):
        self._candidates = candidates
        self.queries: list[str] = []

    async def retrieve(self, query):
        self.queries.append(query.text)
        return RetrievalResult(
            query=query,
            candidates=self._candidates,
            metrics=RetrievalMetrics(
                trace_id="trace-img",
                requested_top_k=5,
                returned_candidates=len(self._candidates),
            ),
        )

    def close(self):
        pass


class PassThroughReranker:
    async def rerank(self, retrieval_result, *, top_n=None):
        candidates = [
            RerankedCandidate(
                chunk_id=item.chunk_id,
                source_id=item.source_id,
                text=item.text,
                original_score=item.score,
                rerank_score=item.score,
                original_rank=index,
                rank=index,
                metadata=dict(item.metadata or {}),
            )
            for index, item in enumerate(retrieval_result.candidates)
        ]
        return RerankedResult(
            query=retrieval_result.query.text,
            candidates=candidates,
            trace_id="trace-img",
        )

    def close(self):
        pass


class RecordingContextBuilder:
    def __init__(self):
        self.seen_metadata = []

    def build(self, result):
        self.seen_metadata = [
            dict(candidate.metadata or {})
            for candidate in result.candidates
        ]
        return AssembledContext(
            query=result.query,
            evidence=[],
            budget=ContextBudgetStats(max_context_tokens=1000),
            metrics=ContextBuilderMetrics(
                input_candidate_count=len(result.candidates),
            ),
            trace_id=result.trace_id,
        )

    def close(self):
        pass


class RecordingPromptBuilder:
    def __init__(self):
        self.images = None
        self.query = None

    def build(self, query, context, *, request=None, images=None):
        self.query = query
        self.images = list(images or [])
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
            trace_id="trace-img",
        )


class DenialLLM:
    async def startup(self):
        pass

    async def shutdown(self):
        pass

    async def generate(self, request):
        class _Generation:
            text = "The prescription is not present in the retrieved data."

        return _Generation()


class OkLLM:
    async def startup(self):
        pass

    async def shutdown(self):
        pass

    async def generate(self, request):
        class _Generation:
            text = "The evidence discusses metformin dosing."

        return _Generation()


def _image_candidate(*, chunk_id: str = "chunk-1") -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        source_id="source-azza",
        text=PRESCRIPTION_TEXT,
        score=0.91,
        metadata=_image_metadata(),
    )


async def _run_pipeline(*, retriever, llm=None, prompt_builder=None, context_builder=None):
    prompt_builder = prompt_builder or RecordingPromptBuilder()
    context_builder = context_builder or RecordingContextBuilder()
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=PassThroughReranker(),
        context_builder=context_builder,
        prompt_builder=prompt_builder,
        llm=llm or OkLLM(),
    )
    await pipeline.startup()
    return await pipeline.run("عايزة روشتة عزة"), prompt_builder, context_builder


@pytest.mark.asyncio
async def test_natural_language_prescription_query_returns_image_url():
    """Wiring: evidence already retrieved without an image keyword still yields a URL."""
    retriever = ImageRetriever([_image_candidate()])
    result, prompt_builder, _ = await _run_pipeline(retriever=retriever)

    assert "صورة" not in retriever.queries[0]
    assert "image" not in retriever.queries[0].casefold()
    assert result.images
    assert result.images[0].url.endswith(
        "/media/images/azza_prescription.jpeg"
    )
    assert prompt_builder.images
    assert prompt_builder.images[0].url.endswith(
        "/media/images/azza_prescription.jpeg"
    )


@pytest.mark.asyncio
async def test_person_name_query_returns_image_without_filename():
    retriever = ImageRetriever([_image_candidate()])
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=PassThroughReranker(),
        context_builder=RecordingContextBuilder(),
        prompt_builder=RecordingPromptBuilder(),
        llm=OkLLM(),
    )
    await pipeline.startup()
    result = await pipeline.run(
        "What prescription is associated with Azza?"
    )

    assert "azza_prescription.jpeg" not in result.query
    assert result.images
    assert "azza_prescription.jpeg" in result.images[0].url


@pytest.mark.asyncio
async def test_image_metadata_survives_context_and_response():
    context_builder = RecordingContextBuilder()
    result, _, context_builder = await _run_pipeline(
        retriever=ImageRetriever([_image_candidate()]),
        context_builder=context_builder,
    )

    assert context_builder.seen_metadata
    assert context_builder.seen_metadata[0]["images"]
    assert result.images
    assert result.reranking.candidates[0].metadata["images"]


@pytest.mark.asyncio
async def test_duplicate_chunks_yield_one_image():
    retriever = ImageRetriever(
        [
            _image_candidate(chunk_id="chunk-1"),
            _image_candidate(chunk_id="chunk-2"),
        ]
    )
    result, _, _ = await _run_pipeline(retriever=retriever)
    assert len(result.images) == 1


@pytest.mark.asyncio
async def test_malformed_metadata_does_not_crash_pipeline():
    retriever = ImageRetriever(
        [
            RetrievalCandidate(
                chunk_id="chunk-bad",
                source_id="source-bad",
                text="unrelated clinical note",
                score=0.2,
                metadata={
                    "images": [None, {"url": None}],
                    "image_url": "",
                },
            )
        ]
    )
    result, _, _ = await _run_pipeline(retriever=retriever)
    assert result.images == ()
    assert result.generation.text == (
        "The evidence discusses metformin dosing."
    )


@pytest.mark.asyncio
async def test_unrelated_query_does_not_return_irrelevant_image():
    retriever = ImageRetriever(
        [
            RetrievalCandidate(
                chunk_id="chunk-text",
                source_id="source-text",
                text="Metformin is used for type 2 diabetes.",
                score=0.88,
                metadata={"file_name": "diabetes.txt", "file_type": "text"},
            )
        ]
    )
    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=PassThroughReranker(),
        context_builder=RecordingContextBuilder(),
        prompt_builder=RecordingPromptBuilder(),
        llm=OkLLM(),
    )
    await pipeline.startup()
    result = await pipeline.run("What is the metformin dose?")
    assert result.images == ()


@pytest.mark.asyncio
async def test_answer_does_not_claim_missing_prescription_when_image_exists():
    result, _, _ = await _run_pipeline(
        retriever=ImageRetriever([_image_candidate()]),
        llm=DenialLLM(),
    )
    answer = public_query_answer(result.generation, result.images)
    assert result.images
    assert "not present in the retrieved data" not in answer.casefold()
    assert "images field" in answer
    assert "not been extracted" in answer.casefold()
    assert "azza" not in answer.casefold()


@pytest.mark.asyncio
async def test_non_image_response_fields_remain_compatible():
    retriever = ImageRetriever(
        [
            RetrievalCandidate(
                chunk_id="chunk-1",
                source_id="source-1",
                text="Medical evidence text.",
                score=0.95,
                metadata={"file_name": "test.txt"},
            )
        ]
    )
    result, _, _ = await _run_pipeline(retriever=retriever, llm=OkLLM())
    assert result.query == "عايزة روشتة عزة"
    assert result.generation.text == (
        "The evidence discusses metformin dosing."
    )
    assert result.trace_id
    assert result.images == ()
    assert result.is_degraded in {True, False}
