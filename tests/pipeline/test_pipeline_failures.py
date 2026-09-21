import pytest

from app.pipeline import (
    PipelineError,
    RAGPipeline,
)


class FailingRetriever:

    async def retrieve(self, query):
        raise RuntimeError(
            "Qdrant unavailable"
        )

    def close(self):
        pass


class Dummy:
    async def startup(self):
        pass

    async def shutdown(self):
        pass


@pytest.mark.asyncio
async def test_retrieval_failure_is_wrapped():

    pipeline = RAGPipeline(
        retriever=FailingRetriever(),
        reranker=Dummy(),
        context_builder=Dummy(),
        prompt_builder=Dummy(),
        llm=Dummy(),
    )

    await pipeline.startup()

    with pytest.raises(
        PipelineError
    ) as error:

        await pipeline.run(
            "test query"
        )

    assert error.value.stage == (
        "retrieval"
    )

    assert error.value.trace_id