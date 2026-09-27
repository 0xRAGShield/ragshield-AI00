from __future__ import annotations

import time
import logging
import uuid

from dataclasses import dataclass
from typing import Any

from app.generation.llm.llm import (
    ChatMessage as LLMChatMessage,
    GenerationResult,
    LLMRequest,
    LLMRuntime,
)

from app.generation.prompts.prompt_builder import (
    PromptBuilder,
    PromptRequest,
)

from app.models.rag_models import (
    AssembledContext,
    LLMPrompt,
    RetrievalQuery,
    RetrievalResult,
    RerankedResult,
)

from app.retrieval.reranker import Reranker
from app.retrieval.retriever import Retriever
from app.context.context_builder import ContextBuilder


logger = logging.getLogger(
    "ragshield.pipeline"
)


class PipelineError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage: str,
        trace_id: str,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.trace_id = trace_id
        self.cause = cause

        if cause is not None:
            self.__cause__ = cause


@dataclass(frozen=True)
class RAGPipelineResult:
    trace_id: str
    query: str
    retrieval: RetrievalResult
    reranking: RerankedResult
    context: AssembledContext
    prompt: LLMPrompt
    generation: GenerationResult
    is_degraded: bool


class RAGPipeline:
    def __init__(
        self,
        *,
        retriever: Retriever,
        reranker: Reranker,
        context_builder: ContextBuilder,
        prompt_builder: PromptBuilder,
        llm: LLMRuntime,
    ) -> None:

        if retriever is None:
            raise ValueError(
                "retriever cannot be None."
            )

        if reranker is None:
            raise ValueError(
                "reranker cannot be None."
            )

        if context_builder is None:
            raise ValueError(
                "context_builder cannot be None."
            )

        if prompt_builder is None:
            raise ValueError(
                "prompt_builder cannot be None."
            )

        if llm is None:
            raise ValueError(
                "llm cannot be None."
            )

        self._retriever = retriever
        self._reranker = reranker
        self._context_builder = context_builder
        self._prompt_builder = prompt_builder
        self._llm = llm

        self._started = False
        self._closed = False

    @property
    def llm(self) -> LLMRuntime:
        return self._llm

    @property
    def is_started(self) -> bool:
        return self._started and not self._closed

    async def startup(self) -> None:
        if self._closed:
            raise RuntimeError(
                "Pipeline has already been closed."
            )

        if self._started:
            return

        await self._llm.startup()

        self._started = True

    async def shutdown(self) -> None:
        if self._closed:
            return

        try:
            await self._llm.shutdown()

        finally:
            self._close_component(
                self._prompt_builder
            )

            self._close_component(
                self._context_builder
            )

            self._close_component(
                self._reranker
            )

            self._close_component(
                self._retriever
            )

            self._closed = True
            self._started = False

    async def run(
        self,
        query: str,
        *,
        top_k: int | None = None,
        top_n: int | None = None,
        prompt_request: PromptRequest | None = None,
        max_new_tokens: int | None = None,
    ) -> RAGPipelineResult:

        if self._closed:
            raise RuntimeError(
                "Pipeline has been closed."
            )

        if not self._started:
            raise RuntimeError(
                "Pipeline.startup() must be called "
                "before run()."
            )

        if not isinstance(query, str):
            raise ValueError(
                "query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "query cannot be empty."
            )

        trace_id = uuid.uuid4().hex

        # ============================================================
        # RETRIEVAL
        # ============================================================

        try:
            retrieval_started = time.perf_counter()

            retrieval_query = RetrievalQuery(
                text=query,
                top_k=top_k,
            )

            retrieval = await self._retriever.retrieve(
                retrieval_query
            )

            retrieval_latency_ms = (
                time.perf_counter() - retrieval_started
            ) * 1000.0

        except Exception as exc:
            raise PipelineError(
                "Retrieval stage failed.",
                stage="retrieval",
                trace_id=trace_id,
                cause=exc,
            ) from exc

        # ============================================================
        # RERANKING
        # ============================================================

        try:
            print(
                ">>> BEFORE RERANK",
                flush=True,
            )

            reranking_started = time.perf_counter()

            reranking = await self._reranker.rerank(
                retrieval,
                top_n=top_n,
            )

            print(
                ">>> AFTER RERANK",
                flush=True,
            )

            reranking_latency_ms = (
                time.perf_counter() - reranking_started
            ) * 1000.0

            print(
                f">>> RERANK TIME: "
                f"{reranking_latency_ms:.2f} ms",
                flush=True,
            )

        except Exception as exc:
            raise PipelineError(
                "Reranking stage failed.",
                stage="reranking",
                trace_id=trace_id,
                cause=exc,
            ) from exc

        # ============================================================
        # CONTEXT
        # ============================================================

        try:
            print(
                ">>> BEFORE CONTEXT",
                flush=True,
            )

            context_started = time.perf_counter()

            context = self._context_builder.build(
                reranking
            )

            context_latency_ms = (
                time.perf_counter() - context_started
            ) * 1000.0

            print(
                ">>> AFTER CONTEXT",
                flush=True,
            )

            print(
                f">>> CONTEXT TIME: "
                f"{context_latency_ms:.2f} ms",
                flush=True,
            )

        except Exception as exc:
            raise PipelineError(
                "Context stage failed.",
                stage="context",
                trace_id=trace_id,
                cause=exc,
            ) from exc

        # ============================================================
        # PROMPT
        # ============================================================

        try:
            print(
                ">>> BEFORE PROMPT",
                flush=True,
            )

            prompt_started = time.perf_counter()

            prompt = self._prompt_builder.build(
                query,
                context,
                request=prompt_request,
            )

            prompt_latency_ms = (
                time.perf_counter() - prompt_started
            ) * 1000.0

            print(
                ">>> AFTER PROMPT",
                flush=True,
            )

            print(
                f">>> PROMPT TIME: "
                f"{prompt_latency_ms:.2f} ms",
                flush=True,
            )

        except Exception as exc:
            logger.exception(
                "Prompt stage failed. trace_id=%s",
                trace_id,
            )

            raise PipelineError(
                f"Prompt stage failed: "
                f"{type(exc).__name__}: {exc}",
                stage="prompt",
                trace_id=trace_id,
                cause=exc,
            ) from exc

        # ============================================================
        # LLM REQUEST CONSTRUCTION
        # ============================================================

        try:
            print(
                ">>> BEFORE LLM REQUEST",
                flush=True,
            )

            llm_messages = [
                LLMChatMessage(
                    role=message.role,
                    content=message.content,
                )
                for message in prompt.messages
            ]

            llm_request = LLMRequest(
                messages=llm_messages,
                max_new_tokens=max_new_tokens,
                trace_id=trace_id,
            )

            print(
                ">>> AFTER LLM REQUEST",
                flush=True,
            )

        except Exception as exc:
            raise PipelineError(
                "LLM request construction failed.",
                stage="llm_request",
                trace_id=trace_id,
                cause=exc,
            ) from exc

        # ============================================================
        # LLM GENERATION
        # ============================================================

        try:
            print(
                ">>> BEFORE LLM",
                flush=True,
            )

            llm_started = time.perf_counter()

            generation = await self._llm.generate(
                llm_request
            )

            llm_latency_ms = (
                time.perf_counter() - llm_started
            ) * 1000.0

            print(
                ">>> AFTER LLM",
                flush=True,
            )

            print(
                f">>> LLM TIME: "
                f"{llm_latency_ms:.2f} ms",
                flush=True,
            )

            print(
                "\n=== RAG LATENCY ==="
                f"\nRetrieval: "
                f"{retrieval_latency_ms:.2f} ms"
                f"\nReranking: "
                f"{reranking_latency_ms:.2f} ms"
                f"\nContext: "
                f"{context_latency_ms:.2f} ms"
                f"\nPrompt: "
                f"{prompt_latency_ms:.2f} ms"
                f"\nLLM: "
                f"{llm_latency_ms:.2f} ms",
                flush=True,
            )

        except Exception as exc:
            raise PipelineError(
                "LLM generation failed.",
                stage="llm",
                trace_id=trace_id,
                cause=exc,
            ) from exc

        # ============================================================
        # DEGRADED STATUS
        # ============================================================

        is_degraded = bool(
            retrieval.is_degraded
            or retrieval.metrics.is_degraded
            or reranking.is_degraded
            or reranking.is_fallback
            or context.is_degraded
            or prompt.is_degraded
        )

        # ============================================================
        # RESULT
        # ============================================================

        return RAGPipelineResult(
            trace_id=trace_id,
            query=query,
            retrieval=retrieval,
            reranking=reranking,
            context=context,
            prompt=prompt,
            generation=generation,
            is_degraded=is_degraded,
        )

    @staticmethod
    def _close_component(
        component: Any,
    ) -> None:

        close = getattr(
            component,
            "close",
            None,
        )

        if callable(close):
            try:
                close()

            except Exception:
                logger.exception(
                    "Failed to close pipeline component."
                )