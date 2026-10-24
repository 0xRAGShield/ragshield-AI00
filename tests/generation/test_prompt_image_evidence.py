from app.core.settings import PromptSettings
from app.generation.prompts.prompt_builder import PromptBuilder
from app.models.rag_models import (
    AssembledContext,
    ContextBudgetStats,
    ContextBuilderMetrics,
    EvidenceBlock,
    EvidenceIdentity,
    EvidenceProvenance,
    RetrievedImage,
)


def test_prompt_includes_image_reference_and_forbids_absence_claim():
    builder = PromptBuilder(
        settings=PromptSettings(
            max_prompt_tokens=8000,
            output_reserve_tokens=256,
            safety_mode="strict",
            require_exact_tokenizer=False,
        )
    )
    query = "عايزة روشتة عزة"
    image = RetrievedImage(
        url="/media/images/azza_prescription.jpeg",
        source_id="source-azza",
        source="azza_prescription.jpeg",
        document_id="source-azza",
        media_type="image/jpeg",
        description=(
            "Prescription image labeled Azza "
            "(from filename, not verified identity)."
        ),
        relative_url="/media/images/azza_prescription.jpeg",
        chunk_id="chunk-1",
    )
    context = AssembledContext(
        query=query,
        evidence=[
            EvidenceBlock(
                identity=EvidenceIdentity(
                    chunk_id="chunk-1",
                    source_id="source-azza",
                ),
                provenance=EvidenceProvenance(
                    source="azza_prescription.jpeg",
                    source_id="source-azza",
                    metadata={
                        "file_name": "azza_prescription.jpeg",
                        "file_type": "image",
                        "image_description": image.description,
                    },
                ),
                rank=1,
                rerank_score=0.8,
                content=image.description or "prescription image",
                token_count=12,
            )
        ],
        budget=ContextBudgetStats(max_context_tokens=1000),
        metrics=ContextBuilderMetrics(input_candidate_count=1),
        trace_id="trace-image",
    )

    prompt = builder.build(query, context, images=[image])
    system = prompt.messages[0].content
    user = prompt.messages[1].content

    assert "BEGIN_RETRIEVED_IMAGE_REFERENCES" in user
    assert "/media/images/azza_prescription.jpeg" in user
    assert "pixel contents were not extracted" in user.casefold()
    assert "do not claim that the image" in system.casefold()
    assert "contents have not been extracted" in system.casefold()
    assert "do not generate or guess medical image references" not in system.casefold()
