from app.models.image_refs import (
    collect_retrieved_images,
    convert_retrieved_images,
    expand_query_with_document_aliases,
    explain_public_image_url,
    public_query_answer,
    reconcile_answer_with_image_evidence,
    summarize_candidate_image_fields,
    to_public_image_urls,
)
from app.models.rag_models import RetrievedImage, RetrievalCandidate


def _image_candidate(
    *,
    chunk_id: str,
    url: str,
    source_id: str = "source-azza",
    extra_metadata: dict | None = None,
) -> RetrievalCandidate:
    metadata = {
        "file_name": "azza_prescription.jpeg",
        "file_type": "image",
        "image_url": url,
        "image_path": url,
        "image_description": (
            "Prescription image labeled Azza "
            "(from filename, not verified identity)."
        ),
        "images": [
            {
                "url": url,
                "source": "azza_prescription.jpeg",
                "source_id": source_id,
                "document_id": source_id,
                "media_type": "image/jpeg",
            }
        ],
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    return RetrievalCandidate(
        chunk_id=chunk_id,
        source_id=source_id,
        text=metadata["image_description"],
        score=0.9,
        metadata=metadata,
    )


def test_expand_query_adds_type_aliases_without_image_keyword():
    expanded = expand_query_with_document_aliases("عايزة روشتة عزة")

    assert "عايزة روشتة عزة" in expanded
    assert "prescription" in expanded.casefold()
    assert "image" not in expanded.casefold().split()


def test_expand_query_folds_egyptian_spelling():
    expanded = expand_query_with_document_aliases("عايزه روشته منار")

    assert "prescription" in expanded.casefold()
    assert "روشتة" in expanded or "روشته" in expanded


def test_query_expansion_overlaps_existing_indexed_type_aliases():
    # Shape of image_description written by current ingest for a
    # *prescription* filename, without requiring a later refresh.
    indexed = (
        "Prescription image labeled Azza "
        "(from filename, not verified identity). "
        "Document type aliases: prescription, medical prescription, روشتة."
    )
    expanded = expand_query_with_document_aliases("عايزة روشتة عزة")

    assert "prescription" in expanded.casefold()
    assert "prescription" in indexed.casefold()
    assert "روشتة" in expanded or "روشتة" in indexed


def test_name_variants_are_not_equalized_by_query_expansion():
    arabic = expand_query_with_document_aliases("هات روشته منار عبدالباقي")
    latin = expand_query_with_document_aliases("Show me Manar Abdelbaky prescription")
    short = expand_query_with_document_aliases("روشتة منار")
    doctor = expand_query_with_document_aliases("عايزه روشتة الدكتور الخطيب")

    for expanded in (arabic, latin, short):
        assert "prescription" in expanded.casefold()

    assert "منار" in arabic
    assert "Manar" in latin
    assert arabic != latin
    assert "الخطيب" in doctor
    assert "Azza" not in doctor
    assert "Manar" not in doctor


def test_expand_query_does_not_alter_unrelated_text():
    query = "What is the recommended metformin dose?"
    assert expand_query_with_document_aliases(query) == query


def test_collect_images_without_filename_in_query():
    images = collect_retrieved_images(
        [
            _image_candidate(
                chunk_id="chunk-1",
                url="/media/images/azza_prescription.jpeg",
            )
        ]
    )

    assert len(images) == 1
    assert images[0].url == "/media/images/azza_prescription.jpeg"
    assert images[0].source == "azza_prescription.jpeg"


def test_distinct_image_urls_are_not_merged():
    images = collect_retrieved_images(
        [
            _image_candidate(
                chunk_id="chunk-1",
                url="/media/images/azza_prescription.jpeg",
                source_id="source-azza",
            ),
            _image_candidate(
                chunk_id="chunk-2",
                url="/media/images/manar_prescription.jpeg",
                source_id="source-manar",
            ),
        ]
    )

    assert [image.url for image in images] == [
        "/media/images/azza_prescription.jpeg",
        "/media/images/manar_prescription.jpeg",
    ]


def test_duplicate_image_urls_are_deduplicated():
    url = "/media/images/maram_abdelbaky_prescription.jpeg"
    images = collect_retrieved_images(
        [
            _image_candidate(
                chunk_id="chunk-1",
                url=url,
                source_id="source-maram",
            ),
            _image_candidate(
                chunk_id="chunk-2",
                url=url,
                source_id="source-maram",
            ),
        ]
    )

    assert len(images) == 1
    assert images[0].url == url


def test_malformed_image_metadata_does_not_raise():
    candidates = [
        RetrievalCandidate(
            chunk_id="chunk-bad-1",
            source_id="source-1",
            text="text only",
            score=0.1,
            metadata={
                "images": [None, {}, {"url": ""}, {"url": "   "}],
                "image_url": None,
                "image_path": "",
            },
        ),
        type(
            "BrokenRecord",
            (),
            {
                "source_id": "source-2",
                "chunk_id": "chunk-bad-2",
                "metadata": "not-a-mapping",
            },
        )(),
    ]

    images = collect_retrieved_images(candidates)
    assert images == []


def test_candidate_summaries_omit_chunk_text():
    summaries = summarize_candidate_image_fields(
        [
            _image_candidate(
                chunk_id="chunk-1",
                url="images/manar_abdelbaky_prescription.jpeg",
            )
        ]
    )
    assert summaries[0]["chunk_id"] == "chunk-1"
    assert summaries[0]["file_name"] == "azza_prescription.jpeg"
    assert "images" in summaries[0]["image_metadata_keys"]
    assert summaries[0]["has_image_reference"] is True
    assert "text" not in summaries[0]
    assert "Prescription image" not in str(summaries)


def test_explain_public_image_url_reasons():
    assert explain_public_image_url(
        "images/manar_abdelbaky_prescription.jpeg",
        "data",
    )[1] == "converted"
    assert explain_public_image_url("../../secret.jpeg", "data")[1] == (
        "parent_traversal"
    )
    assert explain_public_image_url("C:/Windows/foo.jpeg", "data")[1] == (
        "absolute_outside_data_dir"
    )


def test_convert_reports_rejected_paths_without_raw_filesystem_url():
    collected = collect_retrieved_images(
        [
            _image_candidate(
                chunk_id="chunk-1",
                url="../../secret.jpeg",
            )
        ]
    )
    public, rejected = convert_retrieved_images(collected, data_dir="data")
    assert public == []
    assert rejected[0]["reason"] == "parent_traversal"
    assert ".." not in rejected[0]["reason"] or rejected[0]["reason"] == (
        "parent_traversal"
    )


def test_unresolvable_paths_are_not_converted_to_media_urls():
    images = collect_retrieved_images(
        [
            _image_candidate(
                chunk_id="chunk-1",
                url="../../secret.jpeg",
            ),
            _image_candidate(
                chunk_id="chunk-2",
                url="C:/Windows/foo.jpeg",
                source_id="source-abs",
            ),
        ]
    )
    public = to_public_image_urls(images, data_dir="data")
    assert public == []


def test_public_urls_dedupe_path_aliases_of_the_same_file():
    images = collect_retrieved_images(
        [
            _image_candidate(
                chunk_id="chunk-1",
                url="images/azza_prescription.jpeg",
            ),
            _image_candidate(
                chunk_id="chunk-2",
                url="/media/images/azza_prescription.jpeg",
            ),
        ]
    )
    public = to_public_image_urls(images, data_dir="data")

    assert len(public) == 1
    assert public[0].url.endswith("/media/images/azza_prescription.jpeg")


def test_reconcile_denies_absence_when_image_exists():
    image = RetrievedImage(
        url="/media/images/azza_prescription.jpeg",
        source_id="source-azza",
        source="azza_prescription.jpeg",
    )
    rewritten = reconcile_answer_with_image_evidence(
        "The prescription is not present in the retrieved data.",
        [image],
    )

    assert rewritten is not None
    assert "images field" in rewritten
    assert "not present in the retrieved data" not in rewritten.casefold()
    assert "azza" not in rewritten.casefold()
    assert "ocr" in rewritten.casefold()
    assert "not been extracted" in rewritten.casefold()


def test_reconcile_arabic_absence_without_naming_a_person():
    image = RetrievedImage(
        url="/media/images/azza_prescription.jpeg",
        source_id="source-azza",
    )
    rewritten = reconcile_answer_with_image_evidence(
        "لا توجد روشتة في البيانات المسترجعة.",
        [image],
    )

    assert rewritten is not None
    assert "images field" in rewritten
    assert "عزة" not in rewritten


def test_reconcile_keeps_legitimate_content_uncertainty():
    image = RetrievedImage(
        url="/media/images/azza_prescription.jpeg",
        source_id="source-azza",
    )

    dose_answer = (
        "Retrieved evidence is insufficient to determine the metformin dose."
    )
    assert reconcile_answer_with_image_evidence(dose_answer, [image]) == dose_answer

    contents_answer = (
        "The prescription image is available, but its contents have not "
        "been extracted."
    )
    assert (
        reconcile_answer_with_image_evidence(contents_answer, [image])
        == contents_answer
    )


def test_public_query_answer_keeps_compatible_non_image_text():
    class _Generation:
        text = "Metformin is mentioned in the evidence."

    answer = public_query_answer(_Generation(), [])
    assert answer == "Metformin is mentioned in the evidence."
