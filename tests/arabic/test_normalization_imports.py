"""Regression: startup import chain must load OntologyCode from evidence_models."""

from app.arabic import normalization
from app.models.evidence_models import OntologyCode, OntologySystem
from app.models.image_refs import expand_query_with_document_aliases
from app.api.media import build_media_router


def test_ontology_types_exist_on_evidence_models():
    code = OntologyCode(
        system=OntologySystem("ICD10"),
        code="E11",
        display="Type 2 diabetes mellitus",
    )
    assert code.system.value == "ICD10"
    assert code.code == "E11"


def test_normalization_module_imports_ontology_types():
    assert normalization.OntologyCode is OntologyCode
    assert normalization.OntologySystem is OntologySystem
    assert normalization.normalize_arabic_orthography("الإيكو") == (
        normalization.normalize_arabic_orthography("الايكو")
    )


def test_image_refs_and_media_router_import_without_error():
    assert expand_query_with_document_aliases("عايزه الايكو بتاع منار")
    router = build_media_router("data")
    assert router is not None
