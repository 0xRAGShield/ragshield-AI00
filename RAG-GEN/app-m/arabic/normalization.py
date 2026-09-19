"""Arabic/Egyptian normalization with medical entity preservation and ontology codes."""

from __future__ import annotations

import re
import unicodedata
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.arabic.egyptian import dialect_entry_ids
from app.models.evidence_models import OntologyCode, OntologySystem

logger = logging.getLogger(__name__)

_TATWEEL = "\u0640"
_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_ALEF = re.compile("[إأآا]")
_YEH = re.compile("[ىي]")
_HEH = re.compile("[ةه]")
# FIX 1: Modified to preserve numeric-leading tokens (e.g., "5-FU", "10mg")
_LATIN_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-/%]*")
_ICD_SHAPE = re.compile(r"\b[A-TV-Z][0-9]{2}(?:\.[0-9]{1,4})?\b", re.I)


class NormalizedQuery(BaseModel):
    original: str
    normalized: str
    ontology_codes: list[OntologyCode] = Field(default_factory=list)
    preserved_entities: list[str] = Field(default_factory=list)
    catalog_entry_ids: list[str] = Field(default_factory=list)


class OntologyEntry(BaseModel):
    id: str
    surfaces: list[str]
    codes: list[OntologyCode]


class OntologyCatalog:
    """Versioned fixture catalog. Not a full ICD-10/RxNorm distribution."""

    def __init__(self, entries: list[OntologyEntry], version: str) -> None:
        self.version = version
        self.entries = entries
        self._by_id = {entry.id: entry for entry in entries}
        
        # FIX 2: Pre-sort surfaces by length descending at initialization
        # Saves O(n log n) sorting cost on every query
        self._sorted_surfaces: list[tuple[str, OntologyEntry]] = []
        for entry in entries:
            for surface in entry.surfaces:
                self._sorted_surfaces.append((surface, entry))
        self._sorted_surfaces.sort(key=lambda x: len(x[0]), reverse=True)

    @classmethod
    def from_yaml(cls, path: Path) -> OntologyCatalog:
        payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        version = str(payload.get("version", "unknown"))
        entries: list[OntologyEntry] = []
        
        for idx, raw in enumerate(payload.get("entries", [])):
            # FIX 3: Contextual error handling for malformed YAML
            try:
                codes = [
                    OntologyCode(
                        system=OntologySystem(code["system"]),
                        code=str(code["code"]),
                        display=str(code["display"]),
                    )
                    for code in raw.get("codes", [])
                ]
                entries.append(
                    OntologyEntry(
                        id=str(raw["id"]),
                        surfaces=[str(s) for s in raw.get("surfaces", [])],
                        codes=codes,
                    )
                )
            except Exception as e:
                logger.error(f"Failed to load catalog entry in {path} at index {idx}. Data: {raw}")
                raise ValueError(f"Malformed catalog entry in {path} at index {idx}: {str(e)}") from e
                
        return cls(entries, version)

    def get(self, entry_id: str) -> OntologyEntry | None:
        return self._by_id.get(entry_id)

    def match_text(self, text: str) -> list[OntologyEntry]:
        haystack = text.casefold()
        hits: list[OntologyEntry] = []
        seen: set[str] = set()
        matched_intervals: list[tuple[int, int]] = []
        
        # FIX 4: Real longest-match suppression + Word Boundaries
        for surface, entry in self._sorted_surfaces:
            surface_folded = surface.casefold()
            
            # Using \w allows Unicode word boundaries (works for Arabic & English)
            pattern = re.compile(r'(?<!\w)' + re.escape(surface_folded) + r'(?!\w)', re.UNICODE)
            
            for match in pattern.finditer(haystack):
                start, end = match.span()
                
                # If overlap exists with a longer match, skip this shorter substring
                overlap = any(s < end and e > start for s, e in matched_intervals)
                
                if not overlap:
                    matched_intervals.append((start, end))
                    if entry.id not in seen:
                        hits.append(entry)
                        seen.add(entry.id)
                        
        return hits


def normalize_arabic_orthography(text: str) -> str:
    """NFKC + common Arabic letter folding. Latin medical tokens are copied unchanged."""
    nfkc = unicodedata.normalize("NFKC", text).replace(_TATWEEL, "")
    pieces: list[str] = []
    cursor = 0
    for match in _LATIN_TOKEN.finditer(nfkc):
        arabic_span = nfkc[cursor : match.start()]
        pieces.append(_fold_arabic_span(arabic_span))
        pieces.append(match.group(0))
        cursor = match.end()
    pieces.append(_fold_arabic_span(nfkc[cursor:]))
    return "".join(pieces)


def _fold_arabic_span(span: str) -> str:
    out = _DIACRITICS.sub("", span)
    out = _ALEF.sub("ا", out)
    out = _YEH.sub("ي", out)
    out = _HEH.sub("ه", out)
    return out


class ArabicNormalizer:
    def __init__(self, catalog: OntologyCatalog) -> None:
        self._catalog = catalog

    def normalize(self, text: str) -> NormalizedQuery:
        preserved = _LATIN_TOKEN.findall(text) + _ICD_SHAPE.findall(text)
        folded = normalize_arabic_orthography(text)
        
        catalog_hits = self._catalog.match_text(text) + self._catalog.match_text(folded)
        
        # FIX 5: Graceful degradation if dialect mapping fails
        try:
            dialect_ids = dialect_entry_ids(text)
        except Exception as e:
            logger.warning(f"dialect_entry_ids failed on input '{text[:30]}...'. Error: {e}")
            dialect_ids = []

        for entry_id in dialect_ids:
            entry = self._catalog.get(entry_id)
            if entry:
                catalog_hits.append(entry)

        unique_entries: list[OntologyEntry] = []
        seen: set[str] = set()
        for entry in catalog_hits:
            if entry.id not in seen:
                unique_entries.append(entry)
                seen.add(entry.id)

        codes: list[OntologyCode] = []
        code_keys: set[tuple[str, str]] = set()
        for entry in unique_entries:
            for code in entry.codes:
                key = (code.system.value, code.code)
                if key not in code_keys:
                    codes.append(code)
                    code_keys.add(key)

        return NormalizedQuery(
            original=text,
            normalized=folded.strip(),
            ontology_codes=codes,
            preserved_entities=list(dict.fromkeys(preserved)),
            catalog_entry_ids=[entry.id for entry in unique_entries],
        )


@lru_cache(maxsize=4)
def load_catalog(path: str) -> OntologyCatalog:
    return OntologyCatalog.from_yaml(Path(path))
