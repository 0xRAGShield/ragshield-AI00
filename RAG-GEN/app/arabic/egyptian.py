"""Egyptian Arabic dialect surfaces mapped onto shared ontology keys."""

from __future__ import annotations

import re

# FIX 2: Local orthographic folding regexes to handle Arabic variants seamlessly
# This avoids a circular import with normalization.py while ensuring robust matching.
_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_ALEF = re.compile("[إأآا]")
_YEH = re.compile("[ىي]")
_HEH = re.compile("[ةه]")

def _fold_arabic(text: str) -> str:
    """Applies basic Arabic character unification and case-folding."""
    out = _DIACRITICS.sub("", text)
    out = _ALEF.sub("ا", out)
    out = _YEH.sub("ي", out)
    out = _HEH.sub("ه", out)
    # casefold() is harmless for Arabic but future-proofs Latin brand names
    return out.casefold()

# Keys must exist in data/fixtures/ontology_catalog.yaml
# Notice: Enumeration of Alef variants (like الأنسولين vs انسولين) is no longer needed.
EGYPTIAN_SURFACE_TO_ENTRY: dict[str, str] = {
    "سكر": "diabetes_t2",
    "السكر": "diabetes_t2",
    "السكري": "diabetes_t2",
    "مرض السكر": "diabetes_t2",
    "ضغط": "hypertension",
    "الضغط": "hypertension",
    "ضغط الدم": "hypertension",
    "جلوكوفاج": "metformin",
    "انسولين": "insulin",
}

# FIX 3: Precompute at module load time to save O(n log n) sorting cost per query
_SORTED_SURFACES = []
for surface, entry_id in EGYPTIAN_SURFACE_TO_ENTRY.items():
    _SORTED_SURFACES.append((_fold_arabic(surface), entry_id))

# Sort longest-first
_SORTED_SURFACES.sort(key=lambda x: len(x[0]), reverse=True)

# FIX 1 & 4: Precompile Regex with Word Boundaries to prevent morphology collisions
_COMPILED_PATTERNS = []
for folded_surface, entry_id in _SORTED_SURFACES:
    # (?<!\w) and (?!\w) ensure full-word matching ("سكر" will NOT match "عسكري")
    pattern = re.compile(r'(?<!\w)' + re.escape(folded_surface) + r'(?!\w)', re.UNICODE)
    _COMPILED_PATTERNS.append((pattern, entry_id))


def dialect_entry_ids(text: str) -> list[str]:
    """Return catalog entry ids whose Egyptian surfaces appear as substrings."""
    # Apply folding to the incoming text so it matches the precomputed folded keys
    folded_text = _fold_arabic(text.strip())
    hits: list[str] = []
    seen: set[str] = set()
    matched_intervals: list[tuple[int, int]] = []

    for pattern, entry_id in _COMPILED_PATTERNS:
        for match in pattern.finditer(folded_text):
            start, end = match.span()
            
            # Real longest-match suppression: skip if overlapping with a longer matched surface
            overlap = any(s < end and e > start for s, e in matched_intervals)
            
            if not overlap:
                matched_intervals.append((start, end))
                if entry_id not in seen:
                    hits.append(entry_id)
                    seen.add(entry_id)
                    
    return hits
