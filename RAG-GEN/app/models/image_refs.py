from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

from app.arabic.normalization import normalize_arabic_orthography
from app.models.evidence_models import ImageDocument, SourceDocument
from app.models.rag_models import RetrievedImage

logger = logging.getLogger("ragshield.images")


MEDIA_ROUTE_PREFIX = "/media"

IMAGE_FILE_EXTENSIONS = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
    }
)

_IMAGE_SCALAR_KEYS = (
    "image_path",
    "image_url",
    "image_uri",
    "image_reference",
)

_IMAGE_LIST_KEYS = (
    "images",
    "image_urls",
    "image_paths",
    "image_uris",
    "attachments",
)

_IMAGE_EXTENSIONS = IMAGE_FILE_EXTENSIONS

_TOKEN_SPLIT = re.compile(r"[_\-\s.]+")

_GENERIC_FILENAME_TOKENS = frozenset(
    {
        "image",
        "img",
        "photo",
        "picture",
        "pic",
        "file",
        "scan",
        "document",
        "doc",
        "images",
        "data",
    }
)

_TYPE_HINTS: dict[str, tuple[str, ...]] = {
    "prescription": (
        "prescription",
        "medical prescription",
        "روشتة",
        "روشته",
    ),
    "roshte": ("prescription", "روشتة"),
    "rosheta": ("prescription", "روشتة"),
    "roshita": ("prescription", "روشتة"),
    "echo":("echo", "ايكو"),
    "echo":("echo", "اكو"),
    "rx": ("prescription", "روشتة"),
    "xray": ("x-ray", "X-ray", "أشعة"),
    "x-ray": ("x-ray", "أشعة"),
    "radiograph": ("x-ray", "أشعة"),
    "mri": ("MRI", "رنين مغناطيسي"),
    "ct": ("CT scan", "أشعة مقطعية"),
    "ultrasound": ("ultrasound", "سونار"),
    "sonogram": ("ultrasound", "سونار"),
    "lab": ("laboratory result", "تحليل"),
    "labs": ("laboratory result", "تحليل"),
    "analysis": ("analysis", "تحليل"),
    "tahlil": ("laboratory result", "تحليل"),
    "blood": ("blood test", "تحليل دم"),
    "cbc": ("blood analysis", "تحليل دم"),
    "report": ("report", "تقرير"),
}

def _build_document_alias_groups() -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []

    for key, aliases in _TYPE_HINTS.items():
        members = (key.replace("-", " "), *aliases)
        unique = tuple(dict.fromkeys(members))
        groups.append(unique)

    return tuple(groups)


_DOCUMENT_ALIAS_GROUPS = _build_document_alias_groups()

_IMAGE_METADATA_KEYS = (
    *_IMAGE_SCALAR_KEYS,
    *_IMAGE_LIST_KEYS,
    "file_type",
    "media_type",
    "file_name",
    "relative_path",
    "image_description",
    "description",
)

_URL_KEYS = (
    "url",
    "uri",
    "path",
    "href",
    "reference",
    "image_path",
    "image_url",
    "image_uri",
    "image_reference",
    "file_path",
    "storage_path",
)


_DOCUMENT_ABSENCE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:image|prescription|document|scan|x-?ray|"
        r"روشته|صوره|اشعه)"
        r".{0,80}(?:not present|not found|is missing|"
        r"غير موجود|ليست موجود)",
        re.IGNORECASE | re.DOTALL | re.UNICODE,
    ),
    re.compile(
        r"(?:not present|not found|is missing|"
        r"غير موجود|ليست موجود).{0,80}"
        r"(?:image|prescription|document|scan|"
        r"روشته|صوره|اشعه)",
        re.IGNORECASE | re.DOTALL | re.UNICODE,
    ),
    re.compile(
        r"(?:no matching (?:image|prescription|document)|"
        r"no prescription image|"
        r"لا توجد?\s+روشته|مفيش\s+روشته|"
        r"لا توجد?\s+صوره)",
        re.IGNORECASE | re.UNICODE,
    ),
)

_IMAGE_ABSENCE_REWRITE = (
    "A retrieved document image is listed in the images field. "
    "That listing means the image file was retrieved as evidence; "
    "its pixel contents have not been extracted, so medication names "
    "or other OCR details are not available."
)


def reconcile_answer_with_image_evidence(
    answer: str | None,
    images: Sequence[RetrievedImage],
) -> str | None:
   
    if answer is None or not images:
        return answer

    folded = normalize_arabic_orthography(answer).casefold()

    if not any(pattern.search(folded) for pattern in _DOCUMENT_ABSENCE_PATTERNS):
        return answer

    return _IMAGE_ABSENCE_REWRITE


_GENERATION_FAILED_ANSWER = (
    "Answer generation failed, but retrieved sources are included. "
    "Any matching medical images are listed in the images field."
)


def public_query_answer(
    generation: Any,
    images: Sequence[RetrievedImage],
) -> str:
    if generation is None:
        return _GENERATION_FAILED_ANSWER

    text = getattr(generation, "text", None)

    if not isinstance(text, str) or not text.strip():
        return _GENERATION_FAILED_ANSWER

    reconciled = reconcile_answer_with_image_evidence(text, images)
    return reconciled if reconciled is not None else text


def expand_query_with_document_aliases(query: str) -> str:
   
    text = (query or "").strip()

    if not text:
        return text

    folded_query = normalize_arabic_orthography(text).casefold()
    extras: list[str] = []
    seen: set[str] = set()

    for group in _DOCUMENT_ALIAS_GROUPS:
        if not any(
            _alias_mentioned(alias, folded_query) for alias in group
        ):
            continue

        for alias in group:
            key = alias.casefold()

            if key in seen:
                continue

            seen.add(key)
            extras.append(alias)

    if not extras:
        return text

    return f"{text} {' '.join(extras)}"


def source_document_from_image(
    image: ImageDocument,
    data_dir: str | Path | None = None,
) -> SourceDocument:
   
    if not isinstance(image, ImageDocument):
        raise TypeError("image must be an ImageDocument.")

    metadata = dict(image.metadata or {})
    image_path = str(image.image_path).strip()

    if not image_path:
        raise ValueError(
            f"Image document is missing image_path: "
            f"{image.source_id}"
        )

    storage_ref = _storage_reference(
        image_path,
        data_dir,
    )

    existing_url = _first_non_empty_str(
        metadata.get("image_url"),
        metadata.get("image_uri"),
        metadata.get("image_reference"),
    )

    if existing_url and not _is_http_url(existing_url):
        existing_url = _storage_reference(
            existing_url,
            data_dir,
        )

    metadata.update(
        {
            "file_type": metadata.get("file_type") or "image",
            "file_name": metadata.get("file_name") or image.title,
            "media_type": image.media_type,
            "image_path": storage_ref,
            "source_title": image.title,
        }
    )

    if existing_url:
        metadata["image_url"] = existing_url
    else:
        metadata["image_url"] = storage_ref

    description = build_image_description(
        image,
        metadata=metadata,
        data_dir=data_dir,
    )
    metadata["image_description"] = description
    metadata.setdefault("description", description)

    metadata["images"] = _unique_image_dicts(
        [
            {
                "url": metadata["image_url"],
                "image_path": storage_ref,
                "media_type": image.media_type,
                "image_type": _optional_str(
                    metadata.get("image_type")
                ),
                "source_id": image.source_id,
                "source": image.title,
                "document_id": _optional_str(
                    metadata.get("document_id")
                )
                or image.source_id,
                "image_description": description,
            }
        ]
    )

    return SourceDocument(
        source_id=image.source_id,
        title=image.title,
        text=_searchable_image_text(
            image,
            data_dir=data_dir,
            description=description,
        ),
        media_type=image.media_type,
        metadata=metadata,
    )


def preserve_image_metadata(
    metadata: Mapping[str, Any] | None,
    *,
    source_id: str,
    title: str | None = None,
    media_type: str | None = None,
) -> dict[str, Any]:

    preserved = dict(metadata or {})
    images = collect_image_dicts(
        preserved,
        source_id=source_id,
        default_source=title,
        default_media_type=media_type,
    )

    if images:
        preserved["images"] = images

    return preserved


def collect_retrieved_images(
    records: Sequence[Mapping[str, Any] | Any],
) -> list[RetrievedImage]:

    collected: list[RetrievedImage] = []
    seen: set[str] = set()

    for record in records:
        source_id, chunk_id, metadata = _record_fields(record)

        if not source_id or not metadata:
            continue

        for image in collect_image_dicts(
            metadata,
            source_id=source_id,
            default_source=_optional_str(
                metadata.get("source")
                or metadata.get("source_title")
                or metadata.get("file_name")
            ),
            default_media_type=_optional_str(
                metadata.get("media_type")
            ),
        ):
            url = image["url"]
            dedupe_key = url.casefold()

            if dedupe_key in seen:
                continue

            seen.add(dedupe_key)
            collected.append(
                RetrievedImage(
                    url=url,
                    source_id=source_id,
                    media_type=image.get("media_type"),
                    image_type=image.get("image_type"),
                    document_id=(
                        image.get("document_id") or source_id
                    ),
                    source=image.get("source"),
                    chunk_id=chunk_id,
                    description=_optional_str(
                        metadata.get("image_description")
                        or metadata.get("description")
                        or image.get("image_description")
                    ),
                )
            )

    return collected


def summarize_candidate_image_fields(
    records: Sequence[Mapping[str, Any] | Any],
) -> list[dict[str, Any]]:
    """Safe per-candidate image diagnostics. No chunk text or OCR."""

    summaries: list[dict[str, Any]] = []

    for rank, record in enumerate(records, start=1):
        source_id, chunk_id, metadata = _record_fields(record)
        summaries.append(
            {
                "rank": rank,
                "score": _safe_score(record),
                "source_id": source_id,
                "chunk_id": chunk_id,
                "file_name": _optional_str(
                    metadata.get("file_name")
                    or metadata.get("source_title")
                ),
                "image_metadata_keys": [
                    key
                    for key in _IMAGE_METADATA_KEYS
                    if key in metadata and metadata.get(key) not in (None, "", [])
                ],
                "has_image_reference": bool(
                    collect_image_dicts(
                        metadata,
                        source_id=source_id or "unknown",
                    )
                )
                if metadata
                else False,
            }
        )

    return summaries


def convert_retrieved_images(
    images: Sequence[RetrievedImage],
    data_dir: str | Path | None,
    public_base_url: str | None = None,
) -> tuple[list[RetrievedImage], list[dict[str, str]]]:
    """Rewrite URLs and report safe rejection reasons (no raw FS paths)."""

    rewritten: list[RetrievedImage] = []
    rejected: list[dict[str, str]] = []
    seen_public: set[str] = set()

    for image in images:
        media_url, reason = explain_public_image_url(
            image.url,
            data_dir,
        )

        if not media_url:
            rejected.append(
                {
                    "source_id": image.source_id,
                    "chunk_id": image.chunk_id or "",
                    "source": image.source or "",
                    "reason": reason,
                }
            )
            continue

        if _is_http_url(media_url):
            absolute = media_url
            relative: str | None = image.relative_url
            clickable = media_url
        else:
            relative = media_url
            absolute = _join_public_base_url(
                public_base_url,
                media_url,
            )
            clickable = absolute or relative

        dedupe_key = clickable.casefold()

        if dedupe_key in seen_public:
            rejected.append(
                {
                    "source_id": image.source_id,
                    "chunk_id": image.chunk_id or "",
                    "source": image.source or "",
                    "reason": "duplicate_public_url",
                }
            )
            continue

        seen_public.add(dedupe_key)
        rewritten.append(
            image.model_copy(
                update={
                    "url": clickable,
                    "relative_url": relative,
                    "absolute_url": absolute,
                }
            )
        )

    return rewritten, rejected


def explain_public_image_url(
    reference: str,
    data_dir: str | Path | None,
) -> tuple[str, str]:
    text = (reference or "").strip()

    if not text:
        return "", "empty_reference"

    if _is_http_url(text):
        return text, "http_url"

    if text.startswith(f"{MEDIA_ROUTE_PREFIX}/") or text == MEDIA_ROUTE_PREFIX:
        return text.split("?", 1)[0], "media_route"

    replaced = text.replace("\\", "/")

    if ".." in Path(replaced).parts:
        return "", "parent_traversal"

    relative = _relative_to_data_dir(text, data_dir)

    if relative is None:
        if _looks_absolute(replaced):
            return "", "absolute_outside_data_dir"
        return "", "unresolvable_path"

    encoded = quote(relative.as_posix(), safe="/")
    return f"{MEDIA_ROUTE_PREFIX}/{encoded}", "converted"


def log_image_retrieval_diagnostics(
    *,
    query: str,
    trace_id: str,
    retrieved_summaries: Sequence[Mapping[str, Any]],
    reranked_summaries: Sequence[Mapping[str, Any]],
    collected_count: int,
    public_count: int,
    rejected: Sequence[Mapping[str, str]],
    expanded_query_has_type_aliases: bool,
) -> None:
    logger.info(
        "%s",
        {
            "event": "image_retrieval_diagnostics",
            "trace_id": trace_id,
            "query": query,
            "expanded_query_has_type_aliases": (
                expanded_query_has_type_aliases
            ),
            "retrieved_candidates": list(retrieved_summaries),
            "reranked_candidates": list(reranked_summaries),
            "collected_before_url_conversion": collected_count,
            "public_after_url_conversion": public_count,
            "rejected_url_conversions": list(rejected),
        },
    )


def to_public_image_urls(
    images: Sequence[RetrievedImage],
    data_dir: str | Path | None,
    public_base_url: str | None = None,
) -> list[RetrievedImage]:
    """Rewrite local corpus paths to API media URLs. Keep HTTP(S) refs."""

    rewritten, _rejected = convert_retrieved_images(
        images,
        data_dir,
        public_base_url,
    )
    return rewritten


def to_public_image_url(
    reference: str,
    data_dir: str | Path | None,
) -> str:
    media_url, _reason = explain_public_image_url(reference, data_dir)
    return media_url


def _join_public_base_url(
    public_base_url: str | None,
    relative_url: str,
) -> str | None:
    base = (public_base_url or "").strip().rstrip("/")

    if not base:
        return None

    path = (
        relative_url
        if relative_url.startswith("/")
        else f"/{relative_url}"
    )
    return f"{base}{path}"


def media_file_from_data_dir(
    file_path: str,
    data_dir: str | Path,
) -> Path:
   
    relative = _relative_to_data_dir(file_path, data_dir)

    if relative is None or not relative.parts:
        raise ValueError("Invalid media path.")

    root = Path(data_dir).resolve()
    candidate = (root / relative).resolve()

    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("Invalid media path.") from error

    if not candidate.is_file():
        raise ValueError("Image file does not exist.")

    return candidate


def collect_image_dicts(
    metadata: Mapping[str, Any],
    *,
    source_id: str,
    default_source: str | None = None,
    default_media_type: str | None = None,
) -> list[dict[str, str | None]]:
    media_type = (
        _optional_str(metadata.get("media_type"))
        or default_media_type
    )
    file_type = _optional_str(metadata.get("file_type"))
    pending: list[tuple[Any, bool]] = []

    for key in _IMAGE_SCALAR_KEYS:
        if key in metadata:
            pending.append((metadata[key], False))

    if file_type == "image" or _is_image_media_type(media_type):
        for key in ("url", "path", "file_path", "storage_path"):
            if key in metadata:
                pending.append((metadata[key], False))

    for key in _IMAGE_LIST_KEYS:
        value = metadata.get(key)

        if value is None:
            continue

        items = value if isinstance(value, list) else [value]
        require_image_like = key == "attachments"

        for item in items:
            pending.append((item, require_image_like))

    parsed: list[dict[str, str | None]] = []

    for item, require_image_like in pending:
        parsed_item = _parse_image_item(
            item,
            source_id=source_id,
            default_source=default_source,
            default_media_type=media_type,
            require_image_like=require_image_like,
        )

        if parsed_item is not None:
            parsed.append(parsed_item)

    return _unique_image_dicts(parsed)


def build_image_description(
    image: ImageDocument,
    *,
    metadata: Mapping[str, Any] | None = None,
    data_dir: str | Path | None = None,
) -> str:
   
    meta = dict(metadata or {})
    explicit = _first_non_empty_str(
        meta.get("image_description"),
        meta.get("description"),
        meta.get("caption"),
        meta.get("alt_text"),
    )
    path = Path(image.image_path)
    relative = _relative_to_data_dir(str(path), data_dir)
    stem = path.stem
    file_name = meta.get("file_name") or image.title or path.name
    tokens = _filename_tokens(stem)

    if relative is not None:
        for part in relative.parts[:-1]:
            tokens.extend(_filename_tokens(Path(part).stem))

    type_labels, label_tokens = _labels_from_tokens(tokens)
    provided_name = _first_non_empty_str(
        meta.get("patient_name"),
        meta.get("subject_name"),
        meta.get("person_name"),
    )
    name_label = provided_name or " ".join(label_tokens)

    if explicit:
        parts = [explicit.strip()]
    elif type_labels and name_label:
        english_type = type_labels[0]
        parts = [
            f"{english_type.capitalize()} image labeled {name_label} "
            f"(from filename, not verified identity)."
        ]
    elif type_labels:
        parts = [
            f"{type_labels[0].capitalize()} image. "
            "No person name was present in the filename or metadata."
        ]
    elif name_label:
        parts = [
            f"Medical image labeled {name_label} "
            f"(from filename, not verified identity)."
        ]
    else:
        parts = [
            f"Medical image file {file_name}."
        ]

    if type_labels:
        unique_types: list[str] = []
        seen_types: set[str] = set()

        for label in type_labels:
            key = label.casefold()

            if key in seen_types:
                continue

            seen_types.add(key)
            unique_types.append(label)

        parts.append(
            "Document type aliases: " + ", ".join(unique_types) + "."
        )

    aliases = _search_aliases(name_label, type_labels)

    if aliases:
        parts.append("Searchable aliases: " + ", ".join(aliases) + ".")

    parts.append(f"Filename: {file_name}.")
    return " ".join(parts)


def _filename_tokens(stem: str) -> list[str]:
    return [
        token
        for token in _TOKEN_SPLIT.split(stem)
        if token and token.casefold() not in _GENERIC_FILENAME_TOKENS
    ]


def _labels_from_tokens(
    tokens: Sequence[str],
) -> tuple[list[str], list[str]]:
    type_labels: list[str] = []
    name_tokens: list[str] = []

    for token in tokens:
        hints = _TYPE_HINTS.get(token.casefold())

        if hints:
            type_labels.extend(hints)
            continue

        name_tokens.append(token if token.isupper() else token.capitalize())

    return type_labels, name_tokens


def _search_aliases(
    name_label: str,
    type_labels: Sequence[str],
) -> list[str]:
    aliases: list[str] = []
    seen: set[str] = set()
    name_parts = [part for part in name_label.split() if part]
    arabic_types = [
        label
        for label in type_labels
        if any("\u0600" <= char <= "\u06FF" for char in label)
    ]
    english_types = [
        label
        for label in type_labels
        if label not in arabic_types
    ]

    candidates: list[str] = []

    if name_parts and english_types:
        first = name_parts[0]
        full = " ".join(name_parts)
        primary = english_types[0]
        candidates.extend(
            [
                f"{full} {primary}",
                f"{first} {primary}",
                f"{primary} for {full}",
                f"{primary} for {first}",
            ]
        )

        for arabic in arabic_types:
            candidates.extend(
                [
                    f"{arabic} {full}",
                    f"{arabic} {first}",
                ]
            )

            folded = normalize_arabic_orthography(arabic)

            if folded != arabic:
                candidates.extend(
                    [
                        f"{folded} {full}",
                        f"{folded} {first}",
                    ]
                )

    for candidate in candidates:
        key = candidate.casefold()

        if key in seen:
            continue

        seen.add(key)
        aliases.append(candidate)

    return aliases


def _searchable_image_text(
    image: ImageDocument,
    data_dir: str | Path | None = None,
    description: str | None = None,
) -> str:
    path = Path(image.image_path)
    tokens: list[str] = []

    if description:
        # Description already includes filename and type aliases.
        # Do not append the same tokens again or the image may split
        # into extra chunks and be embedded more than once.
        text = description.strip()

        if not text:
            raise ValueError(
                f"Cannot index image without filename provenance: "
                f"{image.source_id}"
            )

        return text

    for value in (
        image.title,
        path.name,
        path.stem.replace("_", " ").replace("-", " "),
    ):
        if value:
            tokens.append(str(value))

    relative = _relative_to_data_dir(str(path), data_dir)

    if relative is not None:
        for part in relative.parts:
            cleaned = (
                Path(part).stem.replace("_", " ").replace("-", " ").strip()
            )

            if cleaned and cleaned not in {".", ".."}:
                tokens.append(cleaned)

    unique: list[str] = []
    seen: set[str] = set()

    for token in tokens:
        key = token.casefold()

        if key in seen:
            continue

        seen.add(key)
        unique.append(token)

    text = " ".join(unique).strip()

    if not text:
        raise ValueError(
            f"Cannot index image without filename provenance: "
            f"{image.source_id}"
        )

    return text


def _storage_reference(
    reference: str,
    data_dir: str | Path | None,
) -> str:
    if _is_http_url(reference):
        return reference.strip()

    relative = _relative_to_data_dir(reference, data_dir)

    if relative is not None:
        return relative.as_posix()

    return reference.strip()


def _relative_to_data_dir(
    reference: str,
    data_dir: str | Path | None,
) -> Path | None:
    text = (reference or "").strip().replace("\\", "/")

    if not text or _is_http_url(text):
        return None

    if ".." in Path(text).parts:
        return None

    if not _looks_absolute(text):
        return Path(text)

    if data_dir is not None:
        try:
            resolved = Path(reference)
            try:
                resolved = resolved.resolve()
            except OSError:
                resolved = resolved.absolute()

            relative = resolved.relative_to(Path(data_dir).resolve())

            if ".." in relative.parts:
                return None

            return relative
        except (OSError, ValueError):
            pass

    marker = "/data/"
    lowered = text.casefold()
    index = lowered.rfind(marker)

    if index == -1:
        return None

    suffix = text[index + len(marker) :]

    if not suffix or ".." in Path(suffix).parts:
        return None

    return Path(suffix)


def _looks_absolute(path: str) -> bool:
    if path.startswith("/"):
        return True

    return len(path) >= 3 and path[1] == ":" and path[2] == "/"


def _is_http_url(value: str) -> bool:
    lowered = value.strip().casefold()
    return lowered.startswith("http://") or lowered.startswith("https://")


def _record_fields(
    record: Mapping[str, Any] | Any,
) -> tuple[str | None, str | None, Mapping[str, Any]]:
    if isinstance(record, Mapping):
        source_id = _optional_str(record.get("source_id"))
        chunk_id = _optional_str(record.get("chunk_id"))
        metadata = record.get("metadata") or {}

        if isinstance(metadata, Mapping):
            return source_id, chunk_id, metadata

        return source_id, chunk_id, {}

    source_id = _optional_str(getattr(record, "source_id", None))
    chunk_id = _optional_str(getattr(record, "chunk_id", None))
    metadata = getattr(record, "metadata", None) or {}

    identity = getattr(record, "identity", None)

    if identity is not None:
        source_id = source_id or _optional_str(
            getattr(identity, "source_id", None)
        )
        chunk_id = chunk_id or _optional_str(
            getattr(identity, "chunk_id", None)
        )

    provenance = getattr(record, "provenance", None)

    if provenance is not None:
        source_id = source_id or _optional_str(
            getattr(provenance, "source_id", None)
        )
        provenance_metadata = getattr(provenance, "metadata", None)

        if isinstance(provenance_metadata, Mapping) and provenance_metadata:
            metadata = provenance_metadata

    if isinstance(metadata, Mapping):
        return source_id, chunk_id, metadata

    return source_id, chunk_id, {}


def _parse_image_item(
    item: Any,
    *,
    source_id: str,
    default_source: str | None,
    default_media_type: str | None,
    require_image_like: bool,
) -> dict[str, str | None] | None:
    if item is None:
        return None

    if isinstance(item, str):
        url = item.strip()

        if not url:
            return None

        if require_image_like and not _looks_like_image_ref(
            url,
            default_media_type,
        ):
            return None

        return {
            "url": url,
            "media_type": default_media_type,
            "image_type": None,
            "source_id": source_id,
            "source": default_source,
            "document_id": source_id,
        }

    if not isinstance(item, Mapping):
        return None

    url = _first_non_empty_str(
        *(item.get(key) for key in _URL_KEYS)
    )

    if not url:
        return None

    media_type = (
        _optional_str(item.get("media_type"))
        or default_media_type
    )
    image_type = _optional_str(
        item.get("image_type") or item.get("file_type")
    )

    if require_image_like and not _looks_like_image_ref(
        url,
        media_type,
        image_type,
    ):
        return None

    document_id = _optional_str(
        item.get("document_id") or item.get("source_id")
    ) or source_id

    return {
        "url": url,
        "media_type": media_type,
        "image_type": image_type,
        "source_id": _optional_str(item.get("source_id")) or source_id,
        "source": (
            _optional_str(
                item.get("source")
                or item.get("source_title")
                or item.get("file_name")
            )
            or default_source
        ),
        "document_id": document_id,
    }


def _unique_image_dicts(
    items: Iterable[Mapping[str, str | None]],
) -> list[dict[str, str | None]]:
    unique: list[dict[str, str | None]] = []
    seen: set[str] = set()

    for item in items:
        url = _optional_str(item.get("url"))

        if not url:
            continue

        key = url.casefold()

        if key in seen:
            continue

        seen.add(key)
        unique.append(dict(item))
        unique[-1]["url"] = url

    return unique


def _looks_like_image_ref(
    value: str,
    media_type: str | None = None,
    image_type: str | None = None,
) -> bool:
    if _is_image_media_type(media_type):
        return True

    if image_type and image_type.casefold() not in {
        "document",
        "text",
        "pdf",
    }:
        return True

    path = value.split("?", 1)[0].split("#", 1)[0]
    suffix = Path(path).suffix.casefold()

    return suffix in _IMAGE_EXTENSIONS


def _is_image_media_type(media_type: str | None) -> bool:
    if not media_type:
        return False

    return media_type.casefold().startswith("image/")


def _alias_mentioned(alias: str, folded_query: str) -> bool:
    needle = normalize_arabic_orthography(alias).casefold().strip()

    if not needle:
        return False

    pattern = re.compile(
        r"(?<!\w)" + re.escape(needle) + r"(?!\w)",
        re.UNICODE,
    )
    return pattern.search(folded_query) is not None


def _safe_score(record: Mapping[str, Any] | Any) -> float | None:
    if isinstance(record, Mapping):
        raw = record.get("score", record.get("rerank_score"))
    else:
        raw = getattr(record, "rerank_score", None)
        if raw is None:
            raw = getattr(record, "score", None)
            if raw is None:
                raw = getattr(record, "original_score", None)

    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None

    if not (value == value) or value in {float("inf"), float("-inf")}:
        return None

    return value


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None

    return None


def _first_non_empty_str(*values: Any) -> str | None:
    for value in values:
        text = _optional_str(value)

        if text:
            return text

    return None
