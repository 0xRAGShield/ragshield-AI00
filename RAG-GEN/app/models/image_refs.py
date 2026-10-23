from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.models.evidence_models import ImageDocument, SourceDocument
from app.models.rag_models import RetrievedImage


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


def source_document_from_image(
    image: ImageDocument,
    data_dir: str | Path | None = None,
) -> SourceDocument:
    """
    Index an image using filename and path provenance only.

    This does not OCR, caption, or interpret pixels.
    """

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
            }
        ]
    )

    return SourceDocument(
        source_id=image.source_id,
        title=image.title,
        text=_searchable_image_text(
            image,
            data_dir=data_dir,
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
    """Copy metadata and normalize any existing image references."""

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
    """
    Collect original image references from retrieved records.

    Only references present on retrieved evidence are returned.
    """

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
                )
            )

    return collected


def to_public_image_urls(
    images: Sequence[RetrievedImage],
    data_dir: str | Path | None,
) -> list[RetrievedImage]:
    """Rewrite local corpus paths to API media URLs. Keep HTTP(S) refs."""

    return [
        image.model_copy(
            update={
                "url": to_public_image_url(
                    image.url,
                    data_dir,
                )
            }
        )
        for image in images
    ]


def to_public_image_url(
    reference: str,
    data_dir: str | Path | None,
) -> str:
    text = (reference or "").strip()

    if not text or _is_http_url(text):
        return text

    relative = _relative_to_data_dir(text, data_dir)

    if relative is None:
        return text

    return f"{MEDIA_ROUTE_PREFIX}/{relative.as_posix()}"


def media_file_from_data_dir(
    file_path: str,
    data_dir: str | Path,
) -> Path:
    """
    Resolve a media request to a file under data_dir.

    Rejects absolute paths, parent traversal, and missing files.
    """

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


def _searchable_image_text(
    image: ImageDocument,
    data_dir: str | Path | None = None,
) -> str:
    path = Path(image.image_path)
    tokens: list[str] = []

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
