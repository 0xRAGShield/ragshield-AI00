from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.models.image_refs import (
    IMAGE_FILE_EXTENSIONS,
    media_file_from_data_dir,
)


MEDIA_ROUTE_PREFIX = "/media"

_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def build_media_router(data_dir: str | Path) -> APIRouter:
    """Serve image files that live under the ingestion data directory."""

    router = APIRouter()
    root = Path(data_dir)

    @router.get(f"{MEDIA_ROUTE_PREFIX}/{{file_path:path}}")
    def serve_media(file_path: str) -> FileResponse:
        try:
            resolved = media_file_from_data_dir(
                file_path,
                root,
            )
        except ValueError:
            raise HTTPException(
                status_code=404,
                detail="Image was not found.",
            ) from None

        suffix = resolved.suffix.casefold()

        if suffix not in IMAGE_FILE_EXTENSIONS:
            raise HTTPException(
                status_code=404,
                detail="Image was not found.",
            )

        return FileResponse(
            resolved,
            media_type=_MEDIA_TYPES.get(suffix),
        )

    return router
