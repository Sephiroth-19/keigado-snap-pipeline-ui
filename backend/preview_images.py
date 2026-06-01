from __future__ import annotations

import mimetypes
from pathlib import Path
from urllib.parse import quote

from fastapi import HTTPException

PREVIEW_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

PREVIEW_IMAGE_MEDIA_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}


def image_media_type(path: Path) -> str:
    """Return an explicit image media type for preview files."""
    suffix = path.suffix.lower()
    return PREVIEW_IMAGE_MEDIA_TYPES.get(suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def safe_resolve_preview_path(base_dir: Path, relative_path: str) -> Path:
    """Resolve a user-supplied preview path and ensure it stays inside base_dir."""
    if not relative_path:
        raise HTTPException(status_code=400, detail="path is required")

    base = base_dir.resolve()
    candidate = (base / relative_path).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid preview image path") from exc

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Preview image not found")
    if candidate.suffix.lower() not in PREVIEW_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Requested file is not a supported preview image")
    return candidate


def list_preview_images(output_dir: Path, base_url: str, workflow_type: str) -> list[dict[str, str]]:
    """Scan an output directory recursively and return previewable image metadata."""
    if not output_dir.exists() or not output_dir.is_dir():
        return []

    images: list[dict[str, str]] = []
    for path in sorted(output_dir.rglob("*"), key=lambda p: p.relative_to(output_dir).as_posix().lower()):
        if not path.is_file() or path.suffix.lower() not in PREVIEW_IMAGE_EXTENSIONS:
            continue
        relative_path = path.relative_to(output_dir).as_posix()
        images.append(
            {
                "name": path.name,
                "relative_path": relative_path,
                "url": f"{base_url}?path={quote(relative_path, safe='/')}",
                "status": "output",
                "workflow_type": workflow_type,
            }
        )
    return images
