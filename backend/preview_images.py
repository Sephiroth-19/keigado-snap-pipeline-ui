from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
FINAL_CATEGORY_DIRS = {
    "final_selected": ("BestShot", "best"),
    "bestshot": ("BestShot", "best"),
    "best_shot": ("BestShot", "best"),
    "best shot": ("BestShot", "best"),
    "passing": ("Passing", "passing"),
    "passing photos": ("Passing", "passing"),
    "other_passing": ("Passing", "passing"),
    "ng_photos": ("NG", "ng"),
    "ng": ("NG", "ng"),
}


def _display_cluster_name(folder_name: str) -> str:
    if folder_name.lower().startswith("cluster_"):
        suffix = folder_name.split("_", 1)[1]
        if suffix.isdigit():
            return f"Cluster {int(suffix)}"
    return folder_name


def _event_name_for(path: Path, output_root: Path, marker: str) -> str | None:
    rel = path.relative_to(output_root)
    parts = rel.parts
    if marker in parts:
        idx = parts.index(marker)
        if idx > 0:
            return parts[idx - 1]
    return None


def _image_payload(path: Path, output_root: Path, category: str, category_key: str, **extra: Any) -> dict[str, Any]:
    rel = path.relative_to(output_root).as_posix()
    payload: dict[str, Any] = {
        "workflow_type": "Snap Photo",
        "category": category,
        "category_key": category_key,
        "relative_path": rel,
        "url": f"/api/snap/preview-file?path={quote(rel)}",
        "name": path.name,
    }
    payload.update({k: v for k, v in extra.items() if v not in (None, "")})
    return payload


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_snap_preview_images(output_root: Path, mode: str = "final") -> list[dict[str, Any]]:
    """Collect read-only Snap preview image metadata from runtime/output."""
    if not output_root.exists():
        return []

    normalized_mode = (mode or "final").lower()
    include_final = normalized_mode in {"final", "all"}
    include_similarity = normalized_mode in {"similarity", "all"}
    include_candidates = normalized_mode in {"candidates", "all"}
    images: list[dict[str, Any]] = []

    if include_final:
        for folder in sorted(output_root.rglob("*")):
            if not folder.is_dir():
                continue
            key = folder.name.lower()
            if key not in FINAL_CATEGORY_DIRS:
                continue
            category, category_key = FINAL_CATEGORY_DIRS[key]
            event_name = folder.parent.name if folder.parent != output_root else None
            for path in sorted(p for p in folder.rglob("*") if _is_image(p)):
                images.append(_image_payload(path, output_root, category, category_key, event_name=event_name))

    if include_similarity:
        for sim_dir in sorted(p for p in output_root.rglob("similarity_clusters") if p.is_dir()):
            event_name = _event_name_for(sim_dir, output_root, "similarity_clusters")
            for path in sorted(p for p in sim_dir.rglob("*") if _is_image(p)):
                rel_parts = path.relative_to(sim_dir).parts
                cluster_folder = rel_parts[0] if len(rel_parts) > 1 else None
                images.append(
                    _image_payload(
                        path,
                        output_root,
                        "Similarity Group",
                        "similarity",
                        event_name=event_name,
                        cluster_name=_display_cluster_name(cluster_folder) if cluster_folder else None,
                        cluster_folder=cluster_folder,
                    )
                )

    if include_candidates:
        for candidates_dir in sorted(p for p in output_root.rglob("dedup_candidates") if p.is_dir()):
            event_name = _event_name_for(candidates_dir, output_root, "dedup_candidates")
            for path in sorted(p for p in candidates_dir.rglob("*") if _is_image(p)):
                images.append(_image_payload(path, output_root, "Candidate", "candidates", event_name=event_name))

    return images
