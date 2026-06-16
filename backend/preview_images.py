from __future__ import annotations

import json
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
BUCKET_ALIASES = {
    "final_selected": "best",
    "best": "best",
    "bestshot": "best",
    "other_passing": "passing",
    "passing": "passing",
    "ng_photos": "ng",
    "ng": "ng",
    "similarity_clusters": "similarity",
    "similarity": "similarity",
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


def _load_event_metadata(output_root: Path) -> dict[str, dict[str, Any]]:
    summary_path = output_root / "all_events_summary.json"
    if not summary_path.exists():
        return {}
    try:
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(rows, list):
        return {}
    metadata: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        output_folder = str(row.get("output_folder") or row.get("event_name") or "")
        if output_folder:
            metadata[output_folder] = row
    return metadata


def _event_extra(output_root: Path, event_folder: str | None, event_metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not event_folder:
        return {}
    row = event_metadata.get(event_folder, {})
    display_name = row.get("display_event_name") or event_folder
    return {
        "event_id": row.get("event_id"),
        "event_name": display_name,
        "display_event_name": display_name,
        "safe_event_name": row.get("event_name") or event_folder,
        "output_folder": row.get("output_folder") or event_folder,
    }


def _matches_event(event_data: dict[str, Any], event_id: str | None, event_name: str | None) -> bool:
    if event_id and event_data.get("event_id") != event_id:
        return False
    if event_name:
        candidates = {
            str(event_data.get("event_name") or ""),
            str(event_data.get("display_event_name") or ""),
            str(event_data.get("safe_event_name") or ""),
            str(event_data.get("output_folder") or ""),
        }
        if event_name not in candidates:
            return False
    return True


def _image_payload(path: Path, output_root: Path, category: str, category_key: str, bucket: str, **extra: Any) -> dict[str, Any]:
    rel = path.relative_to(output_root).as_posix()
    payload: dict[str, Any] = {
        "workflow_type": "Snap Photo",
        "category": category,
        "category_key": category_key,
        "bucket": bucket,
        "relative_path": rel,
        "url": f"/api/snap/preview-file?path={quote(rel)}",
        "name": path.name,
    }
    payload.update({k: v for k, v in extra.items() if v not in (None, "")})
    return payload


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_snap_preview_images(
    output_root: Path,
    mode: str = "final",
    event_id: str | None = None,
    event_name: str | None = None,
    bucket: str | None = None,
) -> list[dict[str, Any]]:
    """Collect read-only Snap preview image metadata from runtime/output."""
    if not output_root.exists():
        return []

    normalized_mode = (mode or "final").lower()
    bucket_key = BUCKET_ALIASES.get((bucket or "").lower())
    include_final = normalized_mode in {"final", "all"} or bucket_key in {"best", "passing", "ng"}
    include_similarity = normalized_mode in {"similarity", "all"} or bucket_key == "similarity"
    include_candidates = normalized_mode in {"candidates", "all"}
    images: list[dict[str, Any]] = []
    event_metadata = _load_event_metadata(output_root)

    if include_final:
        for folder in sorted(output_root.rglob("*")):
            if not folder.is_dir():
                continue
            key = folder.name.lower()
            if key not in FINAL_CATEGORY_DIRS:
                continue
            category, category_key = FINAL_CATEGORY_DIRS[key]
            if bucket_key and category_key != bucket_key:
                continue
            event_folder = folder.parent.name if folder.parent != output_root else None
            event_data = _event_extra(output_root, event_folder, event_metadata)
            if not _matches_event(event_data, event_id, event_name):
                continue
            for path in sorted(p for p in folder.rglob("*") if _is_image(p)):
                images.append(_image_payload(path, output_root, category, category_key, key, **event_data))

    if include_similarity:
        for sim_dir in sorted(p for p in output_root.rglob("similarity_clusters") if p.is_dir()):
            event_folder = _event_name_for(sim_dir, output_root, "similarity_clusters")
            event_data = _event_extra(output_root, event_folder, event_metadata)
            if not _matches_event(event_data, event_id, event_name):
                continue
            for path in sorted(p for p in sim_dir.rglob("*") if _is_image(p)):
                rel_parts = path.relative_to(sim_dir).parts
                cluster_folder = rel_parts[0] if len(rel_parts) > 1 else None
                images.append(
                    _image_payload(
                        path,
                        output_root,
                        "Similarity Group",
                        "similarity",
                        "similarity_clusters",
                        **event_data,
                        cluster_name=_display_cluster_name(cluster_folder) if cluster_folder else None,
                        cluster_folder=cluster_folder,
                    )
                )

    if include_candidates:
        for candidates_dir in sorted(p for p in output_root.rglob("dedup_candidates") if p.is_dir()):
            event_folder = _event_name_for(candidates_dir, output_root, "dedup_candidates")
            event_data = _event_extra(output_root, event_folder, event_metadata)
            if not _matches_event(event_data, event_id, event_name):
                continue
            for path in sorted(p for p in candidates_dir.rglob("*") if _is_image(p)):
                images.append(_image_payload(path, output_root, "Candidate", "candidates", "dedup_candidates", **event_data))

    return images
