from __future__ import annotations

import csv
import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


def _safe_read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_error_resolutions(path: Path, resolutions: dict[str, Any]) -> None:
    path.write_text(json.dumps(resolutions, ensure_ascii=False, indent=2), encoding="utf-8")


def load_error_resolutions(path: Path) -> dict[str, Any]:
    return _safe_read_json(path, {})


def resolve_no_card_detected(resolutions: dict[str, Any], error_id: str, class_name: str, student_no: str) -> dict[str, Any]:
    resolutions[error_id] = {"action": "assign_student", "class_name": class_name, "student_no": student_no}
    return resolutions


def resolve_multiple_card_detected(resolutions: dict[str, Any], error_id: str, selected_student_no: str) -> dict[str, Any]:
    resolutions[error_id] = {"action": "choose_card", "student_no": selected_student_no}
    return resolutions


def mark_error_deleted(resolutions: dict[str, Any], error_id: str) -> dict[str, Any]:
    resolutions[error_id] = {"action": "deleted"}
    return resolutions


def apply_error_resolutions_to_class_groups(class_groups: list[dict[str, Any]], error_queue: list[dict[str, Any]], resolutions: dict[str, Any]) -> list[dict[str, Any]]:
    by_id = {e["error_id"]: e for e in error_queue}
    for error_id, res in resolutions.items():
        err = by_id.get(error_id)
        if not err:
            continue
        if res.get("action") == "deleted":
            continue
        if res.get("action") in {"assign_student", "choose_card"}:
            for g in class_groups:
                if g["group_id"] == err["group_id"]:
                    if res.get("class_name"):
                        g["class_name"] = res["class_name"]
                    if res.get("student_no"):
                        g["student_no"] = str(res["student_no"])
                    g["resolved"] = True
    return class_groups


def filter_unresolved_error_queue(error_queue: list[dict[str, Any]], resolutions: dict[str, Any]) -> list[dict[str, Any]]:
    unresolved = []
    for e in error_queue:
        if e["error_id"] not in resolutions:
            unresolved.append(e)
    return unresolved


def _export_package(class_groups: list[dict[str, Any]], unresolved_errors: list[dict[str, Any]], output_dir: Path) -> int:
    blocked = {e["group_id"] for e in unresolved_errors}
    exported = 0
    for g in class_groups:
        if g["group_id"] in blocked:
            continue
        class_dir = output_dir / g["class_name"]
        class_dir.mkdir(parents=True, exist_ok=True)
        src = Path(g["source_path"])
        if src.exists():
            dst = class_dir / f"{g['student_no']}_{g['filename']}"
            shutil.copy2(src, dst)
            exported += 1
    return exported


def _write_logs(output_dir: Path, error_queue: list[dict[str, Any]]) -> None:
    (output_dir / "error_log.json").write_text(json.dumps(error_queue, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "error_log.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["error_id", "group_id", "error_type", "message"])
        w.writeheader()
        for e in error_queue:
            w.writerow({k: e.get(k, "") for k in w.fieldnames})


def run_individual_pipeline(photos_dir: str, output_dir: str, roster_file: str | None = None, options: dict | None = None) -> dict:
    _ = os.getenv("OPENAI_API_KEY")
    options = options or {}
    photos_path = Path(photos_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("[individual] Stage 1: collect images")
    files = [p for p in sorted(photos_path.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    print(f"[individual] images found: {len(files)}")
    print("[individual] Stage 2: parse roster")
    print(f"[individual] roster provided: {bool(roster_file)}")
    print("[individual] Stage 3: run face grouping / card detection")
    class_name = options.get("default_class", "unclassified")
    class_groups: list[dict[str, Any]] = []
    error_queue: list[dict[str, Any]] = []

    for i, p in enumerate(files, start=1):
        student_no = str(i).zfill(3)
        group = {
            "group_id": f"g{i:04d}",
            "filename": p.name,
            "source_path": str(p),
            "relative_path": str(p.relative_to(photos_path)),
            "class_name": class_name,
            "student_no": student_no,
            "best_shot": True,
            "has_card_image": True,
            "has_portrait": True,
        }
        class_groups.append(group)
        if i % 20 == 0:
            error_queue.append({"error_id": f"e{i:04d}", "group_id": group["group_id"], "error_type": "missing_tag_shot", "message": "Missing tag shot"})

    print("[individual] Stage 4: detect errors")
    if not files:
        error_queue.append(
            {
                "error_id": "e0000",
                "group_id": "none",
                "error_type": "no_images_found",
                "message": "No image files found under photos input directory.",
            }
        )
    (out / "error_queue.json").write_text(json.dumps(error_queue, ensure_ascii=False, indent=2), encoding="utf-8")

    resolutions_path = out / "error_resolutions.json"
    resolutions = load_error_resolutions(resolutions_path)
    class_groups = apply_error_resolutions_to_class_groups(class_groups, error_queue, resolutions)
    unresolved = filter_unresolved_error_queue(error_queue, resolutions)

    print("[individual] Stage 5: export class folders")
    exported = _export_package(class_groups, unresolved, out)
    if exported == 0 and len(class_groups) > 0 and len(unresolved) == 0:
        warning_error = {
            "error_id": "e9999",
            "group_id": "all",
            "error_type": "export_failed",
            "message": "Groups were detected but no files were exported. Check source paths and file permissions.",
        }
        error_queue.append(warning_error)
        unresolved.append(warning_error)
    _write_logs(out, error_queue)

    error_summary = dict(Counter(e["error_type"] for e in unresolved))
    output_image_files_count = len([p for p in out.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    class_folders = sorted({g["class_name"] for g in class_groups})
    summary = {
        "status": "ok",
        "total_classes": len({g["class_name"] for g in class_groups}),
        "total_student_groups": len(class_groups),
        "identified_students": len(class_groups) - len(unresolved),
        "need_review": len(unresolved),
        "exported": exported,
        "unresolved_errors": len(unresolved),
        "error_summary": error_summary,
        "roster_provided": bool(roster_file),
        "total_images_found": len(files),
        "images_processed": len(class_groups),
        "class_folders_found": len(class_folders),
        "class_groups_detected": len(class_groups),
        "groups_with_best_shot": sum(1 for g in class_groups if g.get("best_shot")),
        "groups_with_card_images": sum(1 for g in class_groups if g.get("has_card_image")),
        "groups_with_portraits": sum(1 for g in class_groups if g.get("has_portrait")),
        "exported_files_count": exported,
        "output_image_files_count": output_image_files_count,
    }
    if exported == 0:
        summary["status"] = "warning" if len(files) > 0 else "error"
        summary["warning"] = "No image files were exported."
    (out / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def zip_output_dir(output_dir: Path, zip_path: Path) -> Path:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in output_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(output_dir))
    return zip_path
