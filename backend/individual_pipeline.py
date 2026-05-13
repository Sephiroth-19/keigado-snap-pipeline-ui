from __future__ import annotations

import inspect
import json, os, zipfile
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from backend.individual.roster_parser import parse_roster, save_roster_json
from backend.individual.face_grouper import FaceGrouper
from backend.individual.error_handling import (
    detect_pipeline_errors,
    save_error_queue,
    attach_error_tags_to_groups,
    load_error_resolutions as _load_error_resolutions_impl,
    apply_error_resolutions_to_class_groups,
    filter_unresolved_error_queue,
    build_exportable_class_groups,
    export_error_items,
    write_error_log_json,
    write_error_log_csv,
)
from backend.individual.package_exporter import export_all_classes



# Compatibility wrappers used by backend.app manual-resolution endpoints
def save_error_resolutions(path: Path | str, resolutions):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(resolutions, ensure_ascii=False, indent=2), encoding="utf-8")


def load_error_resolutions(path: Path | str):
    p = Path(path)
    if not p.exists():
        return []
    try:
        return _load_error_resolutions_impl(p)
    except Exception:
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []


def resolve_no_card_detected(resolutions, error_id, class_name, student_no):
    resolutions = list(resolutions or [])
    resolutions.append({
        "error_id": error_id,
        "resolution_type": "manual_student_number",
        "class_id": class_name,
        "student_no": str(student_no),
        "status": "resolved",
    })
    return resolutions


def resolve_multiple_card_detected(resolutions, error_id, selected_student_no):
    resolutions = list(resolutions or [])
    resolutions.append({
        "error_id": error_id,
        "resolution_type": "manual_candidate_selection",
        "student_no": str(selected_student_no),
        "status": "resolved",
    })
    return resolutions


def mark_error_deleted(resolutions, error_id):
    resolutions = list(resolutions or [])
    resolutions.append({
        "error_id": error_id,
        "resolution_type": "manual_delete",
        "status": "deleted",
    })
    return resolutions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


def _list_input_images(photos_dir: str) -> list[str]:
    base = Path(photos_dir)
    return [str(p) for p in base.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def _safe_dump(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _summarize_class_groups(class_groups, limit: int = 3):
    rows = []
    if not isinstance(class_groups, dict):
        return rows
    for idx, (class_name, cg) in enumerate(class_groups.items()):
        if idx >= limit:
            break
        students = getattr(cg, "students", []) or []
        for sidx, sg in enumerate(students[:1]):
            best = getattr(sg, "best_shot", None)
            rows.append({
                "class_name": class_name,
                "student_no": getattr(sg, "attendance_number", None),
                "group_id": f"{class_name}::{sidx}",
                "card_images_count": len(getattr(sg, "card_images", []) or []),
                "portrait_images_count": len(getattr(sg, "portraits", []) or []),
                "best_shot_path": getattr(best, "path", None) if best else None,
            })
    return rows


def run_individual_pipeline(photos_dir:str, output_dir:str, roster_file:str|None=None, options:dict|None=None)->dict:
    load_dotenv()
    options = options or {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        input_images = _list_input_images(photos_dir)
        print(f"[individual][debug] total input images: {len(input_images)}")
        _safe_dump(out / "debug_input_images.json", {"total": len(input_images), "images": input_images})

        school_name = options.get("school_name")
        year = options.get("year")
        scoring = options.get("scoring", "local")
        class_mapping = options.get("class_mapping")
        max_backups = int(options.get("max_backups", 0))
        openai_api_key = options.get("openai_api_key") or os.getenv("OPENAI_API_KEY")

        roster = parse_roster(roster_file, school_name=school_name, openai_api_key=openai_api_key) if roster_file else {}
        save_roster_json(roster, out / "roster.json")
        classes = (roster.get("classes") or {}) if isinstance(roster, dict) else {}
        student_count = sum(len((v or {}).get("students", [])) for v in classes.values()) if isinstance(classes, dict) else 0
        print(f"[individual][debug] roster type={type(roster).__name__} classes={len(classes)} students={student_count}")
        _safe_dump(out / "debug_roster_summary.json", {
            "roster_type": type(roster).__name__,
            "classes_count": len(classes),
            "students_count": student_count,
        })

        openai_client = OpenAI(api_key=openai_api_key) if (scoring == "openai" and openai_api_key) else None
        grouper = FaceGrouper(valid_numbers=set(), scoring=scoring, openai_client=openai_client, roster=roster)
        class_groups = grouper.process_folder(photos_dir)
        class_groups_count = len(class_groups) if hasattr(class_groups, "__len__") else -1
        print(f"[individual][debug] class_groups type={type(class_groups).__name__} len={class_groups_count}")
        _safe_dump(out / "debug_class_groups_raw.json", {
            "type": type(class_groups).__name__,
            "len": class_groups_count,
            "sample": _summarize_class_groups(class_groups),
        })
        if class_groups_count == 0:
            reason = "FaceGrouper.process_folder returned 0 class_groups"
            summary = {
                "status": "error",
                "pipeline_mode": "real_pipeline_debug",
                "reason": reason,
                "debug_stage": "face_grouper",
                "total_images_found": len(input_images),
                "class_groups_count": 0,
                "exported": 0,
            }
            _safe_dump(out / "summary.json", summary)
            return summary

        print(f"[individual] calling detect_pipeline_errors with args: class_groups={type(class_groups).__name__}, roster={type(roster).__name__}, photos_dir={photos_dir}")
        error_queue = detect_pipeline_errors(class_groups, roster, photos_dir)
        print(f"[individual][debug] error_queue len={len(error_queue)}")
        _safe_dump(out / "debug_error_queue_after_detection.json", error_queue)
        save_error_queue(error_queue, out / "error_queue.json")
        attach_error_tags_to_groups(class_groups, error_queue)

        resolutions = load_error_resolutions(out / "error_resolutions.json")
        if not isinstance(resolutions, list):
            resolutions = []

        try:
            print(f"[individual] calling apply_error_resolutions_to_class_groups with args: class_groups={type(class_groups).__name__}, resolutions={len(resolutions)}")
            resolved_class_groups = apply_error_resolutions_to_class_groups(class_groups, resolutions)
            print(f"[individual][debug] class_groups after resolution len={len(resolved_class_groups)}")
            _safe_dump(out / "debug_class_groups_after_resolution.json", {
                "len": len(resolved_class_groups),
                "sample": _summarize_class_groups(resolved_class_groups),
            })

            print(f"[individual] calling filter_unresolved_error_queue with args: error_queue={len(error_queue)}, resolutions={len(resolutions)}")
            unresolved = filter_unresolved_error_queue(error_queue, resolutions)
            print(f"[individual][debug] unresolved len={len(unresolved)}")
            _safe_dump(out / "debug_unresolved_errors.json", unresolved)

            print(f"[individual] calling build_exportable_class_groups with args: class_groups={type(resolved_class_groups).__name__}, error_queue={len(unresolved)}")
            exportable = build_exportable_class_groups(resolved_class_groups, unresolved)
            _safe_dump(out / "debug_exportable_groups.json", {
                "len": len(exportable) if hasattr(exportable, "__len__") else -1,
                "sample": _summarize_class_groups(exportable),
            })
        except Exception as resolution_exc:
            resolution_reason = f"error-resolution step failed: {resolution_exc}"
            unresolved = error_queue
            resolved_class_groups = class_groups
            exportable = class_groups
            (out / "summary.json").write_text(json.dumps({
                "status": "warning",
                "pipeline_mode": "real_pipeline",
                "reason": resolution_reason,
                "resolution_step_failed": True,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            _safe_dump(out / "debug_unresolved_errors.json", unresolved)
            _safe_dump(out / "debug_exportable_groups.json", {"len": len(exportable) if hasattr(exportable, "__len__") else -1})

        export_error_items(unresolved, out)
        write_error_log_json(error_queue, out / "error_log.json")
        write_error_log_csv(error_queue, out / "error_log.csv")

        print(f"[individual][debug] export_all_classes signature: {inspect.signature(export_all_classes)}")
        print(f"[individual] calling export_all_classes with args: class_groups_len={len(exportable) if hasattr(exportable,'__len__') else -1}, roster_type={type(roster).__name__}, output_dir={out}, school_name={school_name}, year={year}, class_mapping={class_mapping}, max_backups={max_backups}")
        export_all_classes(exportable, roster, out, school_name=school_name, year=year, class_mapping=class_mapping, max_backups=max_backups)

        out_images=[p for p in out.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        _safe_dump(out / "debug_output_files_after_export.json", {"count": len(out_images), "files": [str(p) for p in out_images]})
        reason = None
        if len(input_images) == 0:
            reason = "no_input_images"
        elif class_groups_count == 0:
            reason = "face_grouper_returned_empty"
        elif hasattr(exportable, "__len__") and len(exportable) == 0:
            reason = "no_exportable_groups"
        elif len(out_images) == 0:
            reason = "export_all_classes_created_no_files"
        summary = {
            "status": "ok" if out_images else "warning",
            "pipeline_mode": "real_pipeline_debug",
            "exported": len(out_images),
            "unresolved_errors": len(unresolved),
            "error_summary": dict(Counter(e.get("error_type", "unknown") for e in unresolved)),
            "reason": reason,
            "debug_stage": "post_export",
            "total_images_found": len(input_images),
            "class_groups_count": class_groups_count,
            "exportable_groups_count": len(exportable) if hasattr(exportable, "__len__") else -1,
            "output_image_files_count": len(out_images),
        }
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    except Exception as e:
        reason = str(e)
        dep_names = ["mediapipe", "easyocr", "insightface", "onnxruntime", "openai", "fitz"]
        missing = next((d for d in dep_names if d in reason.lower()), None)
        summary = {
            "status": "error",
            "pipeline_mode": "real_pipeline_debug",
            "reason": "export_failed_exception",
            "exception": reason,
            "debug_stage": "exception",
            "missing_dependency_hint": f"Install/verify dependency: {missing}" if missing else None,
            "exported": 0,
        }
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "error_log.json").write_text(json.dumps([{"error_type":"pipeline_failure","message":reason}], ensure_ascii=False, indent=2), encoding="utf-8")
        return summary


def zip_output_dir(output_dir:Path, zip_path:Path)->Path:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as zf:
        for p in output_dir.rglob('*'):
            if p.is_file():
                zf.write(p,p.relative_to(output_dir))
    return zip_path
