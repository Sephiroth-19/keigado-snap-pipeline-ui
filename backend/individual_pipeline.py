from __future__ import annotations

import inspect
import json, os, zipfile
from collections import Counter
from pathlib import Path
import re
import base64
import mimetypes

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


def normalize_card_label_with_folder_context(card_label: str, folder_class_context: str | None):
    trans = str.maketrans("０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ－ー", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ--")
    label = str(card_label or "").translate(trans).strip().upper()
    if not label or not folder_class_context:
        return None
    context = str(folder_class_context).strip().upper()
    m_ctx = re.match(r"^(\d+)\s*([A-Z])$", context)
    if not m_ctx:
        return None
    klass = m_ctx.group(2)
    m = re.match(r"^([A-Z])\s*[-]?\s*(\d{1,3})$", label.replace(" ", ""))
    if m:
        card_letter, digits = m.group(1), m.group(2)
        return {
            "class_id": context,
            "student_number": int(digits),
            "normalized_label": f"{context}_{int(digits):03d}",
            "letter_mismatch": card_letter != klass,
        }
    return None


def read_card_label_openai(image_path: str, folder_class_context: str, client: OpenAI | None, model: str):
    empty = {
        "has_card": False,
        "raw_label": "",
        "class_letter": "",
        "student_number": None,
        "confidence": 0.0,
        "reason": "no clear card visible",
    }
    if not client or not image_path:
        return empty
    try:
        mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
        b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
        prompt = (
            "Read ONLY the physical student card held in this photo. "
            f"Folder class context is {folder_class_context}. "
            "Return strict JSON only with keys: has_card, raw_label, class_letter, student_number, confidence, reason. "
            "Accept patterns like A1, D15, H32. If unclear/no card, has_card=false."
        )
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a strict OCR extractor. Return valid JSON only."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ]},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        obj = json.loads(raw)
        return {
            "has_card": bool(obj.get("has_card")),
            "raw_label": str(obj.get("raw_label") or "").strip(),
            "class_letter": str(obj.get("class_letter") or "").strip().upper(),
            "student_number": int(obj["student_number"]) if obj.get("student_number") is not None else None,
            "confidence": float(obj.get("confidence") or 0.0),
            "reason": str(obj.get("reason") or ""),
        }
    except Exception as e:
        out = dict(empty)
        out["reason"] = f"openai_error: {e}"
        return out


def _detect_class_folders(raw_photos_path: Path) -> list[Path]:
    if not raw_photos_path.exists():
        return []
    direct = [p for p in sorted(raw_photos_path.iterdir()) if p.is_dir() and re.match(r"^\d+[A-Za-z]$", p.name)]
    if direct:
        return direct
    nested: list[Path] = []
    for p in sorted(raw_photos_path.iterdir()):
        if not p.is_dir():
            continue
        for c in sorted(p.iterdir()):
            if c.is_dir() and re.match(r"^\d+[A-Za-z]$", c.name):
                nested.append(c)
    return nested


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


def resolve_pipeline_photos_dir(raw_photos_dir: Path) -> tuple[Path, dict]:
    raw_photos_dir = Path(raw_photos_dir)
    recursive_images = [p for p in raw_photos_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    direct_images = [p for p in raw_photos_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS] if raw_photos_dir.exists() else []
    top_level_dirs = [p for p in raw_photos_dir.iterdir() if p.is_dir()] if raw_photos_dir.exists() else []

    chosen = raw_photos_dir
    reason = "default_raw_dir"
    if direct_images:
        chosen = raw_photos_dir
        reason = "images_directly_under_raw_dir"
    elif len(top_level_dirs) == 1:
        only_dir = top_level_dirs[0]
        only_dir_direct_images = [p for p in only_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if only_dir_direct_images:
            chosen = only_dir
            reason = "single_top_level_dir_with_direct_images"

    debug = {
        "raw_photos_dir": str(raw_photos_dir),
        "resolved_photos_dir": str(chosen),
        "total_images_found_recursive": len(recursive_images),
        "direct_images_in_raw_dir": len(direct_images),
        "top_level_dirs": [str(p) for p in top_level_dirs],
        "chosen_reason": reason,
    }
    return chosen, debug


def run_individual_pipeline(photos_dir:str, output_dir:str, roster_file:str|None=None, options:dict|None=None)->dict:
    load_dotenv()
    options = options or {}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        raw_photos_path = Path(photos_dir)
        resolved_photos_path, photos_resolution_debug = resolve_pipeline_photos_dir(raw_photos_path)
        _safe_dump(out / "debug_photos_dir_resolution.json", photos_resolution_debug)
        input_images = _list_input_images(str(raw_photos_path))

        SCHOOL_NAME = options.get("school_name")
        YEAR = str(options.get("year") or "").strip()
        if len(YEAR) == 4 and YEAR.isdigit():
            YEAR = YEAR[-2:]
        SCORING = options.get("scoring", "local")
        MAX_BACKUPS = int(options.get("max_backups", 0))
        CLASS_MAPPING = options.get("class_mapping")
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not SCHOOL_NAME:
            raise ValueError("SCHOOL_NAME is required (options['school_name']).")
        if not YEAR:
            raise ValueError("YEAR is required (options['year']).")

        # Stage 1
        no_roster_mode = bool(options.get("no_roster_mode", False) or not roster_file)
        roster = None
        if roster_file:
            ext = Path(roster_file).suffix.lower()
            if ext == ".json":
                with open(roster_file, encoding="utf-8") as f:
                    roster = json.load(f)
            else:
                roster = parse_roster(roster_file, school_name=SCHOOL_NAME, openai_api_key=OPENAI_API_KEY)
            save_roster_json(roster, out / "roster.json")

        # Stage 2
        valid_numbers: set[int] = set()
        if roster:
            for cls_data in roster["classes"].values():
                for s in cls_data["students"]:
                    valid_numbers.add(s["number"])

        class_folders = _detect_class_folders(raw_photos_path) if no_roster_mode else []
        top_level_dirs = [p.name for p in class_folders] if class_folders else ([p.name for p in sorted(raw_photos_path.iterdir()) if p.is_dir()] if raw_photos_path.exists() else [])
        folder_class_context = top_level_dirs[0] if len(top_level_dirs) == 1 else None

        openai_client = None
        if OPENAI_API_KEY:
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
        elif SCORING == "openai":
            raise ValueError("SCORING = 'openai' but no OPENAI_API_KEY was provided.")

        processor = FaceGrouper(
            valid_numbers=valid_numbers,
            scoring=SCORING,
            openai_client=openai_client,
            roster=roster,
        )
        card_ocr_model = os.getenv("CARD_LABEL_OCR_MODEL", "gpt-5.4")
        card_ocr_debug: list[dict] = []
        card_ocr_used_count = 0
        card_ocr_fallback_count = 0
        if no_roster_mode and class_folders:
            class_groups = {}
            for folder in class_folders:
                sub_groups = processor.process_folder(str(folder))
                merged_students = []
                template_group = None
                for cg in sub_groups.values():
                    template_group = template_group or cg
                    for st in getattr(cg, "students", []) or []:
                        normalized = None
                        raw_label = ""
                        if getattr(st, "card_images", None):
                            ocr_result = read_card_label_openai(st.card_images[0], folder.name, openai_client if OPENAI_API_KEY else None, card_ocr_model)
                            card_ocr_used_count += 1 if OPENAI_API_KEY else 0
                            raw_label = ocr_result.get("raw_label") or ""
                            if ocr_result.get("has_card") and raw_label:
                                normalized = normalize_card_label_with_folder_context(raw_label, folder.name)
                                if normalized and normalized.get("letter_mismatch"):
                                    normalized = None
                        if not normalized:
                            card_ocr_fallback_count += 1
                            card_text = str(getattr(st, "attendance_number", "") or "")
                            normalized = normalize_card_label_with_folder_context(card_text, folder.name)
                        if normalized:
                            st.attendance_number = normalized["student_number"]
                        else:
                            st.attendance_number = None
                        card_ocr_debug.append({
                            "image_filename": Path(st.card_images[0]).name if getattr(st, "card_images", None) else "",
                            "folder_class_context": folder.name.upper(),
                            "has_card": bool(normalized),
                            "raw_label": raw_label,
                            "normalized_class_id": normalized.get("class_id") if normalized else folder.name.upper(),
                            "normalized_student_number": normalized.get("student_number") if normalized else None,
                            "confidence": ocr_result.get("confidence", 0.0) if 'ocr_result' in locals() else 0.0,
                            "backend": "openai" if OPENAI_API_KEY else "fallback_local",
                            "model": card_ocr_model if OPENAI_API_KEY else "",
                            "reason": (ocr_result.get("reason", "") if 'ocr_result' in locals() else "fallback_local"),
                        })
                        merged_students.append(st)
                if template_group is not None:
                    template_group.class_label = folder.name.upper()
                    template_group.students = merged_students
                    template_group.teacher = None
                    class_groups[folder.name.upper()] = template_group
            class_groups = {k: v for k, v in class_groups.items() if v is not None}
        else:
            class_groups = processor.process_folder(str(resolved_photos_path))
        class_groups_count = len(class_groups)
        error_queue = detect_pipeline_errors(
            class_groups=class_groups,
            roster=roster,
            photos_path=str(resolved_photos_path),
        )
        missing_tag_shot_warning_count = 0
        if no_roster_mode and class_folders:
            converted = []
            for err in error_queue:
                if err.get("error_type") == "missing_tag_shot":
                    missing_tag_shot_warning_count += 1
                    e = dict(err)
                    e["severity"] = "warning"
                    e["needs_review"] = True
                    converted.append(e)
                else:
                    converted.append(err)
            error_queue = converted
        save_error_queue(error_queue, out / "error_queue.json")
        attach_error_tags_to_groups(class_groups, error_queue)

        # Stage 3
        error_resolutions = load_error_resolutions(out / "error_resolutions.json")
        if not isinstance(error_resolutions, list):
            error_resolutions = []
        resolved_class_groups = apply_error_resolutions_to_class_groups(
            class_groups=class_groups,
            resolutions=error_resolutions,
        )
        remaining_error_queue = filter_unresolved_error_queue(
            error_queue=error_queue,
            resolutions=error_resolutions,
        )
        hard_error_queue = remaining_error_queue
        if no_roster_mode and class_folders:
            hard_error_queue = [e for e in remaining_error_queue if e.get("error_type") != "missing_tag_shot"]
        exportable_class_groups = build_exportable_class_groups(
            class_groups=resolved_class_groups,
            error_queue=hard_error_queue,
        )
        all_manifests = export_all_classes(
            class_groups=exportable_class_groups,
            roster=roster,
            output_dir=out,
            school_name=SCHOOL_NAME,
            year=YEAR,
            class_mapping=CLASS_MAPPING,
            max_backups=MAX_BACKUPS,
        )
        copied_error_files = export_error_items(hard_error_queue, out)
        if no_roster_mode:
            manifest_path = out / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for class_id, class_data in (manifest.get("classes") or {}).items():
                    unk_idx = 0
                    for entry in class_data.get("entries", []):
                        num = entry.get("number")
                        number_token = None
                        if isinstance(num, int) and num > 0:
                            number_token = f"{num:03d}"
                            entry["name"] = f"{class_id}_{number_token}"
                        else:
                            number_token = f"UNK{unk_idx:03d}"
                            entry["name"] = f"{class_id}_{number_token}"
                            entry["needs_review"] = True
                            unk_idx += 1
                        files = entry.get("files", {})
                        for tag, old_name in list(files.items()):
                            old_path = out / class_id / old_name
                            if not old_path.exists():
                                continue
                            stem = Path(old_name).stem
                            ext = old_path.suffix or ".JPG"
                            parts = stem.split("_")
                            orig = "_".join(parts[2:-2]) if len(parts) >= 5 else Path(old_name).stem
                            if not orig:
                                orig = Path(old_name).stem
                            new_name = f"{YEAR}_{SCHOOL_NAME}_{orig}_{class_id}_{number_token}_{tag}{ext.upper()}"
                            new_path = old_path.with_name(new_name)
                            if new_path != old_path:
                                old_path.rename(new_path)
                            files[tag] = new_name
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        write_error_log_json(hard_error_queue, out / "error_log.json")
        write_error_log_csv(hard_error_queue, out / "error_log.csv")
        if card_ocr_debug:
            _safe_dump(out / "card_label_ocr_debug.json", card_ocr_debug)

        out_images = [p for p in out.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        summary = {
            "status": "ok" if out_images else "warning",
            "pipeline_mode": "real_pipeline",
            "reason": None if out_images else "export_all_classes_created_no_files",
            "total_images_found": len(input_images),
            "class_groups_count": class_groups_count,
            "error_queue_count": len(error_queue),
            "remaining_error_queue_count": len(remaining_error_queue),
            "hard_error_count": len(hard_error_queue),
            "warnings_count": len([e for e in error_queue if str(e.get("severity", "")).lower() == "warning"]),
            "needs_review_count": len([e for e in error_queue if e.get("needs_review")]),
            "missing_tag_shot_warning_count": missing_tag_shot_warning_count,
            "exported_count": len(out_images),
            "exported": len(out_images),
            "openai_api_key_present": bool(OPENAI_API_KEY),
            "openai_client_created": openai_client is not None,
            "scoring": SCORING,
            "photos_path": str(resolved_photos_path),
            "output_path": str(out),
            "raw_photos_dir": str(raw_photos_path),
            "resolved_photos_dir": str(resolved_photos_path),
            "direct_images_count": len([p for p in raw_photos_path.glob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS]),
            "recursive_images_count": len(input_images),
            "all_manifests_count": len(all_manifests) if isinstance(all_manifests, dict) else 0,
            "copied_error_files": copied_error_files,
            "export_all_classes_signature": str(inspect.signature(export_all_classes)),
            "no_roster_mode": no_roster_mode,
            "roster_file_used": bool(roster_file),
            "folder_class_context": folder_class_context,
            "class_folders_detected": top_level_dirs,
            "normalized_card_labels_count": 0,
            "normalized_card_label_examples": [],
            "card_label_ocr_backend": "openai" if (no_roster_mode and class_folders and OPENAI_API_KEY) else "fallback_local",
            "card_label_ocr_model": card_ocr_model if (no_roster_mode and class_folders and OPENAI_API_KEY) else "",
            "card_label_ocr_used_count": card_ocr_used_count,
            "card_label_ocr_fallback_count": card_ocr_fallback_count,
            "card_label_ocr_examples": [d for d in card_ocr_debug[:5]],
        }
        if no_roster_mode and folder_class_context:
            normalized_count = 0
            examples: list[str] = []
            for group in class_groups.values():
                for student in getattr(group, "students", []) or []:
                    card_text = str(getattr(student, "attendance_number", "") or "")
                    if normalize_card_label_with_folder_context(card_text, folder_class_context):
                        normalized_count += 1
                        if len(examples) < 5:
                            examples.append(f"{folder_class_context}:{card_text}")
            summary["normalized_card_labels_count"] = normalized_count
            summary["normalized_card_label_examples"] = examples
        _safe_dump(out / "summary.json", summary)
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
