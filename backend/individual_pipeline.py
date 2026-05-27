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
from backend.individual.face_offset_calculator import process_manifest_offsets



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


def _list_images_in_folder(folder: Path) -> list[Path]:
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def _safe_dump(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _normalize_card_label_text(card_label: str) -> str:
    trans = str.maketrans("０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ－ー", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ--")
    return str(card_label or "").translate(trans).strip().upper().replace(" ", "")


def normalize_card_label_without_folder_context(card_label: str, default_grade: str = "3"):
    label = _normalize_card_label_text(card_label)
    if not label:
        return None
    m = re.match(r"^([A-Z])[-]?(\d{1,3})$", label)
    if not m:
        return None
    class_letter, digits = m.group(1), m.group(2)
    student_number = int(digits)
    class_id = f"{default_grade}{class_letter}"
    return {
        "class_id": class_id,
        "student_number": student_number,
        "normalized_label": f"{class_id}_{student_number:03d}",
        "letter_mismatch": False,
    }


def normalize_card_label_with_folder_context(card_label: str, folder_class_context: str | None):
    label = _normalize_card_label_text(card_label)
    if not label or not folder_class_context:
        return None
    context = str(folder_class_context).strip().upper()
    m_ctx = re.match(r"^(\d+)\s*([A-Z])$", context)
    if not m_ctx:
        return None
    klass = m_ctx.group(2)
    m = re.match(r"^([A-Z])[-]?(\d{1,3})$", label)
    if m:
        card_letter, digits = m.group(1), m.group(2)
        return {
            "class_id": context,
            "student_number": int(digits),
            "normalized_label": f"{context}_{int(digits):03d}",
            "letter_mismatch": card_letter != klass,
        }
    return None


def read_card_label_openai(image_path: str, folder_class_context: str | None, client: OpenAI | None, model: str):
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
        context_instruction = (
            f"Folder class context is {folder_class_context}. "
            if folder_class_context
            else "No folder class context is available; read the class letter directly from the card label. "
        )
        prompt = (
            "Read ONLY the physical student card held in this photo. "
            + context_instruction
            + "Return strict JSON only with keys: has_card, raw_label, class_letter, student_number, confidence, reason. "
            + "Accept patterns like A1, A2, A11, D15, H28, H32. If unclear/no card, has_card=false."
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


def _build_openai_card_index_for_folder(
    folder: Path,
    folder_class_context: str,
    client: OpenAI | None,
    model: str,
):
    results_by_path: dict[str, dict] = {}
    debug_rows: list[dict] = []
    images = _list_images_in_folder(folder)
    for p in images:
        o = read_card_label_openai(str(p), folder_class_context, client, model)
        normalized = normalize_card_label_with_folder_context(o.get("raw_label", ""), folder_class_context) if o.get("has_card") else None
        accepted = bool(o.get("has_card") and normalized and not normalized.get("letter_mismatch"))
        reject_reason = ""
        if o.get("has_card") and not accepted:
            reject_reason = "label_letter_mismatch" if normalized and normalized.get("letter_mismatch") else "invalid_or_uncertain_label"
        results_by_path[str(p.resolve())] = {
            **o,
            "accepted": accepted,
            "normalized_class_id": normalized.get("class_id") if normalized else None,
            "normalized_student_number": normalized.get("student_number") if normalized else None,
            "reject_reason": reject_reason,
        }
        debug_rows.append({
            "image_filename": p.name,
            "image_path": str(p.resolve()),
            "folder_class_context": folder_class_context.upper(),
            "openai_called": True,
            "has_card": bool(o.get("has_card")),
            "raw_label": o.get("raw_label", ""),
            "normalized_class_id": normalized.get("class_id") if normalized else folder_class_context.upper(),
            "normalized_student_number": normalized.get("student_number") if normalized else None,
            "confidence": float(o.get("confidence") or 0.0),
            "accepted": accepted,
            "reject_reason": reject_reason,
            "backend": "openai" if client else "fallback_local",
            "model": model if client else "",
            "reason": o.get("reason", ""),
        })
    return results_by_path, debug_rows


def _build_openai_card_index_for_mixed_folder(
    folder: Path,
    client: OpenAI | None,
    model: str,
    default_grade: str = "3",
):
    results_by_path: dict[str, dict] = {}
    debug_rows: list[dict] = []
    images = _list_images_in_folder(folder)
    for p in images:
        o = read_card_label_openai(str(p), None, client, model)
        normalized = normalize_card_label_without_folder_context(o.get("raw_label", ""), default_grade) if o.get("has_card") else None
        accepted = bool(o.get("has_card") and normalized)
        reject_reason = ""
        if o.get("has_card") and not accepted:
            reject_reason = "invalid_or_uncertain_label"
        results_by_path[str(p.resolve())] = {
            **o,
            "accepted": accepted,
            "normalized_class_id": normalized.get("class_id") if normalized else None,
            "normalized_student_number": normalized.get("student_number") if normalized else None,
            "reject_reason": reject_reason,
        }
        debug_rows.append({
            "image_filename": p.name,
            "image_path": str(p.resolve()),
            "folder_class_context": None,
            "mixed_folder_mode": True,
            "openai_called": bool(client),
            "has_card": bool(o.get("has_card")),
            "raw_label": o.get("raw_label", ""),
            "normalized_class_id": normalized.get("class_id") if normalized else None,
            "normalized_student_number": normalized.get("student_number") if normalized else None,
            "confidence": float(o.get("confidence") or 0.0),
            "accepted": accepted,
            "reject_reason": reject_reason,
            "backend": "openai" if client else "fallback_local",
            "model": model if client else "",
            "reason": o.get("reason", ""),
        })
    return results_by_path, debug_rows


def _regroup_students_by_card_index(class_groups, card_index: dict[str, dict]):
    regrouped = {}
    templates = {}
    for cg in class_groups.values():
        for st in getattr(cg, "students", []) or []:
            group_paths = [
                str(Path(p).resolve())
                for p in ((getattr(st, "card_images", None) or []) + [
                    sp.path for sp in (getattr(st, "portraits", None) or []) if getattr(sp, "path", None)
                ])
            ]
            accepted_hits = [card_index[p] for p in group_paths if p in card_index and card_index[p].get("accepted")]
            if accepted_hits:
                hit = accepted_hits[0]
                class_id = str(hit.get("normalized_class_id") or "UNK").upper()
                st.attendance_number = hit.get("normalized_student_number")
            else:
                class_id = "UNK"
                st.attendance_number = None
            if class_id not in regrouped:
                template = templates.get(class_id) or cg
                templates[class_id] = template
                new_group = type(template)()
                new_group.class_label = class_id
                new_group.students = []
                new_group.teacher = None
                regrouped[class_id] = new_group
            regrouped[class_id].students.append(st)
    return regrouped


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
        enable_face_offsets = bool(options.get("enable_face_offsets", False))
        frame_config_file_opt = options.get("frame_config_file")
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
        resolved_images = _list_images_in_folder(resolved_photos_path)
        mixed_folder_no_roster_mode = bool(no_roster_mode and not class_folders and resolved_images)
        if no_roster_mode and class_folders:
            class_groups = {}
            for folder in class_folders:
                folder_card_index, folder_debug_rows = _build_openai_card_index_for_folder(
                    folder=folder,
                    folder_class_context=folder.name,
                    client=openai_client if OPENAI_API_KEY else None,
                    model=card_ocr_model,
                )
                card_ocr_debug.extend(folder_debug_rows)
                card_ocr_used_count += len(folder_debug_rows) if OPENAI_API_KEY else 0
                sub_groups = processor.process_folder(str(folder))
                merged_students = []
                template_group = None
                for cg in sub_groups.values():
                    template_group = template_group or cg
                    for st in getattr(cg, "students", []) or []:
                        normalized = None
                        raw_label = ""
                        group_paths = [str(Path(p).resolve()) for p in ((getattr(st, "card_images", None) or []) + [sp.path for sp in (getattr(st, "portraits", None) or []) if getattr(sp, "path", None)])]
                        accepted_hits = [folder_card_index[p] for p in group_paths if p in folder_card_index and folder_card_index[p].get("accepted")]
                        if accepted_hits:
                            hit = accepted_hits[0]
                            raw_label = hit.get("raw_label") or ""
                            normalized = {
                                "class_id": hit.get("normalized_class_id"),
                                "student_number": hit.get("normalized_student_number"),
                            }
                        if not normalized:
                            card_ocr_fallback_count += 1
                            card_text = str(getattr(st, "attendance_number", "") or "")
                            normalized = normalize_card_label_with_folder_context(card_text, folder.name)
                        if normalized:
                            st.attendance_number = normalized["student_number"]
                        else:
                            st.attendance_number = None
                        merged_students.append(st)
                if template_group is not None:
                    template_group.class_label = folder.name.upper()
                    template_group.students = merged_students
                    template_group.teacher = None
                    class_groups[folder.name.upper()] = template_group
            class_groups = {k: v for k, v in class_groups.items() if v is not None}
        elif mixed_folder_no_roster_mode:
            mixed_card_index, mixed_debug_rows = _build_openai_card_index_for_mixed_folder(
                folder=resolved_photos_path,
                client=openai_client if OPENAI_API_KEY else None,
                model=card_ocr_model,
            )
            card_ocr_debug.extend(mixed_debug_rows)
            card_ocr_used_count += len(mixed_debug_rows) if OPENAI_API_KEY else 0
            class_groups = processor.process_folder(str(resolved_photos_path))
            card_ocr_fallback_count += len([d for d in mixed_debug_rows if not d.get("accepted")])
            class_groups = _regroup_students_by_card_index(class_groups, mixed_card_index)
        else:
            class_groups = processor.process_folder(str(resolved_photos_path))
        class_groups_count = len(class_groups)
        error_queue = detect_pipeline_errors(
            class_groups=class_groups,
            roster=roster,
            photos_path=str(resolved_photos_path),
        )
        missing_tag_shot_warning_count = 0
        if no_roster_mode:
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
        if no_roster_mode:
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
        manifest_path = out / "manifest.json"
        frame_config_path = Path(frame_config_file_opt) if frame_config_file_opt else (out / "frame_config.json")
        frame_config_present = frame_config_path.exists()
        face_offset_status = "disabled"
        face_offset_error = None
        if enable_face_offsets:
            if not frame_config_present:
                face_offset_status = "skipped_no_frame_config"
            elif not manifest_path.exists():
                face_offset_status = "error"
                face_offset_error = "manifest.json not found"
            else:
                try:
                    process_manifest_offsets(manifest_path=manifest_path, package_root=out)
                    face_offset_status = "ok"
                except Exception as e:
                    face_offset_status = "error"
                    face_offset_error = str(e)
        if no_roster_mode:
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
            "mixed_folder_no_roster_mode": mixed_folder_no_roster_mode,
            "normalized_card_labels_count": len([d for d in card_ocr_debug if d.get("accepted")]),
            "normalized_card_label_examples": [
                f"{d.get('normalized_class_id')}:{int(d.get('normalized_student_number')):03d}"
                for d in card_ocr_debug[:5]
                if d.get("accepted") and d.get("normalized_class_id") and d.get("normalized_student_number") is not None
            ],
            "card_label_ocr_backend": "openai" if card_ocr_used_count > 0 else "fallback_local",
            "card_label_ocr_model": card_ocr_model if card_ocr_used_count > 0 else "",
            "card_label_ocr_used_count": card_ocr_used_count,
            "card_label_ocr_fallback_count": card_ocr_fallback_count,
            "card_label_ocr_total_images": len(card_ocr_debug),
            "card_label_ocr_debug_records_count": len(card_ocr_debug),
            "card_label_ocr_accepted_count": len([d for d in card_ocr_debug if d.get("accepted")]),
            "card_label_ocr_rejected_count": len([d for d in card_ocr_debug if not d.get("accepted")]),
            "card_label_ocr_examples": [d for d in card_ocr_debug[:5]],
            "face_offset_status": face_offset_status,
            "face_offset_error": face_offset_error,
            "frame_config_present": frame_config_present,
        }
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
