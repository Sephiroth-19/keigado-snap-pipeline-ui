from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

logger = logging.getLogger(__name__)

CHIN_IDX = 152
FOREHEAD_IDX = 10
NOSE_IDX = 1
LEFT_EYE_IDX = 33
RIGHT_EYE_IDX = 263

_mp_face_mesh = mp.solutions.face_mesh
_mp_selfie_seg = mp.solutions.selfie_segmentation

STUDENT_CROWN_K = 0.85
TEACHER_CROWN_K = 0.25
STUDENT_CHIN_K = 0.0
TEACHER_CHIN_K = 0.0

_HARDCODED_FALLBACK = {
    "student": {
        "frame_w_mm": 36.0,
        "frame_h_mm": 44.0,
        "guide_ratios": {"top_ratio": 0.10, "bottom_ratio": 0.86},
        "scale_clamp": {"min": 125, "max": 145},
    },
    "teacher": {
        "frame_w_mm": 46.0,
        "frame_h_mm": 54.0,
        "guide_ratios": {"top_ratio": 0.10, "bottom_ratio": 0.78},
        "scale_clamp": {"min": 124, "max": 144},
    },
}


def _copy_fallback_config() -> dict:
    return json.loads(json.dumps(_HARDCODED_FALLBACK))


def _load_image_unicode_safe(image_path: str) -> np.ndarray | None:
    try:
        buf = np.fromfile(image_path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.warning("Could not read image %s: %s", image_path, exc)
        return None


def _detect_subject_top(
    img: np.ndarray,
    bg_threshold: int = 35,
    min_run: int = 3,
    mask_threshold: float = 0.5,
) -> Optional[float]:
    """
    Return the normalized y-coordinate of the topmost detected subject row.

    Menna v12/v15 correction: use MediaPipe Selfie Segmentation to find the
    actual visible head/hair silhouette instead of relying only on face-landmark
    forehead geometry. ``bg_threshold`` is retained for API compatibility.
    """
    del bg_threshold
    h = img.shape[0]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        with _mp_selfie_seg.SelfieSegmentation(model_selection=1) as seg:
            result = seg.process(rgb)
    except Exception as exc:
        logger.warning("Selfie segmentation failed: %s", exc)
        return None

    mask = getattr(result, "segmentation_mask", None)
    if mask is None:
        logger.info("Selfie segmentation returned no mask")
        return None

    person_rows = (mask > mask_threshold).any(axis=1)
    if not person_rows.any():
        logger.info("Selfie segmentation found no foreground rows")
        return None

    run = 0
    for row in range(h):
        if person_rows[row]:
            run += 1
            if run >= min_run:
                return (row - min_run + 1) / h
        else:
            run = 0
    return None


def load_frame_config(config_path) -> dict:
    path = Path(config_path)
    if not path.is_file():
        logger.warning(
            "frame_config.json not found at %s; using built-in fallback values. "
            "Run ExportFrameDimensions.jsx for template-accurate placement.",
            path,
        )
        return _copy_fallback_config()

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    for role in ("student", "teacher"):
        fallback = _copy_fallback_config()[role]
        if role not in data or not isinstance(data[role], dict):
            data[role] = fallback
            logger.warning("frame_config.json missing %r key; using fallback", role)
            continue
        if "guide_ratios" not in data[role]:
            data[role]["guide_ratios"] = fallback["guide_ratios"]
            logger.warning("frame_config.json[%r] has no guide_ratios; using fallback", role)
        if "scale_clamp" not in data[role]:
            data[role]["scale_clamp"] = fallback["scale_clamp"]
            logger.warning("frame_config.json[%r] has no scale_clamp; using fallback", role)
        data[role].setdefault("frame_w_mm", fallback["frame_w_mm"])
        data[role].setdefault("frame_h_mm", fallback["frame_h_mm"])

    logger.info(
        "Frame config loaded from %s (source=%s, generated_at=%s)",
        path,
        data.get("source_document", "unknown"),
        data.get("generated_at", "unknown"),
    )
    for role in ("student", "teacher"):
        role_data = data[role]
        ratios = role_data.get("guide_ratios") or {}
        clamp = role_data.get("scale_clamp") or {}
        logger.info(
            "%s frame: %.2f×%.2fmm top=%.4f chin=%.4f scale=[%s-%s%%]",
            role,
            float(role_data["frame_w_mm"]),
            float(role_data["frame_h_mm"]),
            float(ratios.get("top_ratio", 0.0)),
            float(ratios.get("bottom_ratio", 0.0)),
            clamp.get("min", "?"),
            clamp.get("max", "?"),
        )
    return data


def get_target_params(role_config: dict, target_eye_dist_override: Optional[float] = None) -> dict:
    """
    Extract target parameters for the v10/v12 offset formulas.

    v10 correction: derive fallback target eye distance as
    ``face_zone_h_mm * 0.28 / frame_w_mm`` and clamp it to 0.18-0.26. The
    previous 0.55 factor over-zoomed portraits and forced max scale.
    """
    guide_ratios = role_config.get("guide_ratios")
    clamp = role_config.get("scale_clamp", {"min": 125, "max": 145})
    frame_w_mm = float(role_config["frame_w_mm"])
    frame_h_mm = float(role_config["frame_h_mm"])

    if guide_ratios:
        target_chin_y = float(guide_ratios["bottom_ratio"])
        target_top_y = float(guide_ratios["top_ratio"])
    else:
        target_chin_y = 0.86
        target_top_y = 0.10

    if target_eye_dist_override is not None:
        target_eye_dist = float(target_eye_dist_override)
    elif guide_ratios:
        face_zone_h_mm = (target_chin_y - target_top_y) * frame_h_mm
        target_eye_dist = round((face_zone_h_mm * 0.28) / frame_w_mm, 3)
        target_eye_dist = max(0.18, min(0.26, target_eye_dist))
    else:
        target_eye_dist = 0.22

    return {
        "target_chin_y": target_chin_y,
        "target_top_y": target_top_y,
        "target_eye_dist": target_eye_dist,
        "scale_clamp_min": float(clamp["min"]),
        "scale_clamp_max": float(clamp["max"]),
    }


def detect_face_landmarks(image_path: str, confidence: float = 0.5) -> tuple[Optional[dict], str]:
    if not os.path.isfile(image_path):
        logger.warning("Face detection failed: missing file: %s", image_path)
        return None, "fallback_missing_file"

    img = _load_image_unicode_safe(image_path)
    if img is None:
        logger.warning("Face detection failed: unreadable image: %s", image_path)
        return None, "fallback_unreadable_image"

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        with _mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=confidence,
        ) as face_mesh:
            results = face_mesh.process(rgb)
    except Exception as exc:
        logger.warning("Face detection failed for %s: %s", image_path, exc)
        return None, "fallback_detection_error"

    if not results.multi_face_landmarks:
        logger.warning("Face detection failed: no face detected in %s", image_path)
        return None, "fallback_no_detection"

    landmarks = results.multi_face_landmarks[0].landmark
    chin = landmarks[CHIN_IDX]
    forehead = landmarks[FOREHEAD_IDX]
    left_eye = landmarks[LEFT_EYE_IDX]
    right_eye = landmarks[RIGHT_EYE_IDX]
    nose = landmarks[NOSE_IDX]
    subject_top_y = _detect_subject_top(img)

    return {
        "chin_y": chin.y,
        "forehead_y": forehead.y,
        "subject_top_y": subject_top_y,
        "eye_center_y": (left_eye.y + right_eye.y) / 2,
        "eye_center_x": (left_eye.x + right_eye.x) / 2,
        "eye_dist": abs(right_eye.x - left_eye.x),
        "face_center_x": nose.x,
        "img_width": w,
        "img_height": h,
        "landmarks": {
            "chin": (chin.x, chin.y),
            "forehead": (forehead.x, forehead.y),
            "nose": (nose.x, nose.y),
            "left_eye": (left_eye.x, left_eye.y),
            "right_eye": (right_eye.x, right_eye.y),
        },
    }, "ok"


def _simulate_fill_proportionally(img_w, img_h, frame_w_mm, frame_h_mm):
    img_aspect = img_w / img_h
    frame_aspect = frame_w_mm / frame_h_mm
    if img_aspect > frame_aspect:
        fitted_h = frame_h_mm
        fitted_w = frame_h_mm * img_aspect
    else:
        fitted_w = frame_w_mm
        fitted_h = frame_w_mm / img_aspect
    origin_x = (frame_w_mm - fitted_w) / 2
    origin_y = (frame_h_mm - fitted_h) / 2
    return fitted_w, fitted_h, origin_x, origin_y


def compute_offsets(
    face_data: dict,
    frame_w_mm: float,
    frame_h_mm: float,
    target_face_center_x: float = 0.50,
    target_chin_y: float = 0.694,
    target_top_y: float = 0.08,
    crown_k: float = 0.25,
    scale_clamp_min: float = 100,
    scale_clamp_max: float = 200,
    target_eye_dist: float = 0.22,
    chin_k: float = 0.0,
) -> dict:
    """
    Compute JSX-compatible InDesign placement offsets.

    This mirrors AutoPlacePhotosAndNames.jsx v15's contract: InDesign first
    FILL_PROPORTIONALLY fits the image, then applies an absolute scale
    multiplier from the centered graphic, then moves by offsetX/offsetY in mm.
    The primary v12 path maps the visible subject top/crown and chin anchor to
    the template guides; the v10 eye-distance path is retained when crown data
    is unavailable.
    """
    img_w = face_data["img_width"]
    img_h = face_data["img_height"]
    fitted_w, fitted_h, origin_x, origin_y = _simulate_fill_proportionally(
        img_w,
        img_h,
        frame_w_mm,
        frame_h_mm,
    )

    face_cx = float(face_data["face_center_x"])
    chin_y = float(face_data["chin_y"])
    eye_dist = float(face_data["eye_dist"])
    forehead_y = face_data.get("forehead_y")

    gc_y = frame_h_mm / 2.0
    target_chin_mm = target_chin_y * frame_h_mm
    target_top_mm = target_top_y * frame_h_mm

    crown_y_est = None
    crown_final_y = None
    subject_top_y = None

    if forehead_y is not None and chin_y > float(forehead_y):
        forehead_y = float(forehead_y)
        face_height_norm = chin_y - forehead_y
        subject_top_y = face_data.get("subject_top_y")

        max_above_forehead = 2.0 * face_height_norm
        seg_is_sane = (
            subject_top_y is not None
            and subject_top_y < forehead_y - 0.005
            and subject_top_y > forehead_y - max_above_forehead
            and subject_top_y >= 0.0
        )

        if seg_is_sane:
            crown_y_est = float(subject_top_y)
            method = "selfie_segmentation_v15"
        else:
            crown_y_est = forehead_y - crown_k * face_height_norm
            method = "crown_k_landmark_fallback"

        # v13/v12 chin correction: anchor below lm152 only when chin_k > 0.
        chin_y_anchor = chin_y + chin_k * face_height_norm
        head_span_fill = (chin_y_anchor - crown_y_est) * fitted_h
        guide_span_mm = (target_chin_y - target_top_y) * frame_h_mm

        raw_scale = (guide_span_mm / head_span_fill * 100) if head_span_fill > 0 else 100.0
        scale_factor = float(np.clip(raw_scale, scale_clamp_min, scale_clamp_max))
        scale_multiplier = scale_factor / 100.0

        chin_raw_y = origin_y + chin_y_anchor * fitted_h
        chin_in_frame_y = gc_y + (chin_raw_y - gc_y) * scale_multiplier
        offset_y = target_chin_mm - chin_in_frame_y

        crown_raw_y = origin_y + crown_y_est * fitted_h
        crown_in_frame_y = gc_y + (crown_raw_y - gc_y) * scale_multiplier
        crown_final_y = crown_in_frame_y + offset_y
    else:
        eye_dist_mm = eye_dist * fitted_w
        desired_eye_dist_mm = target_eye_dist * frame_w_mm
        scale_factor = (desired_eye_dist_mm / eye_dist_mm * 100) if eye_dist_mm > 0 else 100.0
        scale_factor = float(np.clip(scale_factor, scale_clamp_min, scale_clamp_max))
        scale_multiplier = scale_factor / 100.0

        chin_y_anchor = chin_y
        chin_raw_y = origin_y + chin_y * fitted_h
        chin_in_frame_y = gc_y + (chin_raw_y - gc_y) * scale_multiplier
        offset_y = target_chin_mm - chin_in_frame_y
        method = "chin_anchor_eye_scale_v10_fallback"

    # v16 horizontal centering: same center-pivot model as vertical placement.
    gc_x = frame_w_mm / 2.0
    target_face_cx_mm = target_face_center_x * frame_w_mm
    face_raw_x = origin_x + face_cx * fitted_w
    face_in_frame_x = gc_x + (face_raw_x - gc_x) * scale_multiplier
    offset_x = target_face_cx_mm - face_in_frame_x

    result = {
        "offsetX": round(offset_x, 2),
        "offsetY": round(offset_y, 2),
        "scaleFactor": round(scale_factor, 1),
        "chin_y": round(chin_y, 4),
        "chin_y_anchor": round(chin_y_anchor, 4),
        "eye_dist": round(eye_dist, 4),
        "face_center_x": round(face_cx, 4),
        "method": method,
        "chin_in_frame_y_mm": round(chin_in_frame_y, 3),
        "target_chin_mm": round(target_chin_mm, 3),
        "face_in_frame_x_mm": round(face_in_frame_x, 3),
        "target_face_cx_mm": round(target_face_cx_mm, 3),
        "fitted_w_mm": round(fitted_w, 2),
        "fitted_h_mm": round(fitted_h, 2),
        "origin_y_mm": round(origin_y, 2),
    }
    if forehead_y is not None:
        result["forehead_y"] = round(forehead_y, 4)
        result["crown_y"] = round(crown_y_est, 4) if crown_y_est is not None else None
        result["subject_top_y"] = round(subject_top_y, 4) if subject_top_y is not None else None
        result["crown_final_mm"] = round(crown_final_y, 2) if crown_final_y is not None else None
        result["target_top_mm"] = round(target_top_mm, 2)
    return result


def process_manifest_offsets(
    manifest_path,
    package_root,
    target_cx: float = 0.50,
    teacher_target_cx: float = 0.50,
    target_chin_y_override: Optional[float] = None,
    target_eye_dist_override: Optional[float] = None,
    teacher_target_chin_y_override: Optional[float] = None,
    teacher_target_eye_dist_override: Optional[float] = None,
    base_offset_x: float = 0.0,
    base_offset_y: float = 5.0,
    base_scale: int = 130,
    frame_w_mm: Optional[float] = None,
    frame_h_mm: Optional[float] = None,
    teacher_frame_w_mm: Optional[float] = None,
    teacher_frame_h_mm: Optional[float] = None,
    classes: Optional[list] = None,
    confidence: float = 0.5,
    **kwargs,
):
    del kwargs
    manifest_path = Path(manifest_path)
    package_root = Path(package_root)
    frame_cfg = load_frame_config(manifest_path.parent / "frame_config.json")
    student_cfg = frame_cfg["student"]
    teacher_cfg = frame_cfg["teacher"]

    student_fw = float(frame_w_mm or student_cfg["frame_w_mm"])
    student_fh = float(frame_h_mm or student_cfg["frame_h_mm"])
    teacher_fw = float(teacher_frame_w_mm or teacher_cfg["frame_w_mm"])
    teacher_fh = float(teacher_frame_h_mm or teacher_cfg["frame_h_mm"])

    student_params = get_target_params(
        {**student_cfg, "frame_w_mm": student_fw, "frame_h_mm": student_fh},
        target_eye_dist_override,
    )
    teacher_params = get_target_params(
        {**teacher_cfg, "frame_w_mm": teacher_fw, "frame_h_mm": teacher_fh},
        teacher_target_eye_dist_override,
    )
    if target_chin_y_override is not None:
        student_params["target_chin_y"] = float(target_chin_y_override)
    if teacher_target_chin_y_override is not None:
        teacher_params["target_chin_y"] = float(teacher_target_chin_y_override)

    logger.info(
        "Placement parameters (v10/v12 JSX absolute-scale): student frame=%.2f×%.2fmm chin_y=%.4f top_y=%.4f eye_dist=%.4f scale=[%.1f-%.1f%%]",
        student_fw,
        student_fh,
        student_params["target_chin_y"],
        student_params["target_top_y"],
        student_params["target_eye_dist"],
        student_params["scale_clamp_min"],
        student_params["scale_clamp_max"],
    )
    logger.info(
        "Placement parameters (v10/v12 JSX absolute-scale): teacher frame=%.2f×%.2fmm chin_y=%.4f top_y=%.4f eye_dist=%.4f scale=[%.1f-%.1f%%]",
        teacher_fw,
        teacher_fh,
        teacher_params["target_chin_y"],
        teacher_params["target_top_y"],
        teacher_params["target_eye_dist"],
        teacher_params["scale_clamp_min"],
        teacher_params["scale_clamp_max"],
    )

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    classes_to_do = classes or list((manifest.get("classes") or {}).keys())
    all_results = {}
    updated = 0
    fallback = 0
    failed = []

    for class_id in classes_to_do:
        class_data = (manifest.get("classes") or {}).get(class_id)
        if not class_data:
            logger.warning("Class %s not in manifest; continuing", class_id)
            continue
        class_folder = package_root / class_id
        entries = class_data.get("entries", [])
        class_results = []
        logger.info("Processing class %s: %d entries", class_id, len(entries))

        for entry in entries:
            number = entry.get("number", 0)
            name = entry.get("name", "")
            files = entry.get("files", {}) or {}
            is_teacher = bool(entry.get("is_teacher", False))
            is_absent = bool(entry.get("absent", False))
            best_shot = files.get("本01") or files.get("本_01")
            label = "Teacher" if is_teacher else f"#{number:02d}" if isinstance(number, int) else f"#{number}"

            if is_teacher:
                fw, fh = teacher_fw, teacher_fh
                target_face_center_x = teacher_target_cx
                params = teacher_params
                crown_k = TEACHER_CROWN_K
                chin_k = TEACHER_CHIN_K
            else:
                fw, fh = student_fw, student_fh
                target_face_center_x = target_cx
                params = student_params
                crown_k = STUDENT_CROWN_K
                chin_k = STUDENT_CHIN_K

            def set_fallback(method: str, image_path: str | None = None, image_exists: bool = False, readable: bool = False):
                entry["face_offsets"] = {
                    "offsetX": base_offset_x,
                    "offsetY": base_offset_y,
                    "scaleFactor": base_scale,
                    "method": method,
                    "frame_w_mm": round(fw, 2),
                    "frame_h_mm": round(fh, 2),
                    "detection_backend": "fallback",
                    "image_path": image_path,
                    "image_exists": image_exists,
                    "image_read_success": readable,
                }

            if not best_shot or is_absent:
                reason = "fallback_no_image" if not best_shot else "fallback_absent"
                logger.warning("%s %s (%s): %s; applying fallback offsets", class_id, label, name, reason)
                set_fallback(reason)
                fallback += 1
                class_results.append({"number": number, "name": name, "status": reason})
                continue

            img_path = str(class_folder / best_shot)
            image_exists = os.path.isfile(img_path)
            image_read_success = _load_image_unicode_safe(img_path) is not None if image_exists else False
            face, reason = detect_face_landmarks(img_path, confidence)

            if face is None:
                logger.warning(
                    "%s %s (%s): face offset detection failed (%s); applying fallback offsetX=%.2f offsetY=%.2f scale=%s",
                    class_id,
                    label,
                    name,
                    reason,
                    base_offset_x,
                    base_offset_y,
                    base_scale,
                )
                set_fallback(reason, img_path, image_exists, image_read_success)
                fallback += 1
                failed.append({"class_id": class_id, "number": number, "name": name, "reason": reason, "image_path": img_path})
                class_results.append({"number": number, "name": name, "status": reason, "image_path": img_path})
                continue

            offsets = compute_offsets(
                face,
                fw,
                fh,
                target_face_center_x=target_face_center_x,
                target_chin_y=params["target_chin_y"],
                target_top_y=params["target_top_y"],
                crown_k=crown_k,
                scale_clamp_min=params["scale_clamp_min"],
                scale_clamp_max=params["scale_clamp_max"],
                target_eye_dist=params["target_eye_dist"],
                chin_k=chin_k,
            )
            face_offsets = {
                "offsetX": offsets["offsetX"],
                "offsetY": offsets["offsetY"],
                "scaleFactor": offsets["scaleFactor"],
                "chin_y": offsets["chin_y"],
                "eye_dist": offsets["eye_dist"],
                "method": offsets.get("method", "crown_chin_dual_anchor_v12"),
                "frame_w_mm": round(fw, 2),
                "frame_h_mm": round(fh, 2),
                "detection_backend": "mediapipe",
                "image_path": img_path,
                "image_exists": image_exists,
                "image_read_success": image_read_success,
            }
            if offsets.get("forehead_y") is not None:
                face_offsets["forehead_y"] = offsets["forehead_y"]
                face_offsets["crown_y"] = offsets.get("crown_y")
                face_offsets["subject_top_y"] = offsets.get("subject_top_y")
            entry["face_offsets"] = face_offsets
            updated += 1

            logger.info(
                "%s %s (%s) [%0.0f×%0.0fmm]: method=%s scaleFactor=%.1f offsetX=%+.2fmm offsetY=%+.2fmm chin=%.4f eye_dist=%.4f chin_frame=%.3f→%.3fmm faceX=%.3f→%.3fmm",
                class_id,
                label,
                name,
                fw,
                fh,
                offsets["method"],
                offsets["scaleFactor"],
                offsets["offsetX"],
                offsets["offsetY"],
                offsets["chin_y"],
                offsets["eye_dist"],
                offsets["chin_in_frame_y_mm"],
                offsets["target_chin_mm"],
                offsets["face_in_frame_x_mm"],
                offsets["target_face_cx_mm"],
            )
            class_results.append({
                "number": number,
                "name": name,
                "status": "ok",
                "offsets": offsets,
                "face": face,
                "image_path": img_path,
            })

        all_results[class_id] = class_results

    backup = str(manifest_path).replace(".json", "_pre_offsets.json")
    shutil.copy2(manifest_path, backup)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    logger.info("Face offsets complete: %d computed, %d fallback, %d failed", updated, fallback, len(failed))
    logger.info("Manifest backup: %s", backup)
    logger.info("Manifest saved: %s", manifest_path)
    return all_results


__all__ = [
    "compute_offsets",
    "detect_face_landmarks",
    "get_target_params",
    "load_frame_config",
    "process_manifest_offsets",
]
