from __future__ import annotations

import base64
import json
import mimetypes
import os
import shutil
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from openai import OpenAI

from backend.individual.real_pipeline_source import compute_offsets, get_target_params, load_frame_config

CHIN_IDX = 152
NOSE_IDX = 1
LEFT_EYE_IDX = 33
RIGHT_EYE_IDX = 263
_mp_face_mesh = mp.solutions.face_mesh


def _load_image_unicode_safe(image_path: str) -> np.ndarray | None:
    try:
        buf = np.fromfile(image_path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _detect_face_landmarks_mediapipe(image_path: str, confidence: float = 0.5) -> tuple[Optional[dict], str]:
    if not os.path.isfile(image_path):
        return None, "fallback_missing_file"
    img = _load_image_unicode_safe(image_path)
    if img is None:
        return None, "fallback_unreadable_image"
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with _mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=confidence
    ) as face_mesh:
        results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None, "fallback_no_detection"

    lm = results.multi_face_landmarks[0].landmark
    chin, left_eye, right_eye, nose = lm[CHIN_IDX], lm[LEFT_EYE_IDX], lm[RIGHT_EYE_IDX], lm[NOSE_IDX]
    return {
        "chin_y": chin.y,
        "eye_center_y": (left_eye.y + right_eye.y) / 2,
        "eye_center_x": (left_eye.x + right_eye.x) / 2,
        "eye_dist": abs(right_eye.x - left_eye.x),
        "face_center_x": nose.x,
        "img_width": w,
        "img_height": h,
    }, "ok"


def _openai_face_center(image_path: str) -> Optional[dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    model = os.getenv("FACE_OFFSET_OPENAI_MODEL", "gpt-5.4")
    client = OpenAI(api_key=api_key)
    mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    b64 = base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")
    prompt = (
        "Return strict JSON only with keys: face_detected, face_center_x, face_center_y, "
        "face_width, face_height, confidence. Values should be normalized 0..1."
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}]}],
    )
    obj = json.loads(resp.choices[0].message.content or "{}")
    if not obj.get("face_detected"):
        return None
    cx = obj.get("face_center_x")
    if not isinstance(cx, (int, float)):
        return None
    return obj


def process_manifest_offsets(manifest_path, package_root, target_cx: float = 0.50, teacher_target_cx: float = 0.50, base_offset_x: float = -3.5, base_offset_y: float = 12.0, base_scale: int = 127, frame_w_mm: Optional[float] = None, frame_h_mm: Optional[float] = None, teacher_frame_w_mm: Optional[float] = None, teacher_frame_h_mm: Optional[float] = None, classes: Optional[list] = None, confidence: float = 0.5, **kwargs):
    manifest_path = Path(manifest_path)
    package_root = Path(package_root)
    frame_cfg = load_frame_config(manifest_path.parent / "frame_config.json")
    s_cfg, t_cfg = frame_cfg["student"], frame_cfg["teacher"]
    student_fw, student_fh = frame_w_mm or s_cfg["frame_w_mm"], frame_h_mm or s_cfg["frame_h_mm"]
    teacher_fw, teacher_fh = teacher_frame_w_mm or t_cfg["frame_w_mm"], teacher_frame_h_mm or t_cfg["frame_h_mm"]
    student_params, teacher_params = get_target_params({**s_cfg, "frame_w_mm": student_fw, "frame_h_mm": student_fh}), get_target_params({**t_cfg, "frame_w_mm": teacher_fw, "frame_h_mm": teacher_fh})

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    classes_to_do = classes or list((manifest.get("classes") or {}).keys())
    for class_id in classes_to_do:
        class_data = (manifest.get("classes") or {}).get(class_id) or {}
        class_folder = package_root / class_id
        for entry in class_data.get("entries", []):
            files = entry.get("files", {}) or {}
            best_shot = files.get("本01") or files.get("本_01")
            if not best_shot or entry.get("absent", False):
                entry["face_offsets"] = {"offsetX": base_offset_x, "offsetY": base_offset_y, "scaleFactor": base_scale, "method": "fallback_no_image", "detection_backend": "fallback", "image_path": None, "image_exists": False, "image_read_success": False}
                continue
            img_path = str(class_folder / best_shot)
            face, reason = _detect_face_landmarks_mediapipe(img_path, confidence=confidence)
            image_exists = os.path.isfile(img_path)
            image_read_success = _load_image_unicode_safe(img_path) is not None if image_exists else False
            is_teacher = bool(entry.get("is_teacher", False))
            fw, fh = (teacher_fw, teacher_fh) if is_teacher else (student_fw, student_fh)
            tcx = teacher_target_cx if is_teacher else target_cx
            params = teacher_params if is_teacher else student_params
            if face is None:
                openai_obj = _openai_face_center(img_path) if reason == "fallback_no_detection" and image_read_success else None
                if openai_obj:
                    offset_x = round((tcx - float(openai_obj["face_center_x"])) * fw, 2)
                    entry["face_offsets"] = {"offsetX": offset_x, "offsetY": base_offset_y, "scaleFactor": base_scale, "method": "openai_face_center_fallback", "detection_backend": "fallback", "image_path": img_path, "image_exists": image_exists, "image_read_success": image_read_success}
                    continue
                entry["face_offsets"] = {"offsetX": base_offset_x, "offsetY": base_offset_y, "scaleFactor": base_scale, "method": reason, "detection_backend": "fallback", "image_path": img_path, "image_exists": image_exists, "image_read_success": image_read_success}
                continue
            offsets = compute_offsets(face, fw, fh, target_face_center_x=tcx, target_chin_y=params["target_chin_y"], target_eye_dist=params["target_eye_dist"], scale_clamp_min=params["scale_clamp_min"], scale_clamp_max=params["scale_clamp_max"])
            entry["face_offsets"] = {"offsetX": offsets["offsetX"], "offsetY": offsets["offsetY"], "scaleFactor": offsets["scaleFactor"], "chin_y": offsets["chin_y"], "eye_dist": offsets["eye_dist"], "method": "chin_anchor_eye_scale_v6_centre_pivot", "frame_w_mm": round(fw, 2), "frame_h_mm": round(fh, 2), "detection_backend": "mediapipe", "image_path": img_path, "image_exists": image_exists, "image_read_success": image_read_success}

    shutil.copy2(manifest_path, str(manifest_path).replace(".json", "_pre_offsets.json"))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = ["process_manifest_offsets"]
