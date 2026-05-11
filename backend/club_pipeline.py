from __future__ import annotations

import json
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openpyxl import Workbook
from PIL import ExifTags, Image, ImageDraw

try:
    from insightface.app import FaceAnalysis
except Exception:  # optional at runtime
    FaceAnalysis = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
RANK_TOKEN = "本"
RANK_PAD = 2
MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", "gpt-5.4")


@dataclass
class FaceDetail:
    face_index: int
    bbox: tuple[int, int, int, int]
    left_eye_ratio: float
    right_eye_ratio: float
    eye_closed: bool


@dataclass
class ClubImageResult:
    club_name: str
    path: Path
    shooting_date: str
    person_count: int
    eyes_closed_photo: bool
    closed_eye_face_count: int
    face_details: list[FaceDetail] = field(default_factory=list)
    formality_score: float = 0.0
    quality_score: float = 0.0
    expression_score: float = 0.0
    gesture_penalty: float = 0.0
    obscured_penalty: float = 0.0
    ng_flag: bool = False
    total_score: float = 0.0
    rank: int = 0
    renamed: str = ""


def _get_client() -> Any | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key or OpenAI is None:
        return None
    return OpenAI(api_key=key)


def _collect_club_images(extracted_root: Path) -> dict[str, list[Path]]:
    club_images: dict[str, list[Path]] = {}
    for club_dir in sorted([p for p in extracted_root.iterdir() if p.is_dir()]):
        images = [p for p in sorted(club_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if images:
            club_images[club_dir.name] = images
    return club_images


def _shooting_date(path: Path) -> str:
    try:
        img = Image.open(path)
        exif = img.getexif()
        if exif:
            for tag_id, value in exif.items():
                if ExifTags.TAGS.get(tag_id, tag_id) == "DateTimeOriginal" and value:
                    return str(value)[:10].replace(":", "")
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d")


def _eye_ratio(p_top: np.ndarray, p_bottom: np.ndarray, p_left: np.ndarray, p_right: np.ndarray) -> float:
    vertical = float(np.linalg.norm(p_top - p_bottom))
    horizontal = float(np.linalg.norm(p_left - p_right))
    return vertical / (horizontal + 1e-6)


def _analyze_faces(image_bgr: np.ndarray, face_app: Any | None) -> tuple[int, int, list[FaceDetail], np.ndarray]:
    vis = image_bgr.copy()
    details: list[FaceDetail] = []

    if face_app is not None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        faces = face_app.get(rgb)
        for i, f in enumerate(faces, start=1):
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            kps = f.kps if hasattr(f, "kps") else None
            eye_closed = False
            left_ratio = 0.0
            right_ratio = 0.0
            if kps is not None and len(kps) >= 2:
                left_eye = np.array(kps[0])
                right_eye = np.array(kps[1])
                # pseudo-EAR using box around keypoint neighborhood
                left_ratio = _eye_ratio(left_eye + [0, -2], left_eye + [0, 2], left_eye + [-4, 0], left_eye + [4, 0])
                right_ratio = _eye_ratio(right_eye + [0, -2], right_eye + [0, 2], right_eye + [-4, 0], right_eye + [4, 0])
                eye_closed = left_ratio < 0.18 and right_ratio < 0.18
            details.append(FaceDetail(i, (x1, y1, x2, y2), left_ratio, right_ratio, eye_closed))
            color = (0, 0, 255) if eye_closed else (0, 180, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"face{i}:{'closed' if eye_closed else 'open'}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        faces = face.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(50, 50))
        for i, (x, y, w, h) in enumerate(faces, start=1):
            roi = gray[y : y + h, x : x + w]
            found = eye.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=4, minSize=(10, 10))
            eye_closed = len(found) == 0
            details.append(FaceDetail(i, (x, y, x + w, y + h), 0.0, 0.0, eye_closed))
            color = (0, 0, 255) if eye_closed else (0, 180, 0)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

    closed_count = sum(1 for d in details if d.eye_closed)
    return len(details), closed_count, details, vis


def _heuristic_scores(image_bgr: np.ndarray, person_count: int, closed_count: int) -> dict[str, float | bool]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    formality = max(0.0, min(100.0, 65 + (person_count * 2) - (closed_count * 12)))
    quality = max(0.0, min(100.0, (sharp / 8.0) + (bright * 0.3)))
    expression = max(0.0, min(100.0, 75 - (closed_count * 25)))
    gesture_penalty = 0.0
    obscured_penalty = 15.0 if person_count == 0 else 0.0
    ng_flag = closed_count > max(1, person_count // 2) if person_count > 0 else True
    total = formality * 0.35 + quality * 0.35 + expression * 0.30 - gesture_penalty - obscured_penalty - (25.0 if ng_flag else 0.0)
    return {
        "formality_score": round(formality, 2),
        "quality_score": round(quality, 2),
        "expression_score": round(expression, 2),
        "gesture_penalty": round(gesture_penalty, 2),
        "obscured_penalty": round(obscured_penalty, 2),
        "ng_flag": ng_flag,
        "total_score": round(total, 2),
    }


def _gpt_scores(client: Any, image_path: Path) -> dict[str, float | bool] | None:
    try:
        prompt = (
            "Evaluate this club group photo for yearbook best-shot ranking. Return strict JSON with keys: "
            "formality_score, quality_score, expression_score, gesture_penalty, obscured_penalty, ng_flag, total_score. "
            "Scores are 0-100; penalties 0-50; total_score can be negative."
        )
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            tmp = Path(image_path).with_suffix(".tmp.jpg")
            im.save(tmp, format="JPEG", quality=92)
        with tmp.open("rb") as f:
            b = f.read()
        tmp.unlink(missing_ok=True)
        import base64

        b64 = base64.b64encode(b).decode("utf-8")
        resp = client.responses.create(
            model=MODEL_VISION,
            input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}", "detail": "low"}]}],
        )
        text = (resp.output_text or "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            return None
        obj = json.loads(text[start : end + 1])
        return {
            "formality_score": float(obj.get("formality_score", 0.0)),
            "quality_score": float(obj.get("quality_score", 0.0)),
            "expression_score": float(obj.get("expression_score", 0.0)),
            "gesture_penalty": float(obj.get("gesture_penalty", 0.0)),
            "obscured_penalty": float(obj.get("obscured_penalty", 0.0)),
            "ng_flag": bool(obj.get("ng_flag", False)),
            "total_score": float(obj.get("total_score", 0.0)),
        }
    except Exception:
        return None


def run_club_pipeline(input_zip_path: str, output_dir: str) -> dict[str, Any]:
    output_root = Path(output_dir)
    extracted_root = output_root / "extracted"
    club_output = output_root / "Club_Output"
    marked_root = club_output / "marked_images"
    clean_root = club_output / "clean_images"
    ranked_root = club_output / "bestshot_ranked"

    if output_root.exists():
        shutil.rmtree(output_root)
    extracted_root.mkdir(parents=True, exist_ok=True)
    marked_root.mkdir(parents=True, exist_ok=True)
    clean_root.mkdir(parents=True, exist_ok=True)
    ranked_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(input_zip_path, "r") as zf:
        zf.extractall(extracted_root)

    clubs = _collect_club_images(extracted_root)
    face_app = None
    if FaceAnalysis is not None:
        try:
            face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            face_app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception:
            face_app = None

    gpt_client = _get_client()
    results: list[ClubImageResult] = []

    for club_name, images in clubs.items():
        club_marked = marked_root / club_name
        club_clean = clean_root / club_name
        club_marked.mkdir(parents=True, exist_ok=True)
        club_clean.mkdir(parents=True, exist_ok=True)

        for p in images:
            img = cv2.imread(str(p))
            if img is None:
                continue
            person_count, closed_count, face_details, marked = _analyze_faces(img, face_app)
            cv2.imwrite(str(club_marked / p.name), marked)
            cv2.imwrite(str(club_clean / p.name), img)

            score = _gpt_scores(gpt_client, p) if gpt_client else None
            if score is None:
                score = _heuristic_scores(img, person_count, closed_count)
            score["total_score"] = float(score["total_score"]) - (closed_count * 7.5)

            results.append(
                ClubImageResult(
                    club_name=club_name,
                    path=p,
                    shooting_date=_shooting_date(p),
                    person_count=person_count,
                    eyes_closed_photo=closed_count > 0,
                    closed_eye_face_count=closed_count,
                    face_details=face_details,
                    formality_score=float(score["formality_score"]),
                    quality_score=float(score["quality_score"]),
                    expression_score=float(score["expression_score"]),
                    gesture_penalty=float(score["gesture_penalty"]),
                    obscured_penalty=float(score["obscured_penalty"]),
                    ng_flag=bool(score["ng_flag"]),
                    total_score=round(float(score["total_score"]), 2),
                )
            )

    by_club: dict[str, list[ClubImageResult]] = {}
    for r in results:
        by_club.setdefault(r.club_name, []).append(r)

    for club, rows in by_club.items():
        rows.sort(key=lambda x: (x.ng_flag, -x.total_score))
        for idx, row in enumerate(rows, start=1):
            row.rank = idx
            tag = f"{RANK_TOKEN}{idx:0{RANK_PAD}d}"
            row.renamed = f"{club}_{row.shooting_date}_{row.path.stem}_{tag}{row.path.suffix.lower()}"
            shutil.copy2(row.path, ranked_root / row.renamed)

    excel_path = club_output / "club_result.xlsx"
    json_path = club_output / "club_result.json"

    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.append(["club_count", "photo_count", "closed_eye_photo_count", "closed_eye_face_count", "ranked_output_count"])
    ws.append([
        len(by_club),
        len(results),
        sum(1 for r in results if r.eyes_closed_photo),
        sum(r.closed_eye_face_count for r in results),
        len(results),
    ])

    eye_ws = wb.create_sheet("Eye Closure Summary")
    eye_ws.append(["club", "file", "person_count", "closed_eye_faces", "eyes_closed_photo"])
    for r in results:
        eye_ws.append([r.club_name, r.path.name, r.person_count, r.closed_eye_face_count, r.eyes_closed_photo])

    face_ws = wb.create_sheet("Face Detail")
    face_ws.append(["club", "file", "face_index", "bbox", "left_eye_ratio", "right_eye_ratio", "eye_closed"])
    for r in results:
        for f in r.face_details:
            face_ws.append([r.club_name, r.path.name, f.face_index, str(f.bbox), f.left_eye_ratio, f.right_eye_ratio, f.eye_closed])

    rank_ws = wb.create_sheet("Best Shot Ranking")
    rank_ws.append(["club", "rank", "file", "formality", "quality", "expression", "gesture_penalty", "obscured_penalty", "ng_flag", "total_score"])
    for r in sorted(results, key=lambda x: (x.club_name, x.rank)):
        rank_ws.append([r.club_name, r.rank, r.path.name, r.formality_score, r.quality_score, r.expression_score, r.gesture_penalty, r.obscured_penalty, r.ng_flag, r.total_score])

    rename_ws = wb.create_sheet("Rename Output")
    rename_ws.append(["club", "original_file", "renamed_file", "shooting_date", "rank"])
    for r in sorted(results, key=lambda x: (x.club_name, x.rank)):
        rename_ws.append([r.club_name, r.path.name, r.renamed, r.shooting_date, r.rank])

    wb.save(excel_path)

    summary = {
        "total_clubs": len(by_club),
        "total_images": len(results),
        "eyes_closed_count": sum(1 for r in results if r.eyes_closed_photo),
        "club_count": len(by_club),
        "photo_count": len(results),
        "closed_eye_photo_count": sum(1 for r in results if r.eyes_closed_photo),
        "closed_eye_face_count": sum(r.closed_eye_face_count for r in results),
        "ranked_output_count": len(results),
    }

    json_payload = {
        "summary": summary,
        "items": [
            {
                "club_name": r.club_name,
                "original_file": r.path.name,
                "shooting_date": r.shooting_date,
                "person_count": r.person_count,
                "eyes_closed_photo": r.eyes_closed_photo,
                "closed_eye_face_count": r.closed_eye_face_count,
                "formality_score": r.formality_score,
                "quality_score": r.quality_score,
                "expression_score": r.expression_score,
                "gesture_penalty": r.gesture_penalty,
                "obscured_penalty": r.obscured_penalty,
                "ng_flag": r.ng_flag,
                "total_score": r.total_score,
                "rank": r.rank,
                "renamed_file": r.renamed,
                "face_details": [
                    {
                        "face_index": fd.face_index,
                        "bbox": [int(x) for x in fd.bbox],
                        "left_eye_ratio": float(fd.left_eye_ratio),
                        "right_eye_ratio": float(fd.right_eye_ratio),
                        "eye_closed": bool(fd.eye_closed),
                    }
                    for fd in r.face_details
                ],
            }
            for r in sorted(results, key=lambda x: (x.club_name, x.rank))
        ],
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": "completed",
        "summary": summary,
        "excel_path": str(excel_path),
        "json_path": str(json_path),
        "output_dir": str(club_output),
    }
