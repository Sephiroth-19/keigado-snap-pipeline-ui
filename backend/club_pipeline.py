from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
from openpyxl import Workbook
from PIL import Image, ExifTags

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
RANK_TOKEN = "本"
RANK_PAD = 2


@dataclass
class ClubImageResult:
    club_name: str
    path: Path
    shooting_date: str
    eyes_closed: bool
    sharpness: float
    brightness: float
    score: float
    rank: int = 0
    renamed: str = ""



def _collect_club_images(extracted_root: Path) -> dict[str, list[Path]]:
    club_images: dict[str, list[Path]] = {}
    for club_dir in sorted([p for p in extracted_root.iterdir() if p.is_dir()]):
        imgs = [
            p for p in sorted(club_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if imgs:
            club_images[club_dir.name] = imgs
    return club_images


def _shooting_date(path: Path) -> str:
    try:
        img = Image.open(path)
        exif = img.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag == "DateTimeOriginal" and value:
                    return str(value)[:10].replace(":", "")
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d")


def _eyes_closed_and_features(path: Path) -> tuple[bool, float, float]:
    image = cv2.imread(str(path))
    if image is None:
        return False, 0.0, 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    eyes = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    faces = face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))

    eyes_count = 0
    for (x, y, w, h) in faces:
        roi = gray[y : y + h, x : x + w]
        found = eyes.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=4, minSize=(12, 12))
        eyes_count += len(found)

    eyes_closed = len(faces) > 0 and eyes_count == 0
    return eyes_closed, sharpness, brightness


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

    import zipfile

    with zipfile.ZipFile(input_zip_path, "r") as zf:
        zf.extractall(extracted_root)

    club_images = _collect_club_images(extracted_root)
    results: list[ClubImageResult] = []

    for club_name, images in club_images.items():
        for img_path in images:
            eyes_closed, sharpness, brightness = _eyes_closed_and_features(img_path)
            score = sharpness + (brightness * 0.2) - (500.0 if eyes_closed else 0.0)
            results.append(
                ClubImageResult(
                    club_name=club_name,
                    path=img_path,
                    shooting_date=_shooting_date(img_path),
                    eyes_closed=eyes_closed,
                    sharpness=round(sharpness, 2),
                    brightness=round(brightness, 2),
                    score=round(score, 2),
                )
            )

    by_club: dict[str, list[ClubImageResult]] = {}
    for row in results:
        by_club.setdefault(row.club_name, []).append(row)

    for club_name, rows in by_club.items():
        rows.sort(key=lambda x: x.score, reverse=True)
        marked_club = marked_root / club_name
        clean_club = clean_root / club_name
        marked_club.mkdir(parents=True, exist_ok=True)
        clean_club.mkdir(parents=True, exist_ok=True)

        for i, row in enumerate(rows, start=1):
            row.rank = i
            rank_tag = f"{RANK_TOKEN}{i:0{RANK_PAD}d}"
            new_name = f"{club_name}_{row.shooting_date}_{row.path.stem}_{rank_tag}{row.path.suffix.lower()}"
            row.renamed = new_name
            shutil.copy2(row.path, ranked_root / new_name)
            target_folder = marked_club if row.eyes_closed else clean_club
            shutil.copy2(row.path, target_folder / row.path.name)

    excel_path = club_output / "club_result.xlsx"
    json_path = club_output / "club_result.json"

    wb = Workbook()
    ws = wb.active
    ws.title = "club_result"
    headers = [
        "club_name",
        "original_file",
        "shooting_date",
        "eyes_closed",
        "sharpness",
        "brightness",
        "score",
        "rank",
        "renamed_file",
    ]
    ws.append(headers)
    for row in sorted(results, key=lambda x: (x.club_name, x.rank)):
        ws.append(
            [
                row.club_name,
                str(row.path.name),
                row.shooting_date,
                row.eyes_closed,
                row.sharpness,
                row.brightness,
                row.score,
                row.rank,
                row.renamed,
            ]
        )
    wb.save(excel_path)

    summary = {
        "total_clubs": len(by_club),
        "total_images": len(results),
        "eyes_closed_count": sum(1 for r in results if r.eyes_closed),
    }
    payload = {
        "summary": summary,
        "items": [
            {
                "club_name": r.club_name,
                "original_file": r.path.name,
                "shooting_date": r.shooting_date,
                "eyes_closed": r.eyes_closed,
                "sharpness": r.sharpness,
                "brightness": r.brightness,
                "score": r.score,
                "rank": r.rank,
                "renamed_file": r.renamed,
            }
            for r in sorted(results, key=lambda x: (x.club_name, x.rank))
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": "ok",
        "summary": summary,
        "excel_path": str(excel_path),
        "json_path": str(json_path),
        "output_dir": str(club_output),
    }
