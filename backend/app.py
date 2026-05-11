from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from uuid import uuid4
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from backend.teacher_jobs import router as teacher_router
from openpyxl import Workbook

from backend.snap_pipeline import SnapPipeline
from backend.club_pipeline import run_club_pipeline

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
APP_STATE_DIR = ROOT / "runtime"
INPUT_DIR = APP_STATE_DIR / "input"
OUTPUT_DIR = APP_STATE_DIR / "output"

load_dotenv()
app = FastAPI(title="Snap Pipeline Local App")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = SnapPipeline()
latest_summary: dict[str, Any] | None = None
latest_zip_path: Path | None = None


def _reset_dirs() -> None:
    if APP_STATE_DIR.exists():
        shutil.rmtree(APP_STATE_DIR)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_upload(file: UploadFile, dst: Path) -> Path:
    target = dst / file.filename
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return target


def _extract_zip(zip_path: Path, dst: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dst)


def _pack_outputs(output_dir: Path) -> Path:
    zip_path = APP_STATE_DIR / "snap_outputs.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in output_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(output_dir))
    return zip_path


def _has_images(path: Path) -> bool:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
    return any(p.suffix.lower() in exts for p in path.rglob("*") if p.is_file())


def _safe_event_name(name: str) -> str:
    sanitized = re.sub(r"[^\w\-ぁ-んァ-ヶ一-龠々ー]+", "_", name.strip())
    return sanitized.strip("_") or "event"


def _discover_event_dirs(input_root: Path) -> list[tuple[str, Path]]:
    top_dirs = [p for p in sorted(input_root.iterdir()) if p.is_dir()]
    event_dirs: list[tuple[str, Path]] = []

    for d in top_dirs:
        if _has_images(d):
            event_dirs.append((_safe_event_name(d.name), d))

    if event_dirs:
        return event_dirs

    if _has_images(input_root):
        return [("event_1", input_root)]

    return []


def _write_all_events_summary(output_root: Path, event_summaries: list[dict[str, Any]]) -> None:
    (output_root / "all_events_summary.json").write_text(
        json.dumps(event_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    wb = Workbook()
    ws = wb.active
    ws.title = "all_events_summary"
    headers = [
        "event_name",
        "total_input_images",
        "total_clusters",
        "total_representative_candidates",
        "dedup_reduction_rate",
        "final_selected_count",
        "ng_count_after_menna",
        "other_passing_count",
    ]
    ws.append(headers)
    for row in event_summaries:
        ws.append([row.get(k) for k in headers])
    wb.save(output_root / "all_events_summary.xlsx")


@app.post("/api/snap/run")
async def run_snap_pipeline(
    images: list[UploadFile] | None = File(default=None),
    folder_zip: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    global latest_summary, latest_zip_path

    if not images and not folder_zip:
        raise HTTPException(status_code=400, detail="Upload image files or one zip folder.")

    _reset_dirs()

    if images:
        for image in images:
            _save_upload(image, INPUT_DIR)

    if folder_zip:
        zip_path = _save_upload(folder_zip, INPUT_DIR)
        if zip_path.suffix.lower() != ".zip":
            raise HTTPException(status_code=400, detail="folder_zip must be a .zip file.")
        _extract_zip(zip_path, INPUT_DIR)
        zip_path.unlink(missing_ok=True)

    events = _discover_event_dirs(INPUT_DIR)
    event_summaries: list[dict[str, Any]] = []

    for event_name, event_input_dir in events:
        event_output_dir = OUTPUT_DIR / event_name
        summary = pipeline.run(event_input_dir, event_output_dir)
        event_summaries.append({"event_name": event_name, **summary.__dict__})

    if not event_summaries:
        event_summaries = [
            {
                "event_name": "event_1",
                "total_input_images": 0,
                "total_clusters": 0,
                "total_representative_candidates": 0,
                "dedup_reduction_rate": 0.0,
                "final_selected_count": 0,
                "ng_count_after_menna": 0,
                "other_passing_count": 0,
            }
        ]

    _write_all_events_summary(OUTPUT_DIR, event_summaries)

    total_summary = {
        "event_name": "all_events",
        "total_input_images": sum(e["total_input_images"] for e in event_summaries),
        "total_clusters": sum(e["total_clusters"] for e in event_summaries),
        "total_representative_candidates": sum(e["total_representative_candidates"] for e in event_summaries),
        "dedup_reduction_rate": 0.0,
        "final_selected_count": sum(e["final_selected_count"] for e in event_summaries),
        "ng_count_after_menna": sum(e["ng_count_after_menna"] for e in event_summaries),
        "other_passing_count": sum(e["other_passing_count"] for e in event_summaries),
    }
    if total_summary["total_input_images"] > 0:
        total_summary["dedup_reduction_rate"] = round(
            (1 - (total_summary["total_representative_candidates"] / total_summary["total_input_images"])) * 100, 2
        )

    latest_summary = {**total_summary, "event_summaries": event_summaries}
    latest_zip_path = _pack_outputs(OUTPUT_DIR)
    return {"status": "ok", "summary": latest_summary}


@app.get("/api/snap/result")
def get_snap_result() -> dict[str, Any]:
    if latest_summary is None:
        raise HTTPException(status_code=404, detail="No run result yet. Run /api/snap/run first.")
    return latest_summary


@app.get("/api/snap/download")
def download_outputs() -> FileResponse:
    if latest_zip_path is None or not latest_zip_path.exists():
        raise HTTPException(status_code=404, detail="No output zip available. Run /api/snap/run first.")
    return FileResponse(latest_zip_path, media_type="application/zip", filename="snap_outputs.zip")




@app.post("/api/club/run")
async def run_club(folder_zip: UploadFile = File(...)) -> dict[str, Any]:
    if not folder_zip.filename or not folder_zip.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Please upload one .zip file.")

    job_id = f"club_{uuid4().hex[:12]}"
    job_root = APP_STATE_DIR / "club_jobs" / job_id
    input_dir = job_root / "input"
    output_dir = job_root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)

    zip_path = _save_upload(folder_zip, input_dir)
    result = run_club_pipeline(str(zip_path), str(output_dir))

    output_zip = job_root / "club_output.zip"
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        club_output_dir = Path(result["output_dir"])
        for p in club_output_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(club_output_dir.parent))

    return {
        "job_id": job_id,
        "status": result["status"],
        "summary": result["summary"],
        "excel_url": f"/api/club/{job_id}/excel",
        "output_zip_url": f"/api/club/{job_id}/download",
    }


@app.get("/api/club/{job_id}/excel")
def download_club_excel(job_id: str) -> FileResponse:
    excel_path = APP_STATE_DIR / "club_jobs" / job_id / "output" / "Club_Output" / "club_result.xlsx"
    if not excel_path.exists():
        raise HTTPException(status_code=404, detail="Club Excel not found.")
    return FileResponse(excel_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="club_result.xlsx")


@app.get("/api/club/{job_id}/download")
def download_club_output(job_id: str) -> FileResponse:
    zip_path = APP_STATE_DIR / "club_jobs" / job_id / "club_output.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Club output zip not found.")
    return FileResponse(zip_path, media_type="application/zip", filename="club_output.zip")

app.include_router(teacher_router)

if (FRONTEND_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
