"""
FastAPI wrapper for Teacher Photos pipeline.

Usage:
  export OPENAI_API_KEY="..."
  uvicorn teacher_api:app --reload --host 127.0.0.1 --port 8000

API:
  POST /api/teacher/run        form-data: photos_zip=<zip>, roster_pdf=<pdf>
  GET  /api/teacher/status/{job_id}
  GET  /api/teacher/result/{job_id}
  GET  /api/teacher/download/{job_id}
  GET  /api/teacher/excel/{job_id}
"""

from __future__ import annotations

import json
import os
import shutil
import traceback
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from teacher_pipeline_backend import run_teacher_pipeline

APP_ROOT = Path(os.getenv("TEACHER_PIPELINE_WORKDIR", "./teacher_runs")).resolve()
APP_ROOT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

app = FastAPI(title="Keigado Teacher Photos API", version="1.0.0")

# Adjust allow_origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job state for local/demo backend.
# If you deploy this properly, replace it with DB/Redis/job queue.
JOBS: Dict[str, Dict[str, Any]] = {}


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _job_dir(job_id: str) -> Path:
    return APP_ROOT / job_id


def _safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = (extract_dir / member.filename).resolve()
            if not str(target).startswith(str(extract_dir.resolve())):
                raise ValueError(f"Unsafe zip path detected: {member.filename}")
        zf.extractall(extract_dir)


def _flatten_images(extract_dir: Path, flat_images_dir: Path) -> int:
    """
    The current Menna pipeline expects one flat image folder.
    The ZIP may contain a top-level folder, so this helper copies all images found
    inside the ZIP into a single flat folder before running the pipeline.
    This does NOT add per-teacher folder logic; it only normalizes upload format.
    """
    flat_images_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    used_names = set()

    for src in sorted(extract_dir.rglob("*")):
        if not src.is_file() or src.suffix not in IMAGE_EXTENSIONS:
            continue

        name = src.name
        stem = src.stem
        suffix = src.suffix
        i = 1
        while name in used_names or (flat_images_dir / name).exists():
            name = f"{stem}_{i}{suffix}"
            i += 1

        shutil.copy2(src, flat_images_dir / name)
        used_names.add(name)
        count += 1

    return count


async def _save_upload(upload: UploadFile, dest: Path, allowed_suffixes: set[str]) -> None:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in {s.lower() for s in allowed_suffixes}:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {upload.filename}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _process_teacher_job(job_id: str, *, model_text: str | None = None, model_vision: str | None = None) -> None:
    job = JOBS[job_id]
    root = _job_dir(job_id)

    try:
        job.update({"status": "running", "stage": "extract_upload", "updated_at": _now(), "progress": 10})

        zip_path = root / "input" / "photos.zip"
        pdf_path = root / "input" / "roster.pdf"
        extracted_dir = root / "input" / "extracted"
        images_dir = root / "input" / "images_flat"
        output_dir = root / "output"

        _safe_extract_zip(zip_path, extracted_dir)
        image_count = _flatten_images(extracted_dir, images_dir)
        if image_count == 0:
            raise FileNotFoundError("No image files found in uploaded ZIP.")

        job.update({"stage": "run_teacher_pipeline", "updated_at": _now(), "progress": 30, "image_count": image_count})

        result = run_teacher_pipeline(
            images_dir=images_dir,
            roster_pdf_path=pdf_path,
            output_dir=output_dir,
            model_text=model_text,
            model_vision=model_vision,
            face_match_threshold=0.40,
        )

        result_path = root / "result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

        job.update({
            "status": "completed",
            "stage": "completed",
            "updated_at": _now(),
            "progress": 100,
            "summary": result.get("summary", {}),
            "result_path": str(result_path),
            "excel_path": result.get("paths", {}).get("excel_report"),
            "zip_path": result.get("paths", {}).get("output_zip"),
        })

    except Exception as e:
        error_path = root / "error.log"
        error_path.write_text(traceback.format_exc(), encoding="utf-8")
        job.update({
            "status": "failed",
            "stage": "failed",
            "updated_at": _now(),
            "progress": 100,
            "error": str(e),
            "error_log": str(error_path),
        })


@app.get("/api/teacher/health")
def health():
    return {"status": "ok", "time": _now()}


@app.post("/api/teacher/run")
async def run_teacher(
    background_tasks: BackgroundTasks,
    photos_zip: UploadFile = File(...),
    roster_pdf: UploadFile = File(...),
):
    """
    Start Teacher Photos pipeline.

    photos_zip:
      ZIP containing a flat folder of teacher photos.
      If the ZIP has a top-level folder, backend flattens image files into one folder.

    roster_pdf:
      Teacher roster PDF.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set on the backend.")

    job_id = uuid.uuid4().hex[:12]
    root = _job_dir(job_id)
    input_dir = root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    await _save_upload(photos_zip, input_dir / "photos.zip", {".zip"})
    await _save_upload(roster_pdf, input_dir / "roster.pdf", {".pdf"})

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "progress": 0,
        "created_at": _now(),
        "updated_at": _now(),
        "photos_zip_filename": photos_zip.filename,
        "roster_pdf_filename": roster_pdf.filename,
    }

    background_tasks.add_task(_process_teacher_job, job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/api/teacher/status/{job_id}",
        "result_url": f"/api/teacher/result/{job_id}",
        "download_url": f"/api/teacher/download/{job_id}",
        "excel_url": f"/api/teacher/excel/{job_id}",
    }


@app.get("/api/teacher/status/{job_id}")
def get_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/teacher/result/{job_id}")
def get_result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "completed":
        return JSONResponse(status_code=202, content=job)

    result_path = Path(job["result_path"])
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    return json.loads(result_path.read_text(encoding="utf-8"))


@app.get("/api/teacher/download/{job_id}")
def download_output_zip(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed yet")

    zip_path = Path(job.get("zip_path", ""))
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="Output ZIP not found")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"teacher_output_{job_id}.zip",
    )


@app.get("/api/teacher/excel/{job_id}")
def download_excel(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed yet")

    excel_path = Path(job.get("excel_path", ""))
    if not excel_path.exists():
        raise HTTPException(status_code=404, detail="Excel report not found")

    return FileResponse(
        excel_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"teacher_results_{job_id}.xlsx",
    )
