from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.snap_pipeline import SnapPipeline

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
APP_STATE_DIR = ROOT / "runtime"
INPUT_DIR = APP_STATE_DIR / "input"
OUTPUT_DIR = APP_STATE_DIR / "output"

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

    summary = pipeline.run(INPUT_DIR, OUTPUT_DIR)
    latest_summary = summary.__dict__
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


if (FRONTEND_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
