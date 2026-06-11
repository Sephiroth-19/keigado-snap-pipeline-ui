from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path
from urllib.parse import quote
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from uuid import uuid4
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from backend.teacher_jobs import router as teacher_router
from openpyxl import Workbook, load_workbook
from backend.excel_labels import SNAP_SHEET_LABELS, excel_label

from backend.snap_pipeline import DEFAULT_BEST_SHOT_COUNT, SnapPipeline
from backend.preview_images import collect_snap_preview_images
from backend.club_pipeline import run_club_pipeline
from backend.individual_pipeline import (
    filter_unresolved_error_queue,
    load_error_resolutions,
    run_individual_pipeline,
    save_error_resolutions,
    zip_output_dir,
)

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



def _individual_job_output_dir(job_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id or ""):
        raise HTTPException(status_code=400, detail="Invalid job_id")
    root = (APP_STATE_DIR / "individual_jobs").resolve()
    out = (root / job_id / "output").resolve()
    try:
        out.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid job_id") from exc
    return out


def _find_individual_output_file(output_dir: Path, filename: str) -> Path:
    if not filename or Path(filename).name != filename or any(sep in filename for sep in ("/", "\\")):
        raise HTTPException(status_code=400, detail="Invalid preview filename")
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Individual output directory not found")
    output_root = output_dir.resolve()
    for candidate in output_root.rglob("*"):
        if not candidate.is_file() or candidate.name != filename:
            continue
        resolved = candidate.resolve()
        try:
            resolved.relative_to(output_root)
        except ValueError:
            continue
        return resolved
    raise HTTPException(status_code=404, detail="Preview file not found")


def _add_individual_preview_urls(manifest: dict[str, Any], job_id: str) -> dict[str, Any]:
    classes = manifest.get("classes")
    if not isinstance(classes, dict):
        return manifest
    for class_data in classes.values():
        entries = class_data.get("entries") if isinstance(class_data, dict) else None
        if not isinstance(entries, list):
            continue
        for entry in entries:
            files = entry.get("files") if isinstance(entry, dict) else None
            if not isinstance(files, dict):
                continue
            for tag, value in list(files.items()):
                if isinstance(value, str):
                    filename = value
                    files[tag] = {
                        "filename": filename,
                        "preview_url": f"/api/individual/{job_id}/preview/{quote(filename, safe='')}",
                    }
                elif isinstance(value, dict):
                    filename = value.get("filename") or value.get("name")
                    if filename:
                        value.setdefault("filename", filename)
                        value.setdefault(
                            "preview_url",
                            f"/api/individual/{job_id}/preview/{quote(str(filename), safe='')}",
                        )
    return manifest

def _pack_outputs(output_dir: Path) -> Path:
    zip_path = APP_STATE_DIR / "snap_outputs.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in output_dir.rglob("*"):
            rel = p.relative_to(output_dir)
            if "dedup_candidates" in rel.parts:
                continue
            if p.is_file():
                zf.write(p, rel)
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


def _parse_snap_best_shot_count(value: str | None) -> int:
    if value is None or value == "":
        return DEFAULT_BEST_SHOT_COUNT
    if not re.fullmatch(r"[1-9]\d*", value.strip()):
        raise HTTPException(status_code=400, detail="Best Shot Count must be a positive whole number.")
    return int(value.strip())


def _write_all_events_summary(output_root: Path, event_summaries: list[dict[str, Any]]) -> None:
    (output_root / "all_events_summary.json").write_text(
        json.dumps(event_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    wb = Workbook()
    ws = wb.active
    ws.title = SNAP_SHEET_LABELS["all_events_summary"]
    headers = [
        "event_name",
        "total_input_images",
        "total_clusters",
        "total_representative_candidates",
        "dedup_reduction_rate",
        "best_shot_count",
        "final_selected_count",
        "ng_count_after_menna",
        "other_passing_count",
    ]
    ws.append([excel_label(h) for h in headers])
    for row in event_summaries:
        ws.append([row.get(k) for k in headers])
    wb.save(output_root / "all_events_summary.xlsx")


@app.post("/api/snap/run")
async def run_snap_pipeline(
    images: list[UploadFile] | None = File(default=None),
    folder_zip: UploadFile | None = File(default=None),
    best_shot_count: str | None = Form(default=None),
) -> dict[str, Any]:
    global latest_summary, latest_zip_path

    if not images and not folder_zip:
        raise HTTPException(status_code=400, detail="Upload image files or one zip folder.")

    parsed_best_shot_count = _parse_snap_best_shot_count(best_shot_count)

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
        summary = pipeline.run(event_input_dir, event_output_dir, best_shot_count=parsed_best_shot_count)
        event_summaries.append({"event_name": event_name, **summary.__dict__})

    if not event_summaries:
        event_summaries = [
            {
                "event_name": "event_1",
                "total_input_images": 0,
                "total_clusters": 0,
                "total_representative_candidates": 0,
                "dedup_reduction_rate": 0.0,
                "best_shot_count": parsed_best_shot_count,
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
        "best_shot_count": parsed_best_shot_count,
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


@app.get("/api/snap/preview-images")
def preview_snap_images(mode: str = Query(default="final", pattern="^(final|similarity|candidates|all)$")) -> dict[str, Any]:
    images = collect_snap_preview_images(OUTPUT_DIR, mode=mode)
    return {"status": "ok", "mode": mode, "count": len(images), "images": images}


@app.get("/api/snap/preview-file")
def preview_snap_file(path: str) -> FileResponse:
    target = (OUTPUT_DIR / path).resolve()
    output_root = OUTPUT_DIR.resolve()
    try:
        target.relative_to(output_root)
    except ValueError:
        raise HTTPException(status_code=404, detail="Preview file not found.") from None
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Preview file not found.")
    return FileResponse(target)


def _excel_sort_key(path: Path) -> tuple[int, str]:
    name = path.name.lower()
    if name == "snap_pipeline_report.xlsx":
        return (0, path.as_posix())
    if name == "all_events_summary.xlsx":
        return (1, path.as_posix())
    return (2, path.as_posix())


def _cell_to_json(value: Any) -> Any:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _preview_excel_file(path: Path, row_limit: int = 100) -> dict[str, Any]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        sheets: list[dict[str, Any]] = []
        for sheet in workbook.worksheets:
            rows_iter = sheet.iter_rows(values_only=True)
            first_row = next(rows_iter, None)
            headers = [_cell_to_json(v) for v in first_row] if first_row else []
            rows: list[list[Any]] = []
            for idx, row in enumerate(rows_iter):
                if idx >= row_limit:
                    break
                rows.append([_cell_to_json(v) for v in row])
            sheets.append(
                {
                    "sheet_name": sheet.title,
                    "headers": headers,
                    "rows": rows,
                    "row_count": sheet.max_row or 0,
                    "column_count": sheet.max_column or 0,
                }
            )
        return {
            "file_name": path.name,
            "relative_path": path.relative_to(OUTPUT_DIR).as_posix(),
            "sheets": sheets,
        }
    finally:
        workbook.close()


@app.get("/api/snap/preview-excel")
def preview_snap_excel() -> dict[str, Any]:
    if not OUTPUT_DIR.exists():
        return {"status": "ok", "files": [], "message": "プレビュー可能なExcelファイルが見つかりませんでした。"}
    excel_files = sorted((p for p in OUTPUT_DIR.rglob("*.xlsx") if p.is_file()), key=_excel_sort_key)
    if not excel_files:
        return {"status": "ok", "files": [], "message": "プレビュー可能なExcelファイルが見つかりませんでした。"}
    files = [_preview_excel_file(path) for path in excel_files]
    return {"status": "ok", "files": files}


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






@app.post("/api/individual/run")
async def run_individual(
    photos_zip: UploadFile = File(...),
    roster_file: UploadFile | None = File(default=None),
    school_name: str = Form(default=""),
    year: str = Form(default=""),
    scoring: str = Form(default="local"),
) -> dict[str, Any]:
    if not photos_zip.filename or not photos_zip.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="photos_zip must be .zip")

    missing_fields: list[str] = []
    if not school_name or not school_name.strip():
        missing_fields.append("school_name")
    if not year or not year.strip():
        missing_fields.append("year")
    if missing_fields:
        return {
            "status": "error",
            "reason": "missing_required_form_field",
            "missing_fields": missing_fields,
            "received_school_name": school_name,
            "received_year": year,
            "received_scoring": scoring,
        }

    job_id = f"ind_{uuid4().hex[:12]}"
    job_root = APP_STATE_DIR / "individual_jobs" / job_id
    photos_dir = job_root / "input" / "photos"
    roster_dir = job_root / "input" / "roster"
    output_dir = job_root / "output"
    photos_dir.mkdir(parents=True, exist_ok=True)
    roster_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    photos_zip_path = _save_upload(photos_zip, job_root / "input")
    _extract_zip(photos_zip_path, photos_dir)

    roster_path = None
    if roster_file and roster_file.filename:
        roster_path = _save_upload(roster_file, roster_dir)

    print("[individual/app] received school_name=", school_name)
    print("[individual/app] received year=", year)
    print("[individual/app] received scoring=", scoring)

    summary = run_individual_pipeline(
        photos_dir=str(photos_dir),
        output_dir=str(output_dir),
        roster_file=str(roster_path) if roster_path else None,
        options={
            "school_name": school_name,
            "year": year,
            "scoring": scoring,
            "max_backups": 0,
            "class_mapping": None,
            "no_roster_mode": roster_path is None,
        },
    )

    print("[individual] Stage 6: zip output")
    output_zip = job_root / "output.zip"
    zip_output_dir(output_dir, output_zip)

    return {
        "job_id": job_id,
        **summary,
        "received_school_name": school_name,
        "received_year": year,
        "received_scoring": scoring,
        "output_zip_url": f"/api/individual/{job_id}/download",
        "manifest_url": f"/api/individual/{job_id}/result",
    }


@app.get("/api/individual/{job_id}/result")
def get_individual_result(job_id: str) -> dict[str, Any]:
    output_dir = _individual_job_output_dir(job_id)
    manifest = output_dir / "manifest.json"
    if not manifest.exists():
        raise HTTPException(status_code=404, detail="manifest not found")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return _add_individual_preview_urls(data, job_id)


@app.get("/api/individual/{job_id}/preview/{filename}")
def preview_individual_output(job_id: str, filename: str) -> FileResponse:
    output_dir = _individual_job_output_dir(job_id)
    image_path = _find_individual_output_file(output_dir, filename)
    return FileResponse(image_path, filename=image_path.name)


@app.get("/api/individual/{job_id}/download")
def download_individual_output(job_id: str) -> FileResponse:
    zip_path = APP_STATE_DIR / "individual_jobs" / job_id / "output.zip"
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail="output zip not found")
    return FileResponse(zip_path, media_type="application/zip", filename="individual_output.zip")


@app.get("/api/individual/{job_id}/errors")
def get_individual_errors(job_id: str) -> dict[str, Any]:
    out = APP_STATE_DIR / "individual_jobs" / job_id / "output"
    queue = json.loads((out / "error_queue.json").read_text(encoding="utf-8")) if (out / "error_queue.json").exists() else []
    resolutions = load_error_resolutions(out / "error_resolutions.json")
    return {"errors": filter_unresolved_error_queue(queue, resolutions)}


@app.post("/api/individual/{job_id}/resolve")
def resolve_individual_errors(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    out = APP_STATE_DIR / "individual_jobs" / job_id / "output"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "error_resolutions.json"
    existing = load_error_resolutions(path)
    existing.update(payload.get("resolutions", {}))
    save_error_resolutions(path, existing)
    return {"status": "ok", "count": len(existing)}


@app.post("/api/individual/{job_id}/reexport")
def reexport_individual(job_id: str) -> dict[str, Any]:
    root = APP_STATE_DIR / "individual_jobs" / job_id
    photos_dir = root / "input" / "photos"
    out = root / "output"
    summary = run_individual_pipeline(str(photos_dir), str(out), None, {})
    zip_output_dir(out, root / "output.zip")
    return {"job_id": job_id, **summary}


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
        allowed = {"ranked_photos", "ranked_photos_marked", "club_result.xlsx"}
        for p in club_output_dir.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(club_output_dir)
            if rel.parts and rel.parts[0] in allowed:
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
