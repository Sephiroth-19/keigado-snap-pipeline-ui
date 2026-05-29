from __future__ import annotations
import json, os, shutil, traceback, uuid, zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv
from backend.preview_images import image_media_type, list_preview_images, safe_resolve_preview_path

router = APIRouter(prefix='/api/teacher', tags=['teacher'])
JOBS: dict[str, dict[str, Any]] = {}
ROOT = Path(__file__).resolve().parent.parent
TEACHER_ROOT = ROOT / 'runtime' / 'teacher' / 'jobs'


def _now() -> str:
    return datetime.utcnow().isoformat(timespec='seconds') + 'Z'

def _safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for m in zf.infolist():
            target = (extract_dir / m.filename).resolve()
            if not str(target).startswith(str(extract_dir.resolve())):
                raise ValueError(f'Unsafe zip member: {m.filename}')
        zf.extractall(extract_dir)

def _flatten_images(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    exts = {'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff','.heic','.heif'}
    c=0
    for p in sorted(src.rglob('*')):
        if p.is_file() and p.suffix.lower() in exts:
            out = dst / p.name
            i=1
            while out.exists():
                out = dst / f'{p.stem}_{i}{p.suffix.lower()}'; i+=1
            shutil.copy2(p, out); c+=1
    return c

async def _save_upload(upload: UploadFile, dest: Path, suffix: str) -> None:
    if not (upload.filename or '').lower().endswith(suffix):
        raise HTTPException(status_code=400, detail=f'{dest.name} must be a {suffix} file.')
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('wb') as f:
        while chunk := await upload.read(1024*1024):
            f.write(chunk)

def _run_job(job_id: str) -> None:
    job = JOBS[job_id]
    root = TEACHER_ROOT / job_id
    try:
        job.update(status='running', updated_at=_now())
        images_zip = root / 'input' / 'photos.zip'
        pdf = root / 'input' / 'roster.pdf'
        extracted = root / 'input' / 'extracted'
        images = root / 'input' / 'photos'
        output = root / 'output'
        _safe_extract_zip(images_zip, extracted)
        count = _flatten_images(extracted, images)
        if count == 0:
            raise RuntimeError('No image files found in uploaded ZIP.')
        from backend.teacher_pipeline_backend import run_teacher_pipeline
        result = run_teacher_pipeline(images_dir=images, roster_pdf_path=pdf, output_dir=output)
        (root / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        job.update(status='completed', updated_at=_now(), image_count=count, result=result)
    except Exception as e:
        (root/'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        job.update(status='failed', updated_at=_now(), error=str(e))

@router.post('/run')
async def run_teacher(background_tasks: BackgroundTasks, photos_zip: UploadFile = File(...), roster_pdf: UploadFile = File(...)):
    load_dotenv()
    if not os.getenv('OPENAI_API_KEY'):
        raise HTTPException(status_code=500, detail='OPENAI_API_KEY is not set. Please create a local .env file or set the environment variable.')
    job_id = uuid.uuid4().hex
    root = TEACHER_ROOT / job_id
    await _save_upload(photos_zip, root / 'input' / 'photos.zip', '.zip')
    await _save_upload(roster_pdf, root / 'input' / 'roster.pdf', '.pdf')
    JOBS[job_id] = {'job_id': job_id, 'status':'queued', 'created_at':_now(), 'updated_at':_now()}
    background_tasks.add_task(_run_job, job_id)
    return {k: v for k,v in {
        'job_id':job_id,'status_url':f'/api/teacher/status/{job_id}','result_url':f'/api/teacher/result/{job_id}',
        'download_url':f'/api/teacher/download/{job_id}','excel_url':f'/api/teacher/excel/{job_id}'}.items()}

@router.get('/status/{job_id}')
def status(job_id: str):
    if job_id not in JOBS: raise HTTPException(status_code=404, detail='Job not found')
    return JOBS[job_id]

@router.get('/result/{job_id}')
def result(job_id: str):
    job = JOBS.get(job_id)
    if not job: raise HTTPException(status_code=404, detail='Job not found')
    if job['status'] != 'completed':
        return JSONResponse(status_code=202, content={'job_id':job_id,'status':job['status'],'message':job.get('error','Job not completed')})
    return job['result']


@router.get('/{job_id}/preview-images')
def preview_images(job_id: str):
    output_dir = TEACHER_ROOT / job_id / 'output'
    images = list_preview_images(output_dir, f'/api/teacher/{job_id}/preview-image', 'Teacher Photo')
    return {'job_id': job_id, 'count': len(images), 'images': images}

@router.get('/{job_id}/preview-image')
def preview_image(job_id: str, path: str):
    output_dir = TEACHER_ROOT / job_id / 'output'
    image_path = safe_resolve_preview_path(output_dir, path)
    return FileResponse(image_path, media_type=image_media_type(image_path), filename=image_path.name)


@router.get('/download/{job_id}')
def download(job_id: str):
    p = TEACHER_ROOT / job_id / 'output' / 'output.zip'
    if not p.exists(): raise HTTPException(status_code=404, detail='Output zip not available')
    return FileResponse(p, media_type='application/zip', filename=f'teacher_{job_id}.zip')

@router.get('/excel/{job_id}')
def excel(job_id: str):
    p = TEACHER_ROOT / job_id / 'output' / 'results_v12.xlsx'
    if not p.exists(): raise HTTPException(status_code=404, detail='Excel file not available')
    return FileResponse(p, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename=f'teacher_{job_id}.xlsx')
