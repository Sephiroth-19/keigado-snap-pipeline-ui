# keigado-snap-pipeline-ui

FastAPI + static frontend app for Snap Photos and Teacher Photos pipelines.

## Run locally (macOS/Linux)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

## Run locally (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

Teacher pipeline reads `OPENAI_API_KEY` from backend `.env` only at runtime when `POST /api/teacher/run` is called.

Open: http://127.0.0.1:8000

## Environment variables

```env
OPENAI_API_KEY=
OPENAI_MODEL_TEXT=gpt-5.4
OPENAI_MODEL_VISION=gpt-5.4
```

- `.env` must not be committed.
- Snap pipeline works without `OPENAI_API_KEY`.
- Teacher pipeline requires `OPENAI_API_KEY` when calling `POST /api/teacher/run`.

## Snap APIs

- `POST /api/snap/run`
- `GET /api/snap/result`
- `GET /api/snap/download`

## Teacher APIs

- `POST /api/teacher/run` (multipart: `photos_zip` + `roster_pdf`)
- `GET /api/teacher/status/{job_id}`
- `GET /api/teacher/result/{job_id}`
- `GET /api/teacher/download/{job_id}`
- `GET /api/teacher/excel/{job_id}`

Teacher upload format:
- One ZIP containing teacher card + portrait photos.
- One PDF roster file.

## Club APIs (first backend version)

- `POST /api/club/run` (multipart: `folder_zip`)
- `GET /api/club/{job_id}/excel`
- `GET /api/club/{job_id}/download`

Club upload format:
- One ZIP where each top-level folder is one club name.
- Images can be in nested folders under each club.

Example test command:
```bash
curl -X POST "http://127.0.0.1:8000/api/club/run" \
  -F "folder_zip=@/path/to/club_photos.zip"
```
