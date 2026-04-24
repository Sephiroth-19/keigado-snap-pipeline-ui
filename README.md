# keigado-snap-pipeline-ui

Minimal local web app with existing Snap UI + FastAPI backend.

## Run locally

1. Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Start server:

```bash
./run_local.sh
```

3. Open:

- http://127.0.0.1:8000

## API

- `POST /api/snap/run`
  - Multipart form fields:
    - `images`: one or more image files, or
    - `folder_zip`: one zip file
- `GET /api/snap/result`
- `GET /api/snap/download`

The backend executes pipeline stages:
- similarity clustering
- deduplicated representative candidate creation
- best-shot selection
- Excel + output folder generation
