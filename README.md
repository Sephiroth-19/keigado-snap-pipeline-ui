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


## Multi-event ZIP testing

For multi-event processing, create **one ZIP** containing multiple top-level event folders, for example:

```text
multi_event_test.zip
  event_01/
    img001.jpg
    img002.jpg
  event_02/
    img101.jpg
    img102.jpg
```

Example creation command:

```bash
mkdir -p /tmp/multi_event_test/event_01 /tmp/multi_event_test/event_02
cp /path/to/event_01_images/* /tmp/multi_event_test/event_01/
cp /path/to/event_02_images/* /tmp/multi_event_test/event_02/
(cd /tmp/multi_event_test && zip -r multi_event_test.zip event_01 event_02)
```

Upload `multi_event_test.zip` in the Snap UI. The backend processes each top-level folder separately and preserves per-event output structure in the download ZIP.
