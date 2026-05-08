# Teacher Photos Backend

This is a Web-backend callable refactor of Menna's Teacher Photos Colab pipeline.

## Files

- `teacher_pipeline_backend.py`: core Teacher Photos pipeline function
- `teacher_api.py`: FastAPI wrapper
- `teacher_backend_requirements.txt`: Python dependencies

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r teacher_backend_requirements.txt
export OPENAI_API_KEY="your_api_key"
```

## Run

```bash
uvicorn teacher_api:app --reload --host 127.0.0.1 --port 8000
```

## API

### Start pipeline

```bash
curl -X POST "http://127.0.0.1:8000/api/teacher/run" \
  -F "photos_zip=@teacher_photos.zip" \
  -F "roster_pdf=@teacher_roster.pdf"
```

### Check status

```bash
curl "http://127.0.0.1:8000/api/teacher/status/<job_id>"
```

### Get result JSON

```bash
curl "http://127.0.0.1:8000/api/teacher/result/<job_id>"
```

### Download output ZIP

```bash
curl -L "http://127.0.0.1:8000/api/teacher/download/<job_id>" -o output.zip
```

### Download Excel report

```bash
curl -L "http://127.0.0.1:8000/api/teacher/excel/<job_id>" -o results.xlsx
```

## Input structure

The current Menna pipeline is treated as a flat-folder workflow. Upload a ZIP containing card photos and portrait photos. If the ZIP has a top-level folder, the API copies all images into one flat working folder before running the pipeline.

## Outputs

- `Processed/`: matched card photos renamed with `札xx`
- `BestShot/`: matched portrait photos renamed with `本xx`
- `Unknown/`: unmatched card photos and unmatched portraits
- `results_v12.xlsx`: Excel report
- `output.zip`: full output package
