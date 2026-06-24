import pytest
pytest.importorskip("mediapipe")
import io
import zipfile

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import delete

from backend import auth, worker
from backend.app import app
from backend.db import init_db, session_scope
from backend.models import Job


def _zip_bytes() -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        img = io.BytesIO()
        Image.new("RGB", (32, 32), color=(150, 150, 150)).save(img, format="JPEG")
        zf.writestr("Basketball Club/a.jpg", img.getvalue())
    return bio.getvalue()


def _rank_adjust_zip_bytes() -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        for filename, color in [("a.jpg", (150, 150, 150)), ("b.jpg", (170, 170, 170))]:
            img = io.BytesIO()
            Image.new("RGB", (32, 32), color=color).save(img, format="JPEG")
            zf.writestr(f"Basketball Club/{filename}", img.getvalue())
    return bio.getvalue()


def _authenticated_client() -> TestClient:
    init_db()
    auth.seed_owner()
    with session_scope() as s:
        s.execute(delete(Job))  # isolate this test's queue

    client = TestClient(app)
    # All /api/* routes now require a bearer token; log in as the seeded owner and set it
    # as the default header so every request below (run/status/download) is authenticated.
    login = client.post(
        "/api/auth/login",
        json={"username": auth.OWNER_USERNAME, "password": auth.OWNER_PASSWORD},
    )
    assert login.status_code == 200, login.text
    client.headers.update({"Authorization": "Bearer " + login.json()["access_token"]})
    return client


def test_club_run_enqueues_then_worker_produces_zip():
    client = _authenticated_client()
    resp = client.post("/api/club/run", files={"folder_zip": ("club.zip", _zip_bytes(), "application/zip")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "queued"

    # Before the worker runs, the job is queued and no zip exists yet.
    assert client.get(data["status_url"]).json()["status"] == "queued"

    # Run the worker once to drain the single queued job.
    assert worker._process_one() is True
    assert client.get(data["status_url"]).json()["status"] == "completed"

    dl = client.get(data["download_url"])
    assert dl.status_code == 200
    with zipfile.ZipFile(io.BytesIO(dl.content), "r") as zf:
        names = zf.namelist()
        assert any(n.startswith("Club_Output/ranked_photos/") for n in names)
        assert any(n.startswith("Club_Output/ranked_photos_marked/") for n in names)
        assert "Club_Output/club_result.xlsx" in names
        assert not any(n.startswith("Club_Output/clean_images/") for n in names)
        assert not any(n.startswith("Club_Output/marked_images/") for n in names)
        assert not any(n.startswith("Club_Output/ng_photos/") for n in names)


def test_club_rank_adjustment_rebuilds_download_zip():
    client = _authenticated_client()
    resp = client.post("/api/club/run", files={"folder_zip": ("club.zip", _rank_adjust_zip_bytes(), "application/zip")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "queued"
    assert worker._process_one() is True
    assert client.get(data["status_url"]).json()["status"] == "completed"

    result = client.get(data["result_url"])
    assert result.status_code == 200
    data = result.json()
    assert all(item.get("preview_url") for item in data["items"])

    first, second = data["items"]
    adjusted = client.post(
        f"/api/club/{data['job_id']}/adjust-ranks",
        json={
            "adjustments": [
                {
                    "club_name": first["club_name"],
                    "original_file": first["original_file"],
                    "final_rank": None,
                    "excluded": True,
                },
                {
                    "club_name": second["club_name"],
                    "original_file": second["original_file"],
                    "final_rank": 1,
                    "excluded": False,
                },
            ]
        },
    )
    assert adjusted.status_code == 200
    adjusted_data = adjusted.json()

    dl = client.get(adjusted_data["output_zip_url"])
    assert dl.status_code == 200
    with zipfile.ZipFile(io.BytesIO(dl.content), "r") as zf:
        names = zf.namelist()
        assert any(name.endswith("_b_本01.jpg") for name in names)
        assert not any(name.endswith("_a_本01.jpg") for name in names)
