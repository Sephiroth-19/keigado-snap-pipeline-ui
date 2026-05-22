import pytest
pytest.importorskip("mediapipe")
from pathlib import Path
import io
import zipfile

from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app


def _zip_bytes() -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w") as zf:
        img = io.BytesIO()
        Image.new("RGB", (32, 32), color=(150, 150, 150)).save(img, format="JPEG")
        zf.writestr("Basketball Club/a.jpg", img.getvalue())
    return bio.getvalue()


def test_club_run_and_download_zip_contains_club_output():
    client = TestClient(app)
    resp = client.post("/api/club/run", files={"folder_zip": ("club.zip", _zip_bytes(), "application/zip")})
    assert resp.status_code == 200
    data = resp.json()
    dl = client.get(data["output_zip_url"])
    assert dl.status_code == 200
    with zipfile.ZipFile(io.BytesIO(dl.content), "r") as zf:
        names = zf.namelist()
        assert any(n.startswith("Club_Output/ranked_photos/") for n in names)
        assert any(n.startswith("Club_Output/ranked_photos_marked/") for n in names)
        assert "Club_Output/club_result.xlsx" in names
        assert not any(n.startswith("Club_Output/clean_images/") for n in names)
        assert not any(n.startswith("Club_Output/marked_images/") for n in names)
        assert not any(n.startswith("Club_Output/ng_photos/") for n in names)
