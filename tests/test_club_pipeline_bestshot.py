from pathlib import Path
import zipfile

from PIL import Image

from backend.club_pipeline import PHOTO_EVAL_PROMPT, run_club_pipeline


def _make_zip(tmp_path: Path) -> Path:
    root = tmp_path / "in"
    (root / "Basketball Club").mkdir(parents=True)
    (root / "Tennis Club").mkdir(parents=True)
    for i in range(2):
        Image.new("RGB", (40, 40), color=(160 + i * 10, 160, 160)).save(root / "Basketball Club" / f"IMG00{i+1}.jpg")
        Image.new("RGB", (40, 40), color=(120 + i * 10, 120, 120)).save(root / "Tennis Club" / f"IMG10{i+1}.jpg")
    z = tmp_path / "clubs.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for p in root.rglob("*.jpg"):
            zf.write(p, arcname=str(p.relative_to(root)))
    return z


def test_club_outputs_structure_and_excel(tmp_path: Path):
    z = _make_zip(tmp_path)
    out = run_club_pipeline(str(z), str(tmp_path / "out"))
    club_output = Path(out["output_dir"])
    assert (club_output / "ranked_photos").exists()
    assert (club_output / "ranked_photos_marked").exists()
    assert (club_output / "club_result.xlsx").exists()


def test_prompt_allows_thumbs_up_but_rejects_obscene_pose():
    prompt = PHOTO_EVAL_PROMPT.lower()
    assert "thumbs up is acceptable" in prompt
    assert "middle finger" in prompt
    assert "ng_reason and short_comment values must be japanese" in prompt


def test_ranked_marked_uses_marked_image_and_renamed_filename(tmp_path: Path, monkeypatch):
    import numpy as np
    import cv2
    from backend import club_pipeline

    z = _make_zip(tmp_path)

    def fake_analyze_faces(image_bgr, face_app):
        marked = image_bgr.copy()
        marked[:, :] = (0, 0, 255)
        return 1, 0, [], marked

    monkeypatch.setattr(club_pipeline, "_analyze_faces", fake_analyze_faces)

    out = run_club_pipeline(str(z), str(tmp_path / "out"))
    club_output = Path(out["output_dir"])

    ranked_files = sorted([p.relative_to(club_output / "ranked_photos") for p in (club_output / "ranked_photos").rglob("*.jpg")])
    ranked_marked_files = sorted([p.relative_to(club_output / "ranked_photos_marked") for p in (club_output / "ranked_photos_marked").rglob("*.jpg")])
    assert ranked_files == ranked_marked_files

    sample_rel = ranked_marked_files[0]
    clean_img = cv2.imread(str((club_output / "ranked_photos" / sample_rel)))
    marked_img = cv2.imread(str((club_output / "ranked_photos_marked" / sample_rel)))
    assert clean_img is not None
    assert marked_img is not None
    assert np.abs(clean_img.astype(np.int16) - marked_img.astype(np.int16)).sum() > 0
