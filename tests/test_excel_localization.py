from pathlib import Path
from datetime import datetime

from openpyxl import load_workbook

from backend.club_pipeline import run_club_pipeline
from backend.snap_pipeline import SnapPipeline, PipelineResult, ImageRecord
import numpy as np
from PIL import Image
import zipfile


def test_snap_excel_labels_japanese(tmp_path: Path):
    pipeline = SnapPipeline.__new__(SnapPipeline)

    img_path = tmp_path / "a.jpg"
    Image.new("RGB", (16, 16), color=(120, 120, 120)).save(img_path)
    rec = ImageRecord(
        path=img_path,
        capture_time=datetime(2026, 1, 2, 3, 4, 5),
        sequence_tail=1,
        embedding=np.zeros(3, dtype=np.float32),
        focus_score=1.0,
        brightness_score=1.0,
    )
    summary = PipelineResult(1, 1, 1, 0.0, 1, 1, 0, 0)

    pipeline._write_outputs(tmp_path / "out", [[rec]], [rec], [rec], [], [], summary)

    wb = load_workbook(tmp_path / "out" / "snap_pipeline_report.xlsx")
    assert wb.sheetnames == ["サマリー", "類似グループ", "選定結果"]
    assert [c.value for c in wb["サマリー"][1]] == ["項目", "値"]
    assert wb["選定結果"][2][0].value == "ベストショット"


def test_club_excel_labels_japanese(tmp_path: Path):
    root = tmp_path / "in"
    club = root / "soccer"
    club.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(180, 180, 180)).save(club / "x.jpg")

    z = tmp_path / "club.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(club / "x.jpg", arcname="soccer/x.jpg")

    out = run_club_pipeline(str(z), str(tmp_path / "out"))
    wb = load_workbook(out["excel_path"])

    assert wb.sheetnames == ["サマリー", "目つぶり確認サマリー", "顔検出詳細", "ベストショット順位", "リネーム結果"]
    assert [c.value for c in wb["リネーム結果"][1]] == ["部活動名", "元ファイル名", "リネーム後ファイル名", "撮影日", "順位"]
