from pathlib import Path
from datetime import datetime

import pytest
from openpyxl import load_workbook

from backend.club_pipeline import run_club_pipeline
from backend.excel_labels import TEACHER_SHEET_LABELS, excel_label, translate_display_value
import numpy as np
from PIL import Image
import zipfile


def test_snap_excel_labels_japanese(tmp_path: Path):
    pytest.importorskip("torch")
    from backend.snap_pipeline import SnapPipeline, PipelineResult, ImageRecord
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

    assert wb.sheetnames == ["サマリー", "目つぶり確認サマリー", "顔検出詳細", "ベストショット順位", "リネーム結果", "NG写真・要確認"]
    assert [c.value for c in wb["リネーム結果"][1]] == ["部活動名", "元ファイル名", "リネーム後ファイル名", "撮影日", "順位"]


def test_individual_error_log_csv_localized(tmp_path: Path):
    pytest.importorskip("mediapipe")
    from backend.individual.real_pipeline_source import write_error_log_csv
    out_csv = tmp_path / "error_log.csv"
    write_error_log_csv(
        [
            {
                "error_type": "no_card_detected",
                "severity": "warning",
                "detection_unit": "layer1",
                "class_id": "3A",
                "group_key": "g1",
                "group_idx": 1,
                "student_number": 5,
                "image_path": "a.jpg",
                "group_keys": ["g1", "g2"],
                "related_paths": ["a.jpg", "b.jpg"],
                "message": "no clear card visible",
            }
        ],
        out_csv,
    )
    text = out_csv.read_text(encoding="utf-8-sig")
    assert "エラー種別" in text
    assert "重要度" in text
    assert "検出レイヤー" in text
    assert "札検出なし" in text
    assert "警告" in text
    assert "明確な札が見えません" in text


def test_teacher_label_mapping_localized():
    assert TEACHER_SHEET_LABELS["Match Results"] == "照合結果"
    assert TEACHER_SHEET_LABELS["Best Shot Scores"] == "ベストショットスコア"
    assert excel_label("Original Filename") == "元ファイル名"
    assert excel_label("Card Name (OCR)") == "札氏名（OCR）"
    assert translate_display_value("matched") == "照合済み"


def test_club_ranking_sheet_localized_values(tmp_path: Path):
    root = tmp_path / "in2"
    club = root / "basket"
    club.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(180, 180, 180)).save(club / "x.jpg")

    z = tmp_path / "club2.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(club / "x.jpg", arcname="basket/x.jpg")

    out = run_club_pipeline(str(z), str(tmp_path / "out2"))
    wb = load_workbook(out["excel_path"])
    ws = wb["ベストショット順位"]
    headers = [c.value for c in ws[1]]
    assert "画質・見栄え" in headers
    assert "表情スコア" in headers
    assert "雰囲気スコア" in headers
    assert "人数スコア" in headers
    assert "ポーズ減点" in headers
    assert "コメント" in headers
    assert "NG判定" in headers
    assert "NG理由" in headers

    ng_col = headers.index("NG判定") + 1
    ng_value = ws.cell(row=2, column=ng_col).value
    assert ng_value in {"はい", "いいえ"}
