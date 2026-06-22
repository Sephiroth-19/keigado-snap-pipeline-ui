import io
import unittest
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from backend import app as app_module
from backend.snap_pipeline import PipelineResult


class FakeSnapPipeline:
    def __init__(self) -> None:
        self.requested_counts: list[int | None] = []

    def run(self, input_dir: Path, output_dir: Path, best_shot_count: int | None = None) -> PipelineResult:
        self.requested_counts.append(best_shot_count)
        image_paths = sorted(
            p for p in input_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        selected_count = min(best_shot_count if best_shot_count is not None else 1, len(image_paths)) if image_paths else 0
        for folder in ["final_selected", "ng_photos", "other_passing", "similarity_clusters/cluster_001"]:
            (output_dir / folder).mkdir(parents=True, exist_ok=True)
        if image_paths:
            (output_dir / "final_selected" / image_paths[0].name).write_bytes(b"fake image")
            (output_dir / "similarity_clusters" / "cluster_001" / f"REP_{image_paths[0].name}").write_bytes(
                b"fake image"
            )
        if len(image_paths) > 1:
            (output_dir / "other_passing" / image_paths[1].name).write_bytes(b"fake image")
        (output_dir / "snap_pipeline_report.xlsx").write_bytes(b"fake xlsx")
        return PipelineResult(
            total_input_images=len(image_paths),
            total_clusters=1 if image_paths else 0,
            total_representative_candidates=len(image_paths),
            dedup_reduction_rate=0.0,
            best_shot_count=selected_count,
            final_selected_count=selected_count,
            ng_count_after_menna=0,
            other_passing_count=max(0, len(image_paths) - selected_count),
        )


def _zip_upload(entries: dict[str, bytes]) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    buf.seek(0)
    return buf


class SnapApiMultiEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = FakeSnapPipeline()
        app_module.pipeline = self.pipeline
        self.client = TestClient(app_module.app)

    def test_snap_run_multi_event_zip_outputs_separate_event_folders(self) -> None:
        upload = _zip_upload(
            {
                "運動会/IMG1201.jpg": b"a",
                "運動会/IMG1202.jpg": b"b",
                "修学旅行/IMG2201.jpg": b"c",
                "修学旅行/IMG2202.jpg": b"d",
            }
        )

        response = self.client.post("/api/snap/run", files={"folder_zip": ("events.zip", upload, "application/zip")})

        self.assertEqual(response.status_code, 200)
        result = response.json()["summary"]
        self.assertEqual([event["event_name"] for event in result["events"]], ["修学旅行", "運動会"])
        self.assertEqual(result["events"][0]["event_id"], "event_1")
        self.assertEqual(result["events"][0]["output_folder"], "修学旅行")

        preview = self.client.get("/api/snap/preview-images?event_id=event_1&bucket=final_selected").json()
        self.assertEqual(preview["count"], 1)
        self.assertEqual(preview["images"][0]["event_name"], "修学旅行")
        self.assertTrue(preview["images"][0]["relative_path"].startswith("修学旅行/final_selected/"))

        download = self.client.get("/api/snap/download")
        self.assertEqual(download.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(download.content), "r") as zf:
            names = set(zf.namelist())
        self.assertIn("all_events_summary.json", names)
        self.assertIn("all_events_summary.xlsx", names)
        self.assertIn("運動会/final_selected/IMG1201.jpg", names)
        self.assertIn("修学旅行/final_selected/IMG2201.jpg", names)
        self.assertIn("運動会/snap_pipeline_report.xlsx", names)
        self.assertFalse(any("dedup_candidates" in name for name in names))

    def test_snap_run_root_images_uses_event_1_folder(self) -> None:
        upload = _zip_upload({"IMG0001.jpg": b"a", "IMG0002.jpg": b"b"})

        response = self.client.post("/api/snap/run", files={"folder_zip": ("single.zip", upload, "application/zip")})

        self.assertEqual(response.status_code, 200)
        result = response.json()["summary"]
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][0]["event_id"], "event_1")
        self.assertEqual(result["events"][0]["event_name"], "event_1")
        self.assertEqual(result["events"][0]["output_folder"], "event_1")

        download = self.client.get("/api/snap/download")
        self.assertEqual(download.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(download.content), "r") as zf:
            names = set(zf.namelist())
        self.assertIn("event_1/final_selected/IMG0001.jpg", names)
        self.assertIn("event_1/snap_pipeline_report.xlsx", names)

    def test_snap_run_can_repeat_without_count_then_export_with_final_count(self) -> None:
        first_upload = _zip_upload({"first/IMG0001.jpg": b"a", "first/IMG0002.jpg": b"b"})
        second_upload = _zip_upload({"second/IMG1001.jpg": b"c", "second/IMG1002.jpg": b"d"})

        first = self.client.post("/api/snap/run", files={"folder_zip": ("first.zip", first_upload, "application/zip")})
        second = self.client.post("/api/snap/run", files={"folder_zip": ("second.zip", second_upload, "application/zip")})

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(self.pipeline.requested_counts, [None, None])

        download = self.client.get("/api/snap/download")
        self.assertEqual(download.status_code, 200)
        with zipfile.ZipFile(io.BytesIO(download.content), "r") as zf:
            names = set(zf.namelist())
        self.assertIn("second/final_selected/IMG1001.jpg", names)
        self.assertNotIn("first/final_selected/IMG0001.jpg", names)

        export = self.client.post("/api/snap/export", data={"best_shot_count": "2"})

        self.assertEqual(export.status_code, 200)
        self.assertEqual(self.pipeline.requested_counts, [None, None, 2])
        self.assertEqual(export.json()["summary"]["best_shot_count"], 2)


if __name__ == "__main__":
    unittest.main()
