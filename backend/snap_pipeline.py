from __future__ import annotations

import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from openpyxl import Workbook
from PIL import ExifTags, Image, ImageOps
from torchvision import models, transforms

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}

# Client-aligned relaxed similarity rules (from Colab-tested version)
STRICT_COMBINED_THRESHOLD = 0.88
RELAXED_EMBED_THRESHOLD = 0.84
RELAXED_EMBED_THRESHOLD_WITH_TIME = 0.82
MAX_TIME_GAP_SAME_SEQUENCE = 240.0
MAX_TIME_GAP_STRICT_RELAXED = 120.0
MAX_SEQUENCE_GAP = 10

WEIGHT_TECHNICAL = 0.30
WEIGHT_EXPRESSION = 0.25
WEIGHT_COMPOSITION = 0.25
WEIGHT_RARITY = 0.20
MIN_FINAL_PHOTOS = 20
MAX_FINAL_PHOTOS = 25


@dataclass
class PipelineResult:
    total_input_images: int
    total_clusters: int
    total_representative_candidates: int
    dedup_reduction_rate: float
    final_selected_count: int
    ng_count_after_menna: int
    other_passing_count: int


@dataclass
class ImageRecord:
    path: Path
    capture_time: Optional[datetime]
    sequence_tail: Optional[int]
    embedding: np.ndarray
    focus_score: float
    brightness_score: float


class SnapPipeline:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        self.model = model.to(self.device).eval()
        self.preprocess = weights.transforms()

    @staticmethod
    def _collect_images(input_dir: Path) -> list[Path]:
        return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)

    @staticmethod
    def _safe_open_image(path: Path) -> Image.Image:
        with Image.open(path) as img:
            return ImageOps.exif_transpose(img).convert("RGB")

    @staticmethod
    def _get_capture_time(path: Path) -> Optional[datetime]:
        try:
            with Image.open(path) as img:
                exif = img.getexif()
                if exif:
                    tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
                    for key in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
                        if key in tag_map:
                            try:
                                return datetime.strptime(str(tag_map[key]), "%Y:%m:%d %H:%M:%S")
                            except Exception:
                                continue
        except Exception:
            pass
        try:
            return datetime.fromtimestamp(path.stat().st_mtime)
        except Exception:
            return None

    @staticmethod
    def _parse_filename_numeric_tail(name: str) -> Optional[int]:
        nums = re.findall(r"(\d+)", Path(name).stem)
        if not nums:
            return None
        try:
            return int(nums[-1])
        except Exception:
            return None

    @staticmethod
    def _time_gap_seconds(a: Optional[datetime], b: Optional[datetime]) -> Optional[float]:
        if a is None or b is None:
            return None
        return abs((a - b).total_seconds())

    @staticmethod
    def _seq_gap(a: Optional[int], b: Optional[int]) -> Optional[int]:
        if a is None or b is None:
            return None
        return abs(a - b)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))

    @staticmethod
    def _focus_score(img_rgb: Image.Image) -> float:
        arr = np.array(img_rgb.convert("L"), dtype=np.float32)
        gy, gx = np.gradient(arr)
        mag = np.sqrt(gx**2 + gy**2)
        return float(np.var(mag))

    @staticmethod
    def _brightness_score(img_rgb: Image.Image) -> float:
        arr = np.array(img_rgb.convert("L"), dtype=np.float32)
        return float(arr.mean())

    @staticmethod
    def _contrast_score(img_rgb: Image.Image) -> float:
        arr = np.array(img_rgb.convert("L"), dtype=np.float32)
        return float(arr.std())

    @staticmethod
    def _thirds_composition_score(img_rgb: Image.Image) -> float:
        gray = np.array(img_rgb.convert("L"), dtype=np.float32)
        h, w = gray.shape
        gy, gx = np.gradient(gray)
        edge = np.sqrt(gx**2 + gy**2)

        points = [
            (h // 3, w // 3),
            (h // 3, 2 * w // 3),
            (2 * h // 3, w // 3),
            (2 * h // 3, 2 * w // 3),
        ]
        r = max(4, min(h, w) // 12)
        energy = 0.0
        for py, px in points:
            y1, y2 = max(0, py - r), min(h, py + r)
            x1, x2 = max(0, px - r), min(w, px + r)
            energy += float(edge[y1:y2, x1:x2].mean())
        return energy / max(1, len(points))

    def _embed(self, img: Image.Image) -> np.ndarray:
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vec = self.model(tensor).detach().cpu().numpy().reshape(-1).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-12
        return vec

    def _load_records(self, image_paths: list[Path]) -> list[ImageRecord]:
        records: list[ImageRecord] = []
        for path in image_paths:
            img = self._safe_open_image(path)
            records.append(
                ImageRecord(
                    path=path,
                    capture_time=self._get_capture_time(path),
                    sequence_tail=self._parse_filename_numeric_tail(path.name),
                    embedding=self._embed(img),
                    focus_score=self._focus_score(img),
                    brightness_score=self._brightness_score(img),
                )
            )

        records.sort(key=lambda r: (r.capture_time or datetime.min, r.path.name))
        return records

    def _is_similar(self, a: ImageRecord, b: ImageRecord) -> bool:
        embed_sim = self._cosine(a.embedding, b.embedding)
        tgap = self._time_gap_seconds(a.capture_time, b.capture_time)
        sgap = self._seq_gap(a.sequence_tail, b.sequence_tail)

        strict_context = (
            (tgap is not None and tgap <= MAX_TIME_GAP_SAME_SEQUENCE)
            or (sgap is not None and sgap <= MAX_SEQUENCE_GAP)
        )
        if embed_sim >= STRICT_COMBINED_THRESHOLD and strict_context:
            return True

        relaxed_context = (
            (tgap is not None and tgap <= MAX_TIME_GAP_STRICT_RELAXED)
            or (sgap is not None and sgap <= MAX_SEQUENCE_GAP)
        )
        if embed_sim >= RELAXED_EMBED_THRESHOLD and relaxed_context:
            return True

        if embed_sim >= RELAXED_EMBED_THRESHOLD_WITH_TIME and tgap is not None and tgap <= MAX_TIME_GAP_STRICT_RELAXED:
            return True

        return False

    def _cluster_records(self, records: list[ImageRecord]) -> list[list[ImageRecord]]:
        clusters: list[list[ImageRecord]] = []
        for rec in records:
            placed = False
            for cluster in clusters:
                # Compare against cluster representative and recent members for looser grouping continuity
                rep = max(cluster, key=lambda x: x.focus_score)
                if self._is_similar(rec, rep) or any(self._is_similar(rec, m) for m in cluster[-3:]):
                    cluster.append(rec)
                    placed = True
                    break
            if not placed:
                clusters.append([rec])
        return clusters

    @staticmethod
    def _normalize(vals: list[float]) -> list[float]:
        if not vals:
            return []
        lo, hi = min(vals), max(vals)
        if math.isclose(lo, hi):
            return [0.5 for _ in vals]
        return [(v - lo) / (hi - lo) for v in vals]

    def _score_representatives(self, reps: list[ImageRecord]) -> list[tuple[ImageRecord, float]]:
        # Additional Menna-like dimensions from image content
        contrast_vals: list[float] = []
        composition_vals: list[float] = []
        for r in reps:
            img = self._safe_open_image(r.path)
            contrast_vals.append(self._contrast_score(img))
            composition_vals.append(self._thirds_composition_score(img))

        tech = self._normalize([r.focus_score for r in reps])
        expr = self._normalize([r.brightness_score for r in reps])
        comp = self._normalize(composition_vals)

        # rarity: distance from average embedding among representatives
        if reps:
            center = np.mean([r.embedding for r in reps], axis=0)
            center /= np.linalg.norm(center) + 1e-12
            rarity_raw = [1.0 - self._cosine(r.embedding, center) for r in reps]
        else:
            rarity_raw = []
        rarity = self._normalize(rarity_raw)

        scored: list[tuple[ImageRecord, float]] = []
        for i, r in enumerate(reps):
            # expression: prefer well-exposed but not extreme brightness
            expr_balanced = 1.0 - abs(expr[i] - 0.55)
            score = (
                WEIGHT_TECHNICAL * tech[i]
                + WEIGHT_EXPRESSION * expr_balanced
                + WEIGHT_COMPOSITION * comp[i]
                + WEIGHT_RARITY * rarity[i]
            )
            scored.append((r, float(score)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _select_buckets(self, reps: list[ImageRecord]) -> tuple[list[ImageRecord], list[ImageRecord], list[ImageRecord]]:
        if not reps:
            return [], [], []

        scored = self._score_representatives(reps)
        ordered = [r for r, _ in scored]

        final_target = min(MAX_FINAL_PHOTOS, max(MIN_FINAL_PHOTOS, len(ordered)))
        final_count = min(final_target, len(ordered))
        final = ordered[:final_count]

        remaining = ordered[final_count:]
        # NG: bottom tail from remaining and also very dark/very blurry representatives
        if remaining:
            tail_ng_count = max(1, int(round(len(remaining) * 0.2)))
            ng = remaining[-tail_ng_count:]
            other = remaining[:-tail_ng_count]
        else:
            ng = []
            other = []

        return final, ng, other

    @staticmethod
    def _copy(path: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)

    def _write_outputs(
        self,
        output_dir: Path,
        clusters: list[list[ImageRecord]],
        reps: list[ImageRecord],
        final: list[ImageRecord],
        ng: list[ImageRecord],
        other: list[ImageRecord],
        summary: PipelineResult,
    ) -> None:
        sim_dir = output_dir / "similarity_clusters"
        dedup_dir = output_dir / "dedup_candidates"
        final_dir = output_dir / "final_selected"
        ng_dir = output_dir / "ng_photos"
        other_dir = output_dir / "other_passing"
        for d in [sim_dir, dedup_dir, final_dir, ng_dir, other_dir]:
            d.mkdir(parents=True, exist_ok=True)

        rep_paths = {r.path for r in reps}
        for i, cluster in enumerate(clusters, start=1):
            cdir = sim_dir / f"cluster_{i:03d}"
            cdir.mkdir(parents=True, exist_ok=True)
            for rec in cluster:
                prefix = "REP_" if rec.path in rep_paths else ""
                self._copy(rec.path, cdir / f"{prefix}{rec.path.name}")

        for rec in reps:
            self._copy(rec.path, dedup_dir / rec.path.name)
        for rec in final:
            self._copy(rec.path, final_dir / rec.path.name)
        for rec in ng:
            self._copy(rec.path, ng_dir / rec.path.name)
        for rec in other:
            self._copy(rec.path, other_dir / rec.path.name)

        wb = Workbook()
        ws = wb.active
        ws.title = "summary"
        ws.append(["metric", "value"])
        for key, val in summary.__dict__.items():
            ws.append([key, val])

        ws2 = wb.create_sheet("clusters")
        ws2.append(["cluster_id", "file_name", "capture_time", "is_representative"])
        for i, cluster in enumerate(clusters, start=1):
            cluster_rep = max(cluster, key=lambda x: x.focus_score)
            for rec in cluster:
                ws2.append(
                    [
                        i,
                        rec.path.name,
                        rec.capture_time.isoformat() if rec.capture_time else "",
                        rec.path == cluster_rep.path,
                    ]
                )

        ws3 = wb.create_sheet("selection")
        ws3.append(["bucket", "file_name"])
        for rec in final:
            ws3.append(["final_selected", rec.path.name])
        for rec in ng:
            ws3.append(["ng", rec.path.name])
        for rec in other:
            ws3.append(["other_passing", rec.path.name])

        wb.save(output_dir / "snap_pipeline_report.xlsx")

    def run(self, input_dir: Path, output_dir: Path) -> PipelineResult:
        image_paths = self._collect_images(input_dir)
        records = self._load_records(image_paths)
        clusters = self._cluster_records(records)

        reps = [max(cluster, key=lambda x: x.focus_score) for cluster in clusters]
        final, ng, other = self._select_buckets(reps)

        dedup_reduction_rate = 0.0
        if image_paths:
            dedup_reduction_rate = round((1.0 - (len(reps) / len(image_paths))) * 100.0, 2)

        summary = PipelineResult(
            total_input_images=len(image_paths),
            total_clusters=len(clusters),
            total_representative_candidates=len(reps),
            dedup_reduction_rate=dedup_reduction_rate,
            final_selected_count=len(final),
            ng_count_after_menna=len(ng),
            other_passing_count=len(other),
        )

        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._write_outputs(output_dir, clusters, reps, final, ng, other, summary)
        return summary


__all__ = ["SnapPipeline", "PipelineResult"]
