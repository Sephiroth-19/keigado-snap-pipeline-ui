from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from openpyxl import Workbook
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class PipelineResult:
    total_input_images: int
    total_clusters: int
    total_representative_candidates: int
    dedup_reduction_rate: float
    final_selected_count: int
    ng_count_after_menna: int
    other_passing_count: int


class SnapPipeline:
    def __init__(self, cluster_threshold: float = 0.92) -> None:
        self.cluster_threshold = cluster_threshold

    @staticmethod
    def _safe_open_image(path: Path) -> Image.Image:
        with Image.open(path) as img:
            return ImageOps.exif_transpose(img).convert("RGB")

    @staticmethod
    def _collect_images(input_dir: Path) -> list[Path]:
        return sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)

    @staticmethod
    def _embedding(img: Image.Image) -> np.ndarray:
        arr = np.array(img.resize((32, 32)).convert("L"), dtype=np.float32).reshape(-1)
        arr -= arr.mean()
        n = np.linalg.norm(arr) + 1e-12
        return arr / n

    @staticmethod
    def _focus_score(img: Image.Image) -> float:
        arr = np.array(img.convert("L"), dtype=np.float32)
        gy, gx = np.gradient(arr)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        return float(np.var(mag))

    @staticmethod
    def _brightness_score(img: Image.Image) -> float:
        arr = np.array(img.convert("L"), dtype=np.float32)
        return float(arr.mean())

    @staticmethod
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))

    def _cluster(self, images: list[Path]) -> tuple[list[list[Path]], dict[Path, np.ndarray], dict[Path, tuple[float, float]]]:
        embeddings: dict[Path, np.ndarray] = {}
        metrics: dict[Path, tuple[float, float]] = {}
        clusters: list[list[Path]] = []

        for path in images:
            img = self._safe_open_image(path)
            emb = self._embedding(img)
            focus = self._focus_score(img)
            bright = self._brightness_score(img)
            embeddings[path] = emb
            metrics[path] = (focus, bright)

            placed = False
            for cluster in clusters:
                centroid = np.mean([embeddings[p] for p in cluster], axis=0)
                sim = self._cos(emb, centroid)
                if sim >= self.cluster_threshold:
                    cluster.append(path)
                    placed = True
                    break
            if not placed:
                clusters.append([path])

        return clusters, embeddings, metrics

    @staticmethod
    def _pick_representatives(clusters: list[list[Path]], metrics: dict[Path, tuple[float, float]]) -> list[Path]:
        reps: list[Path] = []
        for cluster in clusters:
            best = max(cluster, key=lambda p: metrics[p][0])  # focus-first representative
            reps.append(best)
        return reps

    @staticmethod
    def _normalize(values: list[float]) -> list[float]:
        if not values:
            return []
        low, high = min(values), max(values)
        if math.isclose(low, high):
            return [0.5 for _ in values]
        return [(v - low) / (high - low) for v in values]

    def _select_best_shots(self, reps: list[Path], metrics: dict[Path, tuple[float, float]]) -> tuple[list[Path], list[Path], list[Path]]:
        if not reps:
            return [], [], []

        focus_values = [metrics[p][0] for p in reps]
        brightness_values = [metrics[p][1] for p in reps]
        norm_focus = self._normalize(focus_values)
        norm_brightness = self._normalize(brightness_values)

        scored: list[tuple[Path, float]] = []
        for idx, path in enumerate(reps):
            # Menna-style simplified scoring: technical sharpness + exposure balance.
            exposure_balance = 1.0 - abs(norm_brightness[idx] - 0.55)
            score = (norm_focus[idx] * 0.7) + (exposure_balance * 0.3)
            scored.append((path, score))
        scored.sort(key=lambda x: x[1], reverse=True)

        ng_count = max(0, int(round(len(scored) * 0.1)))
        ng = [p for p, _ in scored[-ng_count:]] if ng_count else []
        passing = [p for p, _ in scored[:-ng_count]] if ng_count else [p for p, _ in scored]

        final_target = min(25, max(20, len(passing)))
        final_count = min(len(passing), final_target)
        final = passing[:final_count]
        others = passing[final_count:]
        return final, ng, others

    def _write_excel(
        self,
        output_dir: Path,
        clusters: list[list[Path]],
        reps: list[Path],
        final: list[Path],
        ng: list[Path],
        others: list[Path],
        summary: PipelineResult,
    ) -> Path:
        wb = Workbook()

        ws_summary = wb.active
        ws_summary.title = "summary"
        ws_summary.append(["metric", "value"])
        for k, v in summary.__dict__.items():
            ws_summary.append([k, v])

        ws_clusters = wb.create_sheet("clusters")
        ws_clusters.append(["cluster_id", "file_name", "is_representative"])
        rep_set = set(reps)
        for idx, cluster in enumerate(clusters, start=1):
            for file in cluster:
                ws_clusters.append([idx, file.name, file in rep_set])

        ws_selection = wb.create_sheet("selection")
        ws_selection.append(["bucket", "file_name"])
        for p in final:
            ws_selection.append(["best_shot", p.name])
        for p in ng:
            ws_selection.append(["ng", p.name])
        for p in others:
            ws_selection.append(["passing", p.name])

        excel_path = output_dir / "snap_pipeline_report.xlsx"
        wb.save(excel_path)
        return excel_path

    def _copy_outputs(self, output_dir: Path, clusters: list[list[Path]], reps: list[Path], final: list[Path], ng: list[Path], others: list[Path]) -> None:
        cluster_root = output_dir / "similarity_clusters"
        dedup_root = output_dir / "dedup_candidates"
        final_root = output_dir / "final_selected"
        ng_root = output_dir / "ng_photos"
        other_root = output_dir / "other_passing"

        for d in [cluster_root, dedup_root, final_root, ng_root, other_root]:
            d.mkdir(parents=True, exist_ok=True)

        rep_set = set(reps)
        for idx, cluster in enumerate(clusters, start=1):
            group_dir = cluster_root / f"cluster_{idx:03d}"
            group_dir.mkdir(parents=True, exist_ok=True)
            for p in cluster:
                prefix = "REP_" if p in rep_set else ""
                shutil.copy2(p, group_dir / f"{prefix}{p.name}")

        for p in reps:
            shutil.copy2(p, dedup_root / p.name)
        for p in final:
            shutil.copy2(p, final_root / p.name)
        for p in ng:
            shutil.copy2(p, ng_root / p.name)
        for p in others:
            shutil.copy2(p, other_root / p.name)

    def run(self, input_dir: Path, output_dir: Path) -> PipelineResult:
        images = self._collect_images(input_dir)
        clusters, _emb, metrics = self._cluster(images)
        reps = self._pick_representatives(clusters, metrics)
        final, ng, others = self._select_best_shots(reps, metrics)

        dedup_reduction = 0.0
        if images:
            dedup_reduction = round((1 - (len(reps) / len(images))) * 100.0, 2)

        summary = PipelineResult(
            total_input_images=len(images),
            total_clusters=len(clusters),
            total_representative_candidates=len(reps),
            dedup_reduction_rate=dedup_reduction,
            final_selected_count=len(final),
            ng_count_after_menna=len(ng),
            other_passing_count=len(others),
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        self._copy_outputs(output_dir, clusters, reps, final, ng, others)
        self._write_excel(output_dir, clusters, reps, final, ng, others, summary)
        return summary


__all__ = ["SnapPipeline", "PipelineResult"]
