# -*- coding: utf-8 -*-
"""
Snap pipeline (integrated version)
2026/4/23

Flow:
1) Load original snap photos
2) Run similarity clustering (looser client-aligned rules)
3) Keep one representative candidate per cluster
4) Pass representative candidates into Menna-style snap scoring
5) Select final top 20-25 photos

Design note:
- Menna's core scoring / final selection logic is preserved conceptually.
- Main integration change is at the input stage: Menna evaluates deduplicated candidates instead of all originals.
- Similarity stage is intentionally slightly looser than the previous updated version.
"""

# ================================
# CELL 1) Install + Imports
# ================================
# !pip -q install -U google-genai openai openpyxl pillow numpy pandas tqdm torch torchvision

import os
import io
import re
import json
import base64
import shutil
import zipfile
import datetime
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ExifTags, ImageDraw
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import models

from google import genai
from google.genai import types
from google.colab import files, drive, userdata
from IPython.display import display
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import time

print("OK - imports successful")

# ================================
# CELL 2) API / Model Configuration
# ================================
try:
    GEMINI_API_KEY = userdata.get("GOOGLE_API_KEY")
except Exception:
    GEMINI_API_KEY = "YOUR_GEMINI_KEY"

client = genai.Client(api_key=GEMINI_API_KEY)

try:
    OPENAI_API_KEY = userdata.get("OPENAI_API_KEY")
    from openai import OpenAI
    oa_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception:
    OPENAI_API_KEY = None
    oa_client = None

MODEL_NAME = "gpt-5.4"
print(f"Model: {MODEL_NAME}")
print(f"OpenAI client initialized: {oa_client is not None}")

# ================================
# CELL 3) Directory Configuration
# ================================
RAW_INPUT_DIR = "/content/snap_raw_inputs"
SIM_OUTPUT_DIR = "/content/snap_similarity_outputs"
DEDUP_INPUT_DIR = "/content/snap_dedup_inputs_for_menna"
FINAL_OUTPUT_DIR = "/content/snap_pipeline_outputs"

for d in [RAW_INPUT_DIR, SIM_OUTPUT_DIR, DEDUP_INPUT_DIR, FINAL_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

BEST_TOKEN = "本"

# Menna selection configuration
WEIGHT_TECHNICAL = 0.30
WEIGHT_EXPRESSION = 0.25
WEIGHT_COMPOSITION = 0.25
WEIGHT_RARITY = 0.20

EVENT_TYPE = "general"
TARGET_GRADE = None
MIN_FINAL_PHOTOS = 20
MAX_FINAL_PHOTOS = 25
MAX_REPEAT_SAME_PERSON = 2

# Similarity configuration (slightly looser than previous updated version)
STRICT_COMBINED_THRESHOLD = 0.88
RELAXED_EMBED_THRESHOLD = 0.84
RELAXED_EMBED_THRESHOLD_WITH_TIME = 0.82
MAX_TIME_GAP_SAME_SEQUENCE = 240.0
MAX_TIME_GAP_STRICT_RELAXED = 120.0
MAX_SEQUENCE_GAP = 10

# ================================
# CELL 4) General Helpers
# ================================
def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)

def reset_dir(d: str):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

def collect_image_paths(root_dir: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif", ".tif", ".tiff")
    image_paths = []
    for root, _, files_ in os.walk(root_dir):
        for f in files_:
            if f.lower().endswith(exts):
                image_paths.append(os.path.join(root, f))
    return sorted(image_paths)

def safe_open_image(path: str) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img

def get_capture_time(path: str) -> Optional[datetime]:
    try:
        img = Image.open(path)
        exif = img.getexif()
        if exif:
            tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
            for key in ["DateTimeOriginal", "DateTimeDigitized", "DateTime"]:
                if key in tag_map:
                    try:
                        return datetime.strptime(str(tag_map[key]), "%Y:%m:%d %H:%M:%S")
                    except Exception:
                        pass
    except Exception:
        pass
    try:
        return datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        return None

def parse_filename_numeric_tail(name: str) -> Optional[int]:
    base = os.path.splitext(os.path.basename(name))[0]
    nums = re.findall(r"(\d+)", base)
    if not nums:
        return None
    try:
        return int(nums[-1])
    except Exception:
        return None

def get_sequence_gap(a: Optional[int], b: Optional[int]) -> Optional[int]:
    if a is None or b is None:
        return None
    return abs(a - b)

def get_time_gap_seconds(a: Optional[datetime], b: Optional[datetime]) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs((a - b).total_seconds())

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def draw_cluster_label(img_rgb: Image.Image, text: str) -> Image.Image:
    img = img_rgb.copy()
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), text, fill=(255, 0, 0))
    return img

def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    k = 2
    while True:
        candidate = f"{base}__dup{k}{ext}"
        if not os.path.exists(candidate):
            return candidate
        k += 1

# ================================
# CELL 5) Load Images
# ================================
drive.mount("/content/drive")

DRIVE_SOURCE_FOLDER = "/content/drive/MyDrive/LTID/Photography_Pipeline/Snap Pictures Samples/Best shot folders/"

def load_images_from_drive_to_raw_input(source_folder: str = DRIVE_SOURCE_FOLDER):
    reset_dir(RAW_INPUT_DIR)
    folder_map = {}
    if os.path.exists(source_folder):
        exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
        found_count = 0
        for root, dirs, filenames in os.walk(source_folder):
            rel_folder = os.path.relpath(root, source_folder)
            if rel_folder == ".":
                rel_folder = "root"
            for f in filenames:
                if f.lower().endswith(exts):
                    src = os.path.join(root, f)
                    unique_name = f"{rel_folder.replace(os.sep, '_')}_{f}" if rel_folder != "root" else f
                    dst = os.path.join(RAW_INPUT_DIR, unique_name)
                    shutil.copy2(src, dst)
                    folder_map[dst] = rel_folder
                    found_count += 1
        image_paths = sorted([os.path.join(RAW_INPUT_DIR, f) for f in os.listdir(RAW_INPUT_DIR) if f.lower().endswith(exts)])
        print(f"Loaded {found_count} images from Drive into RAW_INPUT_DIR.")
        return image_paths, folder_map
    else:
        print(f"Source folder not found: {source_folder}")
        return [], {}

def upload_images_to_raw_input():
    reset_dir(RAW_INPUT_DIR)
    uploaded = files.upload()
    for fn in uploaded.keys():
        src = f"/content/{fn}"
        if fn.lower().endswith(".zip"):
            with zipfile.ZipFile(src, "r") as z:
                z.extractall(RAW_INPUT_DIR)
            os.remove(src)
        else:
            shutil.move(src, os.path.join(RAW_INPUT_DIR, fn))
    image_paths = collect_image_paths(RAW_INPUT_DIR)
    folder_map = {p: "root" for p in image_paths}
    print(f"Uploaded images found: {len(image_paths)}")
    return image_paths, folder_map

# Use Drive by default. Comment out if you want manual upload instead.
image_paths, folder_map = load_images_from_drive_to_raw_input()
# image_paths, folder_map = upload_images_to_raw_input()

# ================================
# CELL 6) Similarity Stage Helpers
# ================================
def compute_focus_score(img_rgb: Image.Image) -> float:
    arr = np.array(img_rgb).astype(np.float32)
    gray = arr.mean(axis=2)
    gy, gx = np.gradient(gray)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.var(mag))

def compute_brightness_score(img_rgb: Image.Image) -> float:
    arr = np.array(img_rgb).astype(np.float32)
    gray = arr.mean(axis=2)
    return float(gray.mean())

def compute_contrast_score(img_rgb: Image.Image) -> float:
    arr = np.array(img_rgb).astype(np.float32)
    gray = arr.mean(axis=2)
    return float(gray.std())

def exposure_balance_score(brightness: float) -> float:
    return float(max(0.0, 1.0 - abs(brightness - 128.0) / 128.0))

def aspect_type(width: int, height: int) -> str:
    if width > height:
        return "horizontal"
    if height > width:
        return "vertical"
    return "square"

def normalize_global(values: List[float], v: float) -> float:
    if not values:
        return 0.5
    lo = float(np.percentile(values, 5))
    hi = float(np.percentile(values, 95))
    if hi <= lo:
        return 0.5
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

def representative_quality_score(record: Dict[str, Any], global_stats: Dict[str, List[float]]) -> float:
    focus_n = normalize_global(global_stats["focus_scores"], record["focus_score"])
    contrast_n = normalize_global(global_stats["contrast_scores"], record["contrast_score"])
    resolution_n = normalize_global(global_stats["resolutions"], record["resolution"])
    exposure_n = exposure_balance_score(record["brightness_score"])

    ratio = record["width"] / max(record["height"], 1)
    ratio_penalty = 0.0
    if ratio > 2.5 or ratio < 0.40:
        ratio_penalty = 0.06

    score = (
        0.48 * focus_n +
        0.22 * exposure_n +
        0.15 * contrast_n +
        0.15 * resolution_n -
        ratio_penalty
    )
    return float(score)

def choose_representative(cluster: Dict[str, Any], member_records: List[Dict[str, Any]], global_stats: Dict[str, List[float]]):
    candidates = []
    for idx in cluster["members"]:
        r = member_records[idx]
        capture_time = r.get("capture_time")
        capture_ts = capture_time.timestamp() if capture_time else float("-inf")
        quality = representative_quality_score(r, global_stats)
        candidates.append((quality, r["focus_score"], capture_ts, r["filename"], idx))
    candidates.sort(reverse=True)
    cluster["representative_member"] = candidates[0][4]
    cluster["representative_quality_score"] = float(candidates[0][0])

def update_cluster_centroid(cluster: Dict[str, Any], member_records: List[Dict[str, Any]]):
    embs = np.vstack([member_records[idx]["embedding"] for idx in cluster["members"]])
    mean_emb = embs.mean(axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
    cluster["centroid_embedding"] = mean_emb

    capture_times = [member_records[idx]["capture_time"] for idx in cluster["members"] if member_records[idx]["capture_time"] is not None]
    if capture_times:
        cluster["min_capture_time"] = min(capture_times)
        cluster["max_capture_time"] = max(capture_times)
    else:
        cluster["min_capture_time"] = None
        cluster["max_capture_time"] = None

    seq_values = [member_records[idx]["sequence_no"] for idx in cluster["members"] if member_records[idx]["sequence_no"] is not None]
    if seq_values:
        cluster["min_sequence_no"] = min(seq_values)
        cluster["max_sequence_no"] = max(seq_values)
    else:
        cluster["min_sequence_no"] = None
        cluster["max_sequence_no"] = None

# ================================
# CELL 7) Setup image embedding model
# ================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

weights = models.ResNet50_Weights.DEFAULT
backbone = models.resnet50(weights=weights)
feature_model = nn.Sequential(*list(backbone.children())[:-1]).to(DEVICE)
feature_model.eval()
preprocess = weights.transforms()

print("Device:", DEVICE)
print("Image embedding model ready")

# ================================
# CELL 8) Extract image-level features
# ================================
records = []
failures = []

for path in tqdm(image_paths, desc="Extract image features"):
    rel_name = os.path.relpath(path, RAW_INPUT_DIR)
    try:
        img = safe_open_image(path)
        capture_time = get_capture_time(path)
        focus_score = compute_focus_score(img)
        brightness_score = compute_brightness_score(img)
        contrast_score = compute_contrast_score(img)
        width, height = img.size
        resolution = int(width * height)
        orientation = aspect_type(width, height)
        sequence_no = parse_filename_numeric_tail(rel_name)

        tensor = preprocess(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            feat = feature_model(tensor).flatten().detach().cpu().numpy().astype(np.float32)
        feat = feat / (np.linalg.norm(feat) + 1e-12)

        records.append({
            "filename": rel_name,
            "path": path,
            "width": width,
            "height": height,
            "resolution": resolution,
            "orientation": orientation,
            "capture_time": capture_time,
            "sequence_no": sequence_no,
            "focus_score": focus_score,
            "brightness_score": brightness_score,
            "contrast_score": contrast_score,
            "embedding": feat,
        })
    except Exception as e:
        failures.append({"filename": rel_name, "error": str(e)})

print("Feature extraction done")
print("Valid images:", len(records))
print("Failures:", len(failures))

GLOBAL_STATS = {
    "focus_scores": [r["focus_score"] for r in records],
    "contrast_scores": [r["contrast_score"] for r in records],
    "resolutions": [r["resolution"] for r in records],
}
for r in records:
    r["representative_quality_score"] = representative_quality_score(r, GLOBAL_STATS)

# ================================
# CELL 9) Similarity clustering
# ================================
def cluster_match_decision(record: Dict[str, Any], cluster: Dict[str, Any], member_records: List[Dict[str, Any]]) -> Tuple[bool, float, Dict[str, Any]]:
    rep = member_records[cluster["representative_member"]]

    emb_sim_centroid = cosine_sim(record["embedding"], cluster["centroid_embedding"])
    emb_sim_rep = cosine_sim(record["embedding"], rep["embedding"])
    emb_sim = max(emb_sim_centroid, emb_sim_rep)

    hash_sim = None

    combined = emb_sim

    time_gap_rep = get_time_gap_seconds(record["capture_time"], rep.get("capture_time"))
    time_gap_cluster = None
    if cluster.get("min_capture_time") and cluster.get("max_capture_time") and record.get("capture_time"):
        time_gap_cluster = min(
            abs((record["capture_time"] - cluster["min_capture_time"]).total_seconds()),
            abs((record["capture_time"] - cluster["max_capture_time"]).total_seconds()),
        )
    time_gap = time_gap_rep if time_gap_cluster is None else min(time_gap_rep or 10**9, time_gap_cluster)

    seq_gap_rep = get_sequence_gap(record.get("sequence_no"), rep.get("sequence_no"))
    seq_gap_cluster = None
    if cluster.get("min_sequence_no") is not None and cluster.get("max_sequence_no") is not None and record.get("sequence_no") is not None:
        seq_gap_cluster = min(
            abs(record["sequence_no"] - cluster["min_sequence_no"]),
            abs(record["sequence_no"] - cluster["max_sequence_no"]),
        )
    seq_gap_candidates = [x for x in [seq_gap_rep, seq_gap_cluster] if x is not None]
    seq_gap = min(seq_gap_candidates) if seq_gap_candidates else None

    orientation_diff = record["orientation"] != rep["orientation"]

    if combined >= STRICT_COMBINED_THRESHOLD:
        return True, combined, {
            "merge_rule": "strict_duplicate_or_near_duplicate",
            "embedding_similarity": emb_sim,
            "phash_similarity": None,
            "combined_similarity": combined,
            "time_gap_seconds": time_gap,
            "sequence_gap": seq_gap,
            "orientation_diff": orientation_diff,
        }

    if (
        emb_sim >= RELAXED_EMBED_THRESHOLD and
        (
            (time_gap is not None and time_gap <= MAX_TIME_GAP_SAME_SEQUENCE) or
            (seq_gap is not None and seq_gap <= MAX_SEQUENCE_GAP)
        )
    ):
        return True, combined, {
            "merge_rule": "relaxed_same_sequence_or_orientation_tolerant",
            "embedding_similarity": emb_sim,
            "phash_similarity": None,
            "combined_similarity": combined,
            "time_gap_seconds": time_gap,
            "sequence_gap": seq_gap,
            "orientation_diff": orientation_diff,
        }

    if (
        emb_sim >= RELAXED_EMBED_THRESHOLD_WITH_TIME and
        (
            (time_gap is not None and time_gap <= MAX_TIME_GAP_STRICT_RELAXED) or
            (seq_gap is not None and seq_gap <= MAX_SEQUENCE_GAP) or
            orientation_diff
        )
    ):
        return True, combined, {
            "merge_rule": "same_scene_group_variation_tolerant",
            "embedding_similarity": emb_sim,
            "phash_similarity": None,
            "combined_similarity": combined,
            "time_gap_seconds": time_gap,
            "sequence_gap": seq_gap,
            "orientation_diff": orientation_diff,
        }

    return False, combined, {
        "merge_rule": "no_merge",
        "embedding_similarity": emb_sim,
        "phash_similarity": None,
        "combined_similarity": combined,
        "time_gap_seconds": time_gap,
        "sequence_gap": seq_gap,
        "orientation_diff": orientation_diff,
    }

clusters = []

for idx, rec in enumerate(tqdm(records, desc="Clustering images")):
    if not clusters:
        cluster_id = f"Cluster_{len(clusters)+1:03d}"
        clusters.append({
            "cluster_id": cluster_id,
            "members": [idx],
            "representative_member": idx,
            "representative_quality_score": rec["representative_quality_score"],
            "centroid_embedding": rec["embedding"],
            "min_capture_time": rec["capture_time"],
            "max_capture_time": rec["capture_time"],
            "min_sequence_no": rec["sequence_no"],
            "max_sequence_no": rec["sequence_no"],
        })
        rec["cluster_id"] = cluster_id
        rec["cluster_similarity"] = 1.0
        rec["embedding_similarity"] = 1.0
        rec["phash_similarity"] = None
        rec["merge_rule"] = "new_cluster"
        rec["time_gap_seconds"] = 0.0
        rec["sequence_gap"] = 0
        rec["orientation_diff"] = False
        rec["is_new_cluster"] = True
        continue

    best_idx = None
    best_score = -1.0
    best_detail = None

    for ci, cluster in enumerate(clusters):
        should_merge, score, detail = cluster_match_decision(rec, cluster, records)
        if should_merge and score > best_score:
            best_idx = ci
            best_score = score
            best_detail = detail

    if best_idx is None:
        cluster_id = f"Cluster_{len(clusters)+1:03d}"
        clusters.append({
            "cluster_id": cluster_id,
            "members": [idx],
            "representative_member": idx,
            "representative_quality_score": rec["representative_quality_score"],
            "centroid_embedding": rec["embedding"],
            "min_capture_time": rec["capture_time"],
            "max_capture_time": rec["capture_time"],
            "min_sequence_no": rec["sequence_no"],
            "max_sequence_no": rec["sequence_no"],
        })
        rec["cluster_id"] = cluster_id
        rec["cluster_similarity"] = 1.0
        rec["embedding_similarity"] = 1.0
        rec["phash_similarity"] = None
        rec["merge_rule"] = "new_cluster"
        rec["time_gap_seconds"] = 0.0
        rec["sequence_gap"] = 0
        rec["orientation_diff"] = False
        rec["is_new_cluster"] = True
    else:
        cluster = clusters[best_idx]
        cluster["members"].append(idx)
        choose_representative(cluster, records, GLOBAL_STATS)
        update_cluster_centroid(cluster, records)

        rec["cluster_id"] = cluster["cluster_id"]
        rec["cluster_similarity"] = float(best_score)
        rec["embedding_similarity"] = float(best_detail["embedding_similarity"])
        rec["phash_similarity"] = best_detail["phash_similarity"]
        rec["merge_rule"] = best_detail["merge_rule"]
        rec["time_gap_seconds"] = best_detail["time_gap_seconds"]
        rec["sequence_gap"] = best_detail["sequence_gap"]
        rec["orientation_diff"] = bool(best_detail["orientation_diff"])
        rec["is_new_cluster"] = False

for c in clusters:
    choose_representative(c, records, GLOBAL_STATS)
    update_cluster_centroid(c, records)

rep_member_to_cluster = {c["representative_member"]: c["cluster_id"] for c in clusters}
for idx, rec in enumerate(records):
    rec["is_representative"] = idx in rep_member_to_cluster
    rec["for_snap_candidate"] = rec["is_representative"]

print("Clusters built:", len(clusters))
print("Representative candidates for Menna stage:", sum(r["for_snap_candidate"] for r in records))

# ================================
# CELL 10) Save similarity outputs + prepare Menna input
# ================================
reset_dir(SIM_OUTPUT_DIR)
reset_dir(os.path.join(SIM_OUTPUT_DIR, "annotated_all"))
reset_dir(os.path.join(SIM_OUTPUT_DIR, "representatives"))
reset_dir(DEDUP_INPUT_DIR)

image_rows = []
cluster_rows = []

for rec in records:
    image_rows.append({
        "filename": rec["filename"],
        "cluster_id": rec["cluster_id"],
        "cluster_similarity": rec["cluster_similarity"],
        "embedding_similarity": rec["embedding_similarity"],
        "phash_similarity": rec["phash_similarity"],
        "merge_rule": rec["merge_rule"],
        "is_representative": rec["is_representative"],
        "for_snap_candidate": rec["for_snap_candidate"],
        "representative_quality_score": rec["representative_quality_score"],
        "focus_score": rec["focus_score"],
        "brightness_score": rec["brightness_score"],
        "contrast_score": rec["contrast_score"],
        "capture_time": rec["capture_time"].isoformat() if rec["capture_time"] else None,
        "time_gap_seconds": rec["time_gap_seconds"],
        "sequence_no": rec["sequence_no"],
        "sequence_gap": rec["sequence_gap"],
        "orientation": rec["orientation"],
        "orientation_diff": rec["orientation_diff"],
        "width": rec["width"],
        "height": rec["height"],
        "resolution": rec["resolution"],
    })

for c in clusters:
    rep = records[c["representative_member"]]
    cluster_rows.append({
        "cluster_id": c["cluster_id"],
        "num_images": len(c["members"]),
        "representative_filename": rep["filename"],
        "representative_quality_score": rep["representative_quality_score"],
        "representative_focus_score": rep["focus_score"],
        "representative_capture_time": rep["capture_time"].isoformat() if rep["capture_time"] else None,
        "representative_orientation": rep["orientation"],
    })

for rec in tqdm(records, desc="Saving annotated similarity outputs"):
    img = safe_open_image(rec["path"])
    label = f"{rec['cluster_id']} | rep={rec['is_representative']} | {rec['merge_rule']}"
    annotated = draw_cluster_label(img, label)

    out_name = os.path.basename(rec["filename"])
    annotated.save(os.path.join(SIM_OUTPUT_DIR, "annotated_all", out_name))

    if rec["is_representative"]:
        annotated.save(os.path.join(SIM_OUTPUT_DIR, "representatives", out_name))
        shutil.copy2(rec["path"], os.path.join(DEDUP_INPUT_DIR, out_name))

df_sim_image = pd.DataFrame(image_rows).sort_values(["cluster_id", "is_representative", "filename"], ascending=[True, False, True])
df_sim_cluster = pd.DataFrame(cluster_rows).sort_values(["num_images", "cluster_id"], ascending=[False, True])
df_sim_summary = pd.DataFrame([{
    "total_input_images": len(records),
    "total_clusters": len(clusters),
    "total_representative_candidates": int(df_sim_image["for_snap_candidate"].sum()) if not df_sim_image.empty else 0,
    "dedup_reduction_rate": round(1.0 - (float(df_sim_image["for_snap_candidate"].sum()) / max(len(records), 1)), 4) if not df_sim_image.empty else 0.0,
}])
df_sim_fail = pd.DataFrame(failures)

sim_image_csv = os.path.join(SIM_OUTPUT_DIR, "similarity_image_level.csv")
sim_cluster_csv = os.path.join(SIM_OUTPUT_DIR, "similarity_cluster_gallery.csv")
sim_summary_csv = os.path.join(SIM_OUTPUT_DIR, "similarity_summary.csv")
sim_fail_csv = os.path.join(SIM_OUTPUT_DIR, "similarity_failures.csv")

df_sim_image.to_csv(sim_image_csv, index=False)
df_sim_cluster.to_csv(sim_cluster_csv, index=False)
df_sim_summary.to_csv(sim_summary_csv, index=False)
df_sim_fail.to_csv(sim_fail_csv, index=False)

print("Similarity stage outputs saved.")
print("Deduplicated candidate folder for Menna stage:", DEDUP_INPUT_DIR)

# ================================
# CELL 11) Menna event-aware rules + helpers
# ================================
EVENT_RULES = {
    "general": {
        "required_tag_groups": [],
        "must_have_landmark": 0,
        "cheering_range": (0, 0),
        "max_transit_or_meal": None,
        "balance_by_class": False,
        "balance_by_day": False,
        "require_ball_visible": False,
    },
    "entrance_ceremony": {
        "required_tag_groups": [
            ["entrance sign", "memorial"],
            ["full group", "group shot"],
            ["procession", "entrance procession"],
            ["principal speech"],
            ["student representative speech", "representative speech"],
            ["homeroom teacher introduction", "teacher introduction"],
        ],
        "must_have_landmark": 0,
        "cheering_range": (0, 0),
        "max_transit_or_meal": None,
        "balance_by_class": False,
        "balance_by_day": False,
        "require_ball_visible": False,
    },
    "field_trip": {
        "required_tag_groups": [["landmark", "destination"]],
        "must_have_landmark": 2,
        "cheering_range": (0, 0),
        "max_transit_or_meal": 2,
        "balance_by_class": False,
        "balance_by_day": False,
        "require_ball_visible": False,
    },
    "sports_festival": {
        "required_tag_groups": [["athlete oath"], ["award ceremony", "results announcement"]],
        "must_have_landmark": 0,
        "cheering_range": (1, 2),
        "max_transit_or_meal": None,
        "balance_by_class": True,
        "balance_by_day": False,
        "require_ball_visible": False,
    },
    "other_athletic": {
        "required_tag_groups": [["start"], ["finish"]],
        "must_have_landmark": 0,
        "cheering_range": (0, 1),
        "max_transit_or_meal": None,
        "balance_by_class": False,
        "balance_by_day": False,
        "require_ball_visible": True,
    },
    "cultural_festival": {
        "required_tag_groups": [["stage performance"], ["class exhibit", "entrance gate", "club activity"]],
        "must_have_landmark": 0,
        "cheering_range": (0, 0),
        "max_transit_or_meal": None,
        "balance_by_class": True,
        "balance_by_day": False,
        "require_ball_visible": False,
    },
    "other_cultural": {
        "required_tag_groups": [["award ceremony"], ["group shot"]],
        "must_have_landmark": 0,
        "cheering_range": (0, 0),
        "max_transit_or_meal": None,
        "balance_by_class": True,
        "balance_by_day": False,
        "require_ball_visible": False,
    },
    "school_trip": {
        "required_tag_groups": [["landmark", "destination"]],
        "must_have_landmark": 2,
        "cheering_range": (0, 0),
        "max_transit_or_meal": 2,
        "balance_by_class": False,
        "balance_by_day": True,
        "require_ball_visible": False,
    },
    "other_off_campus": {
        "required_tag_groups": [["landmark", "destination"]],
        "must_have_landmark": 1,
        "cheering_range": (0, 0),
        "max_transit_or_meal": 2,
        "balance_by_class": False,
        "balance_by_day": True,
        "require_ball_visible": False,
    },
}

EVENT_NAME_HINTS = {
    "entrance_ceremony": ["entrance_ceremony", "school_entrance", "entrance ceremony", "入学式", "school entrance"],
    "field_trip": ["field_trip", "field trip", "遠足"],
    "sports_festival": ["sports_festival", "sport_event", "sports event", "体育祭", "athletic_meet"],
    "other_athletic": ["marathon", "ball game", "ball_game", "relay", "athletic"],
    "cultural_festival": ["cultural_festival", "cultural festival", "文化祭"],
    "other_cultural": ["choir", "speech_contest", "speech contest", "合唱", "弁論"],
    "school_trip": ["school_trip", "school trip", "修学旅行"],
    "other_off_campus": ["forest school", "british hills", "off_campus", "校外"],
}

def get_file_date_string(path: str) -> str:
    try:
        img = Image.open(path)
        exif = img._getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if tag == "DateTimeOriginal" and value:
                    return str(value)[:10].replace(":", "")
    except Exception:
        pass
    try:
        mod_time = os.path.getmtime(path)
        return datetime.fromtimestamp(mod_time).strftime("%Y%m%d")
    except Exception:
        return "00000000"

def load_image_for_gemini(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

def parse_json_response(text: str) -> dict:
    if not text:
        raise ValueError("Empty model response")
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON object in response: {text}")
    json_str = match.group(0)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        cleaned = re.sub(r'\n', ' ', json_str)
        return json.loads(cleaned)

def coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    return default

def normalize_event_type(event_type: Optional[str]) -> str:
    if not event_type:
        return "general"
    key = str(event_type).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "entrance": "entrance_ceremony",
        "entrance_ceremony": "entrance_ceremony",
        "field_trip": "field_trip",
        "sports": "sports_festival",
        "sports_festival": "sports_festival",
        "athletic": "other_athletic",
        "other_athletic": "other_athletic",
        "cultural": "cultural_festival",
        "cultural_festival": "cultural_festival",
        "other_cultural": "other_cultural",
        "school_trip": "school_trip",
        "off_campus": "other_off_campus",
        "other_off_campus": "other_off_campus",
        "general": "general",
    }
    return aliases.get(key, "general")

def infer_event_type_from_folder(folder_name: Optional[str]) -> str:
    if not folder_name:
        return normalize_event_type(EVENT_TYPE)
    raw = str(folder_name).strip().lower().replace("-", "_")
    for event_key, hints in EVENT_NAME_HINTS.items():
        for hint in hints:
            hint_norm = str(hint).lower().replace("-", "_")
            if hint_norm in raw:
                return event_key
    return normalize_event_type(EVENT_TYPE)

def get_event_rules(event_type: Optional[str]) -> Dict[str, Any]:
    return EVENT_RULES.get(normalize_event_type(event_type), EVENT_RULES["general"])

def normalize_tags(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        tags = value
    elif isinstance(value, str):
        tags = re.split(r"[,;|]", value)
    else:
        tags = [str(value)]
    return [str(t).strip().lower() for t in tags if str(t).strip()]

def normalize_class_hint(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if not text or text == "unknown":
        return "unknown"
    class_ids = []
    for match in re.finditer(r"(?:class|grade|組)?\s*(\d{1,2})\s*(?:組|gumi|class|grade)?", text):
        class_ids.append(match.group(1))
    if not class_ids:
        class_ids = re.findall(r"\d{1,2}", text)
    unique_ids = []
    for class_id in class_ids:
        if class_id not in unique_ids:
            unique_ids.append(class_id)
    if unique_ids:
        return "class_" + "_".join(unique_ids)
    cleaned = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return cleaned or "unknown"

def build_photo_eval_prompt(event_type: str, target_grade: Optional[str], folder_name: str) -> str:
    normalized_event = normalize_event_type(event_type)
    rules = get_event_rules(normalized_event)

    event_guidance = {
        "general": "Use general school-event best-shot criteria.",
        "entrance_ceremony": "Include key ceremony moments and balanced coverage.",
        "field_trip": "Prefer photos that clearly show landmarks so the destination is identifiable.",
        "sports_festival": "Use photos from all classes as evenly as possible. Include cheering if available.",
        "other_athletic": "Apply sports logic. For ball games, prioritize shots where the ball is visible.",
        "cultural_festival": "Use photos from all classes as evenly as possible and cover multiple activities.",
        "other_cultural": "Apply cultural selection logic, including award ceremonies and group shots if available.",
        "school_trip": "Include landmarks and distribute photos across itinerary days.",
        "other_off_campus": "Apply field trip and school trip criteria.",
    }

    target_grade_text = (
        f"Target grade year is '{target_grade}'. Prioritize this grade by visual cues."
        if target_grade else
        "Target grade is not explicitly provided. Infer likely relevance carefully."
    )

    required_groups = rules.get("required_tag_groups", [])
    required_as_text = "; ".join([" / ".join(group) for group in required_groups]) if required_groups else "none"

    return f"""
あなたはプロの学校卒業アルバム写真選定者です。画像を分析し、JSONのみを返してください。
すべての文字列フィールドは必ず日本語で回答してください。

コンテキスト:
- イベント種別: {normalized_event}
- ソースフォルダ: {folder_name}
- {target_grade_text}
- イベントガイダンス: {event_guidance.get(normalized_event, event_guidance['general'])}
- 必須シーンカテゴリ: {required_as_text}

タスク1: NGフィルター
- 主役が明らかにブレている
- 主役の顔が切れている/隠れている
- 極端な露出不良
- 主役が完全に目を閉じている
- 明らかな失敗瞬間

タスク2: スコアリング（0〜10の整数）
- technical_score
- expression_score
- composition_score
- rarity_score
- target_grade_relevance
- event_relevance

タスク3: タグ付け
- shot_type_tags
- class_hint
- day_hint
- person_signature

出力形式（厳密なJSONのみ）:
{{
  "is_ng": true または false,
  "ng_reason": "文字列 または null",
  "technical_score": 整数,
  "expression_score": 整数,
  "composition_score": 整数,
  "rarity_score": 整数,
  "target_grade_relevance": 整数,
  "event_relevance": 整数,
  "blur_detected": true または false,
  "exposure_issue": "なし" または "露出不足" または "露出過多" または "逆光",
  "protagonist_eyes_closed": true または false,
  "protagonist_imperfect_moment": true または false,
  "bystander_issue": true または false,
  "eyes_closed": true または false,
  "face_visible": true または false,
  "landmark_visible": true または false,
  "ball_visible": true または false,
  "cheering_shot": true または false,
  "shot_type_tags": ["タグ1", "タグ2"],
  "class_hint": "文字列",
  "day_hint": "文字列",
  "person_signature": "文字列",
  "short_comment": "一文で（日本語）"
}}
""".strip()

def evaluate_photo(
    image_path: str,
    folder_name: str = "unknown",
    event_type: str = EVENT_TYPE,
    target_grade: Optional[str] = TARGET_GRADE,
    model_name: str = MODEL_NAME,
) -> dict:
    try:
        prompt = build_photo_eval_prompt(event_type, target_grade, folder_name)

        if "gpt" in model_name.lower():
            if not oa_client:
                raise ValueError("OpenAI client not initialized. Check API Key.")
            with open(image_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode("utf-8")

            response = oa_client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }],
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
        else:
            img = load_image_for_gemini(image_path)
            response = client.models.generate_content(
                model=model_name,
                contents=[prompt, img],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=2048,
                    response_mime_type="application/json"
                )
            )
            result = parse_json_response(response.text)

        defaults = {
            "is_ng": False,
            "ng_reason": None,
            "technical_score": 5,
            "expression_score": 5,
            "composition_score": 5,
            "rarity_score": 5,
            "blur_detected": False,
            "exposure_issue": "なし",
            "eyes_closed": False,
            "protagonist_eyes_closed": False,
            "protagonist_imperfect_moment": False,
            "bystander_issue": False,
            "face_visible": True,
            "short_comment": "",
            "target_grade_relevance": 5,
            "event_relevance": 5,
            "landmark_visible": False,
            "ball_visible": False,
            "cheering_shot": False,
            "shot_type_tags": [],
            "class_hint": "不明",
            "day_hint": "不明",
            "person_signature": "不明",
        }
        for k, v in defaults.items():
            result.setdefault(k, v)

        result["is_ng"] = coerce_bool(result.get("is_ng"), False)
        result["blur_detected"] = coerce_bool(result.get("blur_detected"), False)
        result["face_visible"] = coerce_bool(result.get("face_visible"), True)
        result["landmark_visible"] = coerce_bool(result.get("landmark_visible"), False)
        result["ball_visible"] = coerce_bool(result.get("ball_visible"), False)
        result["cheering_shot"] = coerce_bool(result.get("cheering_shot"), False)
        result["protagonist_eyes_closed"] = coerce_bool(result.get("protagonist_eyes_closed"), False)
        result["protagonist_imperfect_moment"] = coerce_bool(result.get("protagonist_imperfect_moment"), False)
        result["bystander_issue"] = coerce_bool(result.get("bystander_issue"), False)
        result["eyes_closed"] = result["protagonist_eyes_closed"]
        result["shot_type_tags"] = normalize_tags(result.get("shot_type_tags", []))

        tags = set(result["shot_type_tags"])
        result["has_burst"] = any("連写" in t or "burst" in t for t in tags)
        result["has_similar_composition"] = any(("類似" in t and "構図" in t) or ("similar" in t and "composition" in t) for t in tags)
        result["has_duplicate_subject"] = any("被写体重複" in t or "duplicate" in t for t in tags)

        if result.get("is_ng") and result.get("ng_reason"):
            reason_lower = str(result["ng_reason"]).lower()
            eyes_keywords = ("eye", "closed", "blink", "imperfect", "awkward", "目閉", "まばたき", "失敗", "不格好", "目をつぶ")
            if any(kw in reason_lower for kw in eyes_keywords):
                if not result["protagonist_eyes_closed"] and not result["protagonist_imperfect_moment"]:
                    result["is_ng"] = False
                    result["ng_reason"] = None
                    result["bystander_issue"] = True
                    result["expression_score"] = max(0, int(result.get("expression_score", 5)) - 1)

        return result

    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {e}")
        return {"is_ng": True, "ng_reason": f"評価エラー: {str(e)}"}

# ================================
# CELL 12) Build Menna input from deduplicated candidates
# ================================
dedup_image_paths = collect_image_paths(DEDUP_INPUT_DIR)
dedup_folder_map = {p: "root" for p in dedup_image_paths}
print(f"Deduplicated candidate images for Menna stage: {len(dedup_image_paths)}")

# ================================
# CELL 13) Process all deduplicated candidates
# ================================
def process_all_photos(image_paths: List[str], folder_map: Dict[str, str]) -> pd.DataFrame:
    all_results = []

    for img_path in tqdm(image_paths, desc="Evaluating deduplicated candidates"):
        folder = folder_map.get(img_path, "unknown")
        eval_result = evaluate_photo(
            image_path=img_path,
            folder_name=folder,
            event_type=infer_event_type_from_folder(folder),
            target_grade=TARGET_GRADE,
        )

        result = {
            "file_name": os.path.basename(img_path),
            "path": img_path,
            "folder": folder,
            "event_type": infer_event_type_from_folder(folder),
            "is_ng": bool(eval_result.get("is_ng", False)),
            "ng_reason": eval_result.get("ng_reason"),
            "technical_score": eval_result.get("technical_score", 0),
            "expression_score": eval_result.get("expression_score", 0),
            "composition_score": eval_result.get("composition_score", 0),
            "rarity_score": eval_result.get("rarity_score", 0),
            "blur_detected": eval_result.get("blur_detected", False),
            "exposure_issue": eval_result.get("exposure_issue", "なし"),
            "eyes_closed": eval_result.get("eyes_closed", False),
            "protagonist_eyes_closed": eval_result.get("protagonist_eyes_closed", False),
            "protagonist_imperfect_moment": eval_result.get("protagonist_imperfect_moment", False),
            "bystander_issue": eval_result.get("bystander_issue", False),
            "face_visible": eval_result.get("face_visible", True),
            "short_comment": eval_result.get("short_comment", ""),
            "target_grade_relevance": eval_result.get("target_grade_relevance", 5),
            "event_relevance": eval_result.get("event_relevance", 5),
            "landmark_visible": eval_result.get("landmark_visible", False),
            "ball_visible": eval_result.get("ball_visible", False),
            "cheering_shot": eval_result.get("cheering_shot", False),
            "shot_type_tags": eval_result.get("shot_type_tags", []),
            "class_hint": eval_result.get("class_hint", "不明"),
            "day_hint": eval_result.get("day_hint", "不明"),
            "person_signature": eval_result.get("person_signature", "不明"),
            "has_burst": eval_result.get("has_burst", False),
            "has_similar_composition": eval_result.get("has_similar_composition", False),
            "has_duplicate_subject": eval_result.get("has_duplicate_subject", False),
        }
        all_results.append(result)
        time.sleep(5)

    df = pd.DataFrame(all_results)
    df["redundancy_penalty"] = (
        df["has_burst"].astype(int) * 0.15 +
        df["has_similar_composition"].astype(int) * 0.10 +
        df["has_duplicate_subject"].astype(int) * 0.20
    )

    df["quality_score"] = (
        df["technical_score"] * WEIGHT_TECHNICAL +
        df["expression_score"] * WEIGHT_EXPRESSION +
        df["composition_score"] * WEIGHT_COMPOSITION +
        df["rarity_score"] * WEIGHT_RARITY
    )

    df["total_score"] = (
        df["quality_score"] * 0.80 +
        df["event_relevance"] * 0.15 +
        df["target_grade_relevance"] * 0.05
    )
    df["total_score"] = df["total_score"] * (1 - df["redundancy_penalty"])
    df.loc[df["is_ng"] == True, "total_score"] = 0
    return df

df_results = process_all_photos(dedup_image_paths, dedup_folder_map)

# ================================
# CELL 14) Final Selection Logic
# ================================
def select_best_by_threshold(df: pd.DataFrame) -> pd.DataFrame:

    def row_has_any_tag(tags: List[str], terms: List[str]) -> bool:
        return any(any(term.lower() in t or t in term.lower() for term in terms) for t in tags)

    def is_transit_or_meal(tags: List[str]) -> bool:
        return row_has_any_tag(tags, [
            "transit", "train", "plane", "meal", "lunch", "food",
            "移動中", "電車", "飛行機", "食事", "昼食",
        ])

    work = df.copy()
    if work.empty:
        return work

    if "event_type" not in work.columns:
        work["event_type"] = work["folder"].apply(infer_event_type_from_folder)

    def select_for_one_group(group_df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
        local = group_df.copy()

        for col, default in {
            "shot_type_tags": [],
            "person_signature": "不明",
            "class_hint": "不明",
            "day_hint": "不明",
        }.items():
            if col not in local.columns:
                local[col] = [default for _ in range(len(local))]

        local["shot_type_tags_norm"] = local["shot_type_tags"].apply(normalize_tags)
        local["event_relevance"] = pd.to_numeric(local.get("event_relevance", 5), errors="coerce").fillna(5)
        local["target_grade_relevance"] = pd.to_numeric(local.get("target_grade_relevance", 5), errors="coerce").fillna(5)
        local["is_transit_or_meal"] = local["shot_type_tags_norm"].apply(is_transit_or_meal)

        df_ok = local[local["is_ng"] == False].copy()
        if df_ok.empty:
            return df_ok

        df_ok = df_ok.sort_values(["event_relevance", "target_grade_relevance", "total_score"], ascending=False)
        selected_idx = []

        person_counts = {}
        class_counts = {}
        scene_counts = {}
        composition_counts = {}
        burst_count = 0

        MAX_SAME_PERSON = MAX_REPEAT_SAME_PERSON
        MAX_SAME_CLASS = 3
        MAX_SAME_SCENE = 2
        MAX_SAME_COMPOSITION = 3
        MAX_BURST_ALLOWED = 1

        def get_class_key(row):
            return normalize_class_hint(row.get("class_hint", "不明"))

        def get_class_members(row):
            class_key = get_class_key(row)
            if class_key in ("unknown", "不明"):
                return []
            return class_key.split("_")[1:]

        def get_scene_key(row):
            tags = "_".join(sorted(row["shot_type_tags_norm"]))
            return f"{row['event_type']}_{tags}_{get_class_key(row)}"

        def get_composition_key(row):
            return tuple(sorted(row["shot_type_tags_norm"]))

        def get_person_sig(row):
            sig = str(row.get("person_signature", "不明")).strip().lower()
            if sig in ("unknown", "不明"):
                sig = f"class_{get_class_key(row)}_face_{row['face_visible']}"
            return sig

        def can_pick(row, strict=True):
            nonlocal burst_count
            sig = get_person_sig(row)
            class_members = get_class_members(row)
            scene_key = get_scene_key(row)
            comp_key = get_composition_key(row)

            if strict and person_counts.get(sig, 0) >= MAX_SAME_PERSON:
                return False
            if strict and any(class_counts.get(m, 0) >= MAX_SAME_CLASS for m in class_members):
                return False
            if strict and scene_counts.get(scene_key, 0) >= MAX_SAME_SCENE:
                return False
            if strict and composition_counts.get(comp_key, 0) >= MAX_SAME_COMPOSITION:
                return False
            if strict and row.get("has_burst", False):
                if burst_count >= MAX_BURST_ALLOWED:
                    return False

            max_tm = rules.get("max_transit_or_meal")
            if max_tm is not None and row.get("is_transit_or_meal", False):
                current_tm = sum(1 for i in selected_idx if df_ok.loc[i, "is_transit_or_meal"])
                if current_tm >= max_tm:
                    return False
            return True

        def add_pick(idx, reason, strict=True):
            nonlocal burst_count
            if idx in selected_idx:
                return False
            row = df_ok.loc[idx]
            if not can_pick(row, strict=strict):
                return False

            selected_idx.append(idx)
            sig = get_person_sig(row)
            class_members = get_class_members(row)
            scene_key = get_scene_key(row)
            comp_key = get_composition_key(row)

            person_counts[sig] = person_counts.get(sig, 0) + 1
            for m in class_members:
                class_counts[m] = class_counts.get(m, 0) + 1
            scene_counts[scene_key] = scene_counts.get(scene_key, 0) + 1
            composition_counts[comp_key] = composition_counts.get(comp_key, 0) + 1

            if row.get("has_burst", False):
                burst_count += 1

            df_ok.loc[idx, "selection_reason"] = reason
            return True

        for group in rules.get("required_tag_groups", []):
            candidates = df_ok[df_ok["shot_type_tags_norm"].apply(lambda tags: row_has_any_tag(tags, group))]
            for idx in candidates.index:
                if add_pick(idx, f"required: {'/'.join(group)}"):
                    break

        must_landmark = int(rules.get("must_have_landmark", 0) or 0)
        if must_landmark > 0:
            for idx in df_ok[df_ok["landmark_visible"] == True].index:
                if sum(df_ok.loc[i, "landmark_visible"] for i in selected_idx) >= must_landmark:
                    break
                add_pick(idx, "landmark")

        cheer_min, _ = rules.get("cheering_range", (0, 0))
        if cheer_min > 0:
            for idx in df_ok[df_ok["cheering_shot"] == True].index:
                if sum(df_ok.loc[i, "cheering_shot"] for i in selected_idx) >= cheer_min:
                    break
                add_pick(idx, "cheering")

        if rules.get("require_ball_visible", False):
            for idx in df_ok[df_ok["ball_visible"] == True].index:
                if any(df_ok.loc[i, "ball_visible"] for i in selected_idx):
                    break
                add_pick(idx, "ball")

        balance_col = None
        if rules.get("balance_by_class", False):
            balance_col = "class_key"
        elif rules.get("balance_by_day", False):
            balance_col = "day_hint"

        if balance_col:
            if balance_col == "class_key":
                local["class_key"] = local["class_hint"].apply(normalize_class_hint)
                df_ok["class_key"] = df_ok["class_hint"].apply(normalize_class_hint)
            grouped = {}
            for idx, row in df_ok.iterrows():
                key = str(row.get(balance_col, "不明"))
                grouped.setdefault(key, []).append(idx)
            for key in grouped:
                grouped[key].sort(key=lambda i: df_ok.loc[i, "total_score"], reverse=True)

            while len(selected_idx) < MAX_FINAL_PHOTOS:
                changed = False
                for key in sorted(grouped.keys(), key=lambda k: len(grouped[k])):
                    if not grouped[key]:
                        continue
                    idx = grouped[key].pop(0)
                    if add_pick(idx, f"balanced:{balance_col}"):
                        changed = True
                    if len(selected_idx) >= MAX_FINAL_PHOTOS:
                        break
                if not changed:
                    break

        for idx in df_ok.index:
            if len(selected_idx) >= MAX_FINAL_PHOTOS:
                break
            add_pick(idx, "top", strict=True)

        if len(selected_idx) < MIN_FINAL_PHOTOS:
            for idx in df_ok.index:
                if len(selected_idx) >= MIN_FINAL_PHOTOS:
                    break
                add_pick(idx, "fallback_relaxed", strict=False)

        if len(selected_idx) < MAX_FINAL_PHOTOS:
            for idx in df_ok.index:
                if len(selected_idx) >= MAX_FINAL_PHOTOS:
                    break
                add_pick(idx, "fill_to_max_relaxed", strict=False)

        selected_df = local.loc[selected_idx].copy() if selected_idx else df_ok.head(0).copy()
        if not selected_df.empty:
            selected_df = selected_df.sort_values("total_score", ascending=False)
        return selected_df

    selected_groups = []
    for folder_name, group_df in work.groupby("folder", dropna=False):
        inferred_event = infer_event_type_from_folder(folder_name)
        rules = get_event_rules(inferred_event)
        picked = select_for_one_group(group_df, rules)
        if not picked.empty:
            picked["event_type"] = inferred_event
            selected_groups.append(picked)

    if not selected_groups:
        print("No selectable non-NG photos found.")
        return work.head(0).copy()

    return pd.concat(selected_groups, ignore_index=False)

df_selected = select_best_by_threshold(df_results)
print(f"Selected final best shots: {len(df_selected)}")
display(df_selected)

# ================================
# CELL 15) Export final outputs
# ================================
def write_df_to_worksheet(ws, df: pd.DataFrame):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return
    def _excel_safe(value: Any) -> Any:
        if isinstance(value, (list, dict, tuple, set)):
            return json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
        return value
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append([_excel_safe(v) for v in r])

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

SELECTED_OUTPUT_DIR = os.path.join(FINAL_OUTPUT_DIR, f"{ts}_best_photos")
NG_OUTPUT_DIR = os.path.join(FINAL_OUTPUT_DIR, f"{ts}_ng_photos")
OTHERS_OUTPUT_DIR = os.path.join(FINAL_OUTPUT_DIR, f"{ts}_other_passing_photos")
for d in [SELECTED_OUTPUT_DIR, NG_OUTPUT_DIR, OTHERS_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

def ensure_event_output_dir(base_dir: str, event_type: str) -> str:
    event_dir = os.path.join(base_dir, normalize_event_type(event_type))
    os.makedirs(event_dir, exist_ok=True)
    return event_dir

df_ng = df_results[df_results["is_ng"] == True].copy()

rename_rows = []
if not df_selected.empty:
    df_selected = df_selected.sort_values(["event_type", "total_score"], ascending=[True, False])
    for _, row in df_selected.iterrows():
        src_path = row["path"]
        event_type = row.get("event_type", infer_event_type_from_folder(row.get("folder", "")))
        event_dir = ensure_event_output_dir(SELECTED_OUTPUT_DIR, event_type)
        original_name = row["file_name"]
        stem, ext = os.path.splitext(original_name)
        date_prefix = get_file_date_string(src_path)
        new_name = f"{date_prefix}_{stem}_{BEST_TOKEN}_{int(row['total_score'])}{ext}"
        if os.path.exists(src_path):
            dst_path = unique_path(os.path.join(event_dir, new_name))
            shutil.copy2(src_path, dst_path)
            rename_rows.append({
                "folder": row["folder"],
                "event_type": normalize_event_type(event_type),
                "original_file_name": original_name,
                "new_file_name": os.path.basename(dst_path),
                "total_score": row["total_score"],
                "selection_reason": row.get("selection_reason", ""),
            })

if not df_ng.empty:
    for _, row in df_ng.iterrows():
        src_path = row["path"]
        event_dir = ensure_event_output_dir(NG_OUTPUT_DIR, row.get("event_type", infer_event_type_from_folder(row.get("folder", ""))))
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(event_dir, row["file_name"]))

selected_paths = set(df_selected["path"].tolist()) if not df_selected.empty else set()
df_others = df_results[(df_results["is_ng"] == False) & (~df_results["path"].isin(selected_paths))].copy()

if not df_others.empty:
    for _, row in df_others.iterrows():
        src_path = row["path"]
        event_dir = ensure_event_output_dir(OTHERS_OUTPUT_DIR, row.get("event_type", infer_event_type_from_folder(row.get("folder", ""))))
        if os.path.exists(src_path):
            shutil.copy2(src_path, os.path.join(event_dir, row["file_name"]))

pipeline_summary_df = pd.DataFrame([{
    "raw_input_count": len(records),
    "deduplicated_candidate_count": len(dedup_image_paths),
    "final_selected_count": len(df_selected),
    "ng_count_after_menna": int((df_results["is_ng"] == True).sum()) if not df_results.empty else 0,
    "other_passing_count": len(df_others),
    "dedup_reduction_rate": round(1 - len(dedup_image_paths) / max(len(records), 1), 4),
}])

excel_path = os.path.join(FINAL_OUTPUT_DIR, f"{ts}_snap_pipeline_results.xlsx")
wb = Workbook()
wb.remove(wb.active)

sheets = [
    ("Pipeline_Summary", pipeline_summary_df),
    ("Similarity_Summary", df_sim_summary),
    ("Similarity_Clusters", df_sim_cluster),
    ("Similarity_Image_Level", df_sim_image),
    ("Scored_All_Candidates", df_results),
    ("NG_Log", df_ng),
    ("Selected", df_selected),
    ("Others", df_others),
    ("Rename_Map", pd.DataFrame(rename_rows)),
]
for name, df in sheets:
    write_df_to_worksheet(wb.create_sheet(title=name[:31]), df)

wb.save(excel_path)

print("Integrated pipeline complete.")
print("Excel:", excel_path)
print("Selected:", SELECTED_OUTPUT_DIR)
print("NG:", NG_OUTPUT_DIR)
print("Others:", OTHERS_OUTPUT_DIR)
print("Similarity summary:", sim_summary_csv)
print("Deduplicated candidate folder:", DEDUP_INPUT_DIR)

# Optional downloads
# files.download(excel_path)
