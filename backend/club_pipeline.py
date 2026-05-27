from __future__ import annotations

import base64
import io
import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openpyxl import Workbook
from PIL import ExifTags, Image

from backend.excel_labels import CLUB_SHEET_LABELS, excel_label, translate_display_value

try:
    from insightface.app import FaceAnalysis
except Exception:
    FaceAnalysis = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", os.getenv("MODEL_VISION", "gpt-5.4"))
RANK_TOKEN = "本"
RANK_PAD = 2
FACE_PADDING_PCT = 0.25
GPT_MIN_FACE_PX = 40
GPT_RETRY = 2
COLOR_CLOSED = (0, 0, 210)
COLOR_OPEN = (0, 180, 60)
COLOR_NO_FACE = (0, 200, 220)

PHOTO_EVAL_PROMPT = """
You are a professional photo editor selecting the best shot for a graduation album.
Return STRICT JSON only.
JSON keys must stay English.
ng_reason and short_comment values MUST be Japanese.

<<<<<<< codex/connect-face_offset_calculator-to-individual-workflow-tzcsdq
Latest client pose rule override:
- Thumbs up is acceptable and must NOT be NG.
- Middle finger is NG.
- Obscene/improper poses are NG.
=======
Important pose rule:
- Thumbs up is acceptable and must NOT be treated as NG.
- NG hand/pose examples: middle finger, obscene/improper gestures, sexual/improper posing.
- JSON keys must remain English.
- ng_reason and short_comment values MUST be Japanese.
>>>>>>> MayFourthWeek

{
  "formality_score": 0-10,
  "beauty_score": 0-10,
  "expression_score": 0-10,
  "emotion_score": 0-10,
  "people_count_score": 0-10,
  "has_hand_gesture": true|false,
  "has_exaggerated_expression": true|false,
  "eyes_closed": true|false,
  "face_obscured": true|false,
  "subject_falling": true|false,
  "eating_or_mouth_full": true|false,
  "strange_obscene_posing": true|false,
  "bad_pose": true|false,
  "visible_underwear": true|false,
  "severe_framing_issue": true|false,
  "gesture_expression_penalty": 0-5,
  "is_ng": true|false,
<<<<<<< codex/connect-face_offset_calculator-to-individual-workflow-tzcsdq
  "ng_reason": "日本語",
  "short_comment": "日本語"
=======
  "ng_reason": "日本語の短い理由",
  "short_comment": "日本語の短いコメント"
>>>>>>> MayFourthWeek
}
""".strip()

EYE_PROMPT = """
You are examining a CROPPED IMAGE of a single FACE.
Return ONLY JSON:
{"has_clear_face": true|false, "eyes_closed": true|false, "confidence": "high"|"medium"|"low"}

Rules:
- If eye region is not visible or too occluded, has_clear_face=false.
- smiling squint is NOT closed eyes if iris/pupil is visible.
- For East Asian faces, narrow eyes can still be open.
- If has_clear_face is false, eyes_closed must be false.
""".strip()


@dataclass
class EvalRow:
    club: str
    file_name: str
    path: str
    group_index: int
    data: dict[str, Any]
    total_score: float = 0.0
    rank: int = 0
    new_file_name: str = ""


def _client():
    key = os.getenv("OPENAI_API_KEY")
    if not key or OpenAI is None:
        return None
    return OpenAI(api_key=key)


def _images_by_club(root: Path) -> list[tuple[str, list[Path]]]:
    out = []
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        imgs = sorted([p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        if imgs:
            out.append((d.name, imgs))
    return out


def _date_str(path: Path) -> str:
    try:
        exif = Image.open(path).getexif()
        for tag_id, value in (exif or {}).items():
            if ExifTags.TAGS.get(tag_id, tag_id) == "DateTimeOriginal" and value:
                return str(value)[:10].replace(":", "")
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d")


def _extract_json(text: str) -> dict[str, Any]:
    t = (text or "").strip()
    s, e = t.find("{"), t.rfind("}")
    if s == -1 or e == -1:
        return {}
    return json.loads(t[s : e + 1])


def _eval_photo(client: Any, image_path: Path, people_count: int) -> dict[str, Any]:
    if client is None:
        return {"is_ng": True, "ng_reason": "評価不能", "short_comment": "OpenAI未設定"}
    try:
        img = Image.open(image_path).convert("RGB")
        b = io.BytesIO()
        img.save(b, format="JPEG", quality=93)
        b64 = base64.b64encode(b.getvalue()).decode("utf-8")
        resp = client.responses.create(
            model=MODEL_VISION,
            input=[{"role": "user", "content": [{"type": "input_text", "text": PHOTO_EVAL_PROMPT + f"\nDetected people count: {people_count}"}, {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}", "detail": "high"}]}],
            reasoning={"effort": "medium"},
        )
        return _extract_json(resp.output_text)
    except Exception as e:
        return {"is_ng": True, "ng_reason": f"評価エラー: {e}", "short_comment": "評価に失敗しました"}


<<<<<<< codex/connect-face_offset_calculator-to-individual-workflow-tzcsdq
def _ask_eye(client: Any, crop: np.ndarray) -> tuple[bool, bool, str]:
    h, w = crop.shape[:2]
    if client is None or w < GPT_MIN_FACE_PX or h < GPT_MIN_FACE_PX:
        return False, False, "low"
    ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 93])
    if not ok:
        return False, False, "low"
    b64 = base64.b64encode(buf).decode("utf-8")
    for _ in range(GPT_RETRY):
        try:
            resp = client.responses.create(
                model=MODEL_VISION,
                input=[{"role": "user", "content": [{"type": "input_text", "text": EYE_PROMPT}, {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}", "detail": "high"}]}],
                reasoning={"effort": "medium"},
            )
            d = _extract_json(resp.output_text)
            return bool(d.get("has_clear_face", False)), bool(d.get("eyes_closed", False)), str(d.get("confidence", "medium"))
        except Exception:
            continue
    return False, False, "low"


def _score(row: dict[str, Any]) -> float:
    s = (
        float(row.get("formality_score", 0)) * 0.30
        + float(row.get("beauty_score", 0)) * 0.25
        + float(row.get("expression_score", 0)) * 0.15
        + float(row.get("emotion_score", 0)) * 0.10
        + float(row.get("people_count_score", 0)) * 0.20
    )
    penalty = float(row.get("gesture_expression_penalty", 0))
    return round(s - penalty, 3)
=======
def _fallback_eval(person_count:int, closed_count:int)->dict[str,Any]:
    return {"formality_score":max(0,8-closed_count),"beauty_score":7,"expression_score":max(0,8-closed_count),"emotion_score":7,"people_count_score":min(10, max(1,person_count)),"has_hand_gesture":False,"has_exaggerated_expression":False,"eyes_closed":closed_count>0,"face_obscured":person_count==0,"subject_falling":False,"eating_or_mouth_full":False,"strange_obscene_posing":False,"bad_pose":False,"visible_underwear":False,"severe_framing_issue":False,"gesture_expression_penalty":0,"is_ng":person_count==0,"ng_reason":"顔検出なし" if person_count==0 else "問題なし","short_comment":"自動評価（簡易）"}


def _score(eval_data: dict[str, Any], closed_count: int)->tuple[float,bool,str]:
    w = (float(eval_data.get("formality_score",0))*0.30 + float(eval_data.get("beauty_score",0))*0.25 + float(eval_data.get("expression_score",0))*0.15 + float(eval_data.get("emotion_score",0))*0.10 + float(eval_data.get("people_count_score",0))*0.20)
    total = w - float(eval_data.get("gesture_expression_penalty",0)) - (closed_count*0.75)
    ng = bool(eval_data.get("is_ng", False)) or bool(eval_data.get("eyes_closed", False) and closed_count>0) or bool(eval_data.get("strange_obscene_posing", False))
    reason = str(eval_data.get("ng_reason", ""))
    return round(total,3), ng, reason
>>>>>>> MayFourthWeek


def run_club_pipeline(input_zip_path: str, output_dir: str) -> dict[str, Any]:
    out = Path(output_dir)
    extracted = out / "extracted"
    club_out = out / "Club_Output"
    ranked = club_out / "ranked_photos"
    ranked_marked = club_out / "ranked_photos_marked"
    clean = club_out / "clean_images"
    marked = club_out / "marked_images"
    ng_root = club_out / "ng_photos"
    if out.exists():
        shutil.rmtree(out)
    for d in [extracted, ranked, ranked_marked, clean, marked, ng_root]:
        d.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(input_zip_path).extractall(extracted)

    face_app = None
    if FaceAnalysis is not None:
        try:
<<<<<<< codex/connect-face_offset_calculator-to-individual-workflow-tzcsdq
            face_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=["CPUExecutionProvider"])
            face_app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.35)
        except Exception:
            face_app = None
    client = _client()
    eval_rows: list[EvalRow] = []
    face_detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    rename_rows: list[dict[str, Any]] = []

    clubs = _images_by_club(extracted)
    for group_index, (club, imgs) in enumerate(clubs, 1):
        club_eval: list[EvalRow] = []
        copied = 0
        copy_failed: list[str] = []
        for p in imgs:
            bgr = cv2.imread(str(p))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            faces = face_app.get(rgb) if face_app is not None else []
            annotated = bgr.copy()
            clear_faces = 0
            closed_faces = 0
            for idx, face in enumerate(faces, 1):
                x1, y1, x2, y2 = map(int, face.bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(bgr.shape[1], x2), min(bgr.shape[0], y2)
                pw, ph = int((x2 - x1) * FACE_PADDING_PCT), int((y2 - y1) * FACE_PADDING_PCT)
                px1, py1 = max(0, x1 - pw), max(0, y1 - ph)
                px2, py2 = min(bgr.shape[1], x2 + pw), min(bgr.shape[0], y2 + ph)
                crop = bgr[py1:py2, px1:px2]
                has_clear_face, eyes_closed, conf = _ask_eye(client, crop)
                face_detail_rows.append({"file_name": p.name, "face_index": idx, "has_clear_face": has_clear_face, "eyes_closed": eyes_closed, "confidence": conf, "bbox": f"{x1},{y1},{x2},{y2}"})
                if has_clear_face:
                    clear_faces += 1
                    if eyes_closed:
                        closed_faces += 1
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_CLOSED, 2)
                    else:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_OPEN, 2)
                else:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_NO_FACE, 1)
            cv2.putText(annotated, f"People Detected : {len(faces)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(annotated, f"Clear Faces     : {clear_faces}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_OPEN, 2)
            cv2.putText(annotated, f"Eyes Closed     : {closed_faces}", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_CLOSED, 2)

            (clean / club).mkdir(parents=True, exist_ok=True)
            (marked / club).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(clean / club / p.name), bgr)
            cv2.imwrite(str(marked / club / p.name), annotated)

            ev = _eval_photo(client, p, clear_faces)
            ev["people_count_score"] = ev.get("people_count_score", min(10, max(1, clear_faces)))
            row = EvalRow(club=club, file_name=p.name, path=str(p), group_index=group_index, data=ev)
            row.total_score = _score(ev)
            club_eval.append(row)

        rankable = [r for r in club_eval if not bool(r.data.get("is_ng", False))]
        rankable.sort(key=lambda x: x.total_score, reverse=True)
        for i, r in enumerate(rankable, 1):
            r.rank = i
        for r in club_eval:
            if r.rank == 0:
                r.rank = len(rankable) + 1
            date_prefix = _date_str(Path(r.path))
            stem, ext = os.path.splitext(r.file_name)
            r.new_file_name = f"{date_prefix}_{stem}_{RANK_TOKEN}{r.rank:0{RANK_PAD}d}{ext.lower()}"
            (ranked / club).mkdir(parents=True, exist_ok=True)
            (ranked_marked / club).mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(clean / club / r.file_name, ranked / club / r.new_file_name)
                shutil.copy2(marked / club / r.file_name, ranked_marked / club / r.new_file_name)
                copied += 1
            except Exception:
                copy_failed.append(r.file_name)
            if bool(r.data.get("is_ng", False)):
                (ng_root / club).mkdir(parents=True, exist_ok=True)
                shutil.copy2(clean / club / r.file_name, ng_root / club / r.file_name)
            rename_rows.append({"group_index": group_index, "club": club, "rank": r.rank, "original_file_name": r.file_name, "original_path": r.path, "new_file_name": r.new_file_name, "total_score": r.total_score})
            eval_rows.append(r)

        summary_rows.append({"group_index": group_index, "club": club, "input_count": len(club_eval), "ranked_count": len(rankable), "ng_count": len([x for x in club_eval if bool(x.data.get("is_ng", False))]), "copied_count": copied, "best_file_name": rankable[0].file_name if rankable else "", "best_reason": (rankable[0].data.get("short_comment") if rankable else ""), "copy_failures": " | ".join(copy_failed)})

    excel_path = club_out / "club_result.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = CLUB_SHEET_LABELS["Summary"]
    summary_cols = ["group_index", "club", "input_count", "ranked_count", "ng_count", "copied_count", "best_file_name", "best_reason", "copy_failures"]
    ws.append([excel_label(c) for c in summary_cols])
    for row in summary_rows:
        ws.append([translate_display_value(row.get(c)) for c in summary_cols])
    rename_ws = wb.create_sheet(CLUB_SHEET_LABELS["Rename Output"])
    rename_cols = ["group_index", "club", "rank", "original_file_name", "original_path", "new_file_name", "total_score"]
    rename_ws.append([excel_label(c) for c in rename_cols])
    for row in rename_rows:
        rename_ws.append([translate_display_value(row.get(c)) for c in rename_cols])
    ranking_cols = ["formality_score","beauty_score","expression_score","emotion_score","people_count_score","has_hand_gesture","has_exaggerated_expression","eyes_closed","face_obscured","subject_falling","eating_or_mouth_full","strange_obscene_posing","bad_pose","visible_underwear","severe_framing_issue","gesture_expression_penalty","is_ng","ng_reason","short_comment","file_name","path","group_index","total_score","rank"]
    ranking_ws = wb.create_sheet(CLUB_SHEET_LABELS["Best Shot Ranking"])
    ranking_ws.append([excel_label(c) for c in ranking_cols])
    all_ws = wb.create_sheet(CLUB_SHEET_LABELS["All Evals"])
    all_ws.append([excel_label(c) for c in ranking_cols])
    ng_ws = wb.create_sheet(CLUB_SHEET_LABELS["NG Photos"])
    ng_ws.append([excel_label(c) for c in ranking_cols])
    for r in sorted(eval_rows, key=lambda x: (x.group_index, x.rank, x.file_name)):
        row = [r.data.get("formality_score"),r.data.get("beauty_score"),r.data.get("expression_score"),r.data.get("emotion_score"),r.data.get("people_count_score"),r.data.get("has_hand_gesture"),r.data.get("has_exaggerated_expression"),r.data.get("eyes_closed"),r.data.get("face_obscured"),r.data.get("subject_falling"),r.data.get("eating_or_mouth_full"),r.data.get("strange_obscene_posing"),r.data.get("bad_pose"),r.data.get("visible_underwear"),r.data.get("severe_framing_issue"),r.data.get("gesture_expression_penalty"),r.data.get("is_ng"),r.data.get("ng_reason"),r.data.get("short_comment"),r.file_name,r.path,r.group_index,r.total_score,r.rank]
        out_row = [translate_display_value(v) for v in row]
        ranking_ws.append(out_row)
        all_ws.append(out_row)
        if bool(r.data.get("is_ng", False)):
            ng_ws.append(out_row)
    eye_sum = wb.create_sheet(CLUB_SHEET_LABELS["Eye Closure Summary"])
    eye_sum.append([excel_label("file_name"), "faces_detected", "clear_faces", "closed_eye_faces"])
    # aggregate from face_detail_rows
    agg: dict[str, dict[str, int]] = {}
    for d in face_detail_rows:
        a = agg.setdefault(d["file_name"], {"faces": 0, "clear": 0, "closed": 0})
        a["faces"] += 1
        if d["has_clear_face"]:
            a["clear"] += 1
        if d["eyes_closed"]:
            a["closed"] += 1
    for fn, a in sorted(agg.items()):
        eye_sum.append([fn, a["faces"], a["clear"], a["closed"]])
    face_ws = wb.create_sheet(CLUB_SHEET_LABELS["Face Detail"])
    face_cols = ["file_name", "face_index", "has_clear_face", "eyes_closed", "confidence", "bbox"]
    face_ws.append(face_cols)
    for d in face_detail_rows:
        face_ws.append([translate_display_value(d.get(c)) for c in face_cols])
    wb.save(excel_path)
    (club_out / "club_result.json").write_text(json.dumps({"summary": {"club_count": len(clubs), "photo_count": len(eval_rows)}}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "completed", "summary": {"club_count": len(clubs), "photo_count": len(eval_rows)}, "excel_path": str(excel_path), "json_path": str(club_out / "club_result.json"), "output_dir": str(club_out)}
=======
            face_app=FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"]); face_app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception: face_app=None
    client=_get_client(); results=[]
    marked_outputs: dict[tuple[str, str], Path] = {}
    for club,images in clubs.items():
        for p in images:
            img=cv2.imread(str(p));
            if img is None: continue
            pc,cc,fd,vis=_analyze_faces(img,face_app)
            (marked/club).mkdir(parents=True,exist_ok=True); (clean/club).mkdir(parents=True,exist_ok=True)
            marked_path = marked / club / p.name
            clean_path = clean / club / p.name
            cv2.imwrite(str(marked_path), vis)
            cv2.imwrite(str(clean_path), img)
            marked_outputs[(club, p.name)] = marked_path
            ev=_evaluate_with_gpt(client,p,pc) if client else None
            if not ev: ev=_fallback_eval(pc,cc)
            total,ng,reason=_score(ev,cc)
            results.append(ClubImageResult(club,p,_shooting_date(p),pc,cc>0,cc,fd,ev,ng,reason,total))
    by={}
    for r in results: by.setdefault(r.club_name,[]).append(r)
    summary_rows = []
    rename_rows = []
    for group_index, (club,rows) in enumerate(by.items(), 1):
        rows.sort(key=lambda x:(x.ng_flag,-x.total_score))
        copied_count = 0
        copy_failures = 0
        for i,r in enumerate(rows,1):
            r.rank=i; r.renamed=f"{club}_{r.shooting_date}_{r.path.stem}_{RANK_TOKEN}{i:02d}{r.path.suffix.lower()}"
            (ranked/club).mkdir(parents=True,exist_ok=True); (ranked_marked/club).mkdir(parents=True,exist_ok=True)
            try:
                shutil.copy2(clean / club / r.path.name, ranked / club / r.renamed); copied_count += 1
            except Exception:
                copy_failures += 1
            marked_source = marked_outputs.get((club, r.path.name), marked / club / r.path.name)
            try:
                shutil.copy2(marked_source, ranked_marked / club / r.renamed)
            except Exception:
                copy_failures += 1
            if r.ng_flag:
                (ng_root/club).mkdir(parents=True,exist_ok=True); shutil.copy2(clean/club/r.path.name, ng_root/club/r.path.name)
            rename_rows.append({
                "group_index": group_index, "club": club, "rank": r.rank, "original_file_name": r.path.name,
                "original_path": str(r.path), "new_file_name": r.renamed, "total_score": r.total_score
            })
        best = rows[0] if rows else None
        summary_rows.append({
            "group_index": group_index, "club": club, "input_count": len(rows), "ranked_count": len(rows),
            "ng_count": len([x for x in rows if x.ng_flag]), "copied_count": copied_count,
            "best_file_name": best.path.name if best else "", "best_reason": translate_display_value((best.eval_data or {}).get("short_comment", "")) if best else "", "copy_failures": copy_failures
        })

    excel=club_out/"club_result.xlsx"; jsonp=club_out/"club_result.json"
    wb=Workbook(); ws=wb.active; ws.title=CLUB_SHEET_LABELS["Summary"]
    ws.append([excel_label(k) for k in ["group_index","club","input_count","ranked_count","ng_count","copied_count","best_file_name","best_reason","copy_failures"]])
    for row in summary_rows:
        ws.append([translate_display_value(row.get(k)) for k in ["group_index","club","input_count","ranked_count","ng_count","copied_count","best_file_name","best_reason","copy_failures"]])
    nws=wb.create_sheet(CLUB_SHEET_LABELS["Rename Output"])
    nws.append([excel_label(k) for k in ["group_index","club","rank","original_file_name","original_path","new_file_name","total_score"]])
    for row in rename_rows:
        nws.append([translate_display_value(row.get(k)) for k in ["group_index","club","rank","original_file_name","original_path","new_file_name","total_score"]])
    rws=wb.create_sheet(CLUB_SHEET_LABELS["Best Shot Ranking"])
    ranking_cols = ["formality_score","beauty_score","expression_score","emotion_score","people_count_score","has_hand_gesture","has_exaggerated_expression","eyes_closed","face_obscured","subject_falling","eating_or_mouth_full","strange_obscene_posing","bad_pose","visible_underwear","severe_framing_issue","gesture_expression_penalty","is_ng","ng_reason","short_comment","file_name","path","group_index","total_score","rank"]
    rws.append([excel_label(c) for c in ranking_cols])
    for r in sorted(results,key=lambda x:(x.club_name,x.rank)):
        ev=r.eval_data
        group_index = list(by.keys()).index(r.club_name) + 1
        row = [ev.get("formality_score"),ev.get("beauty_score"),ev.get("expression_score"),ev.get("emotion_score"),ev.get("people_count_score"),ev.get("has_hand_gesture"),ev.get("has_exaggerated_expression"),ev.get("eyes_closed"),ev.get("face_obscured"),ev.get("subject_falling"),ev.get("eating_or_mouth_full"),ev.get("strange_obscene_posing"),ev.get("bad_pose"),ev.get("visible_underwear"),ev.get("severe_framing_issue"),ev.get("gesture_expression_penalty"),r.ng_flag,r.ng_reason,ev.get("short_comment"),r.path.name,str(r.path),group_index,r.total_score,r.rank]
        rws.append([translate_display_value(v) for v in row])
    aws=wb.create_sheet(CLUB_SHEET_LABELS["All Evals"])
    aws.append([excel_label(c) for c in ranking_cols])
    for row in rws.iter_rows(min_row=2, values_only=True):
        aws.append(list(row))
    ngs=wb.create_sheet(CLUB_SHEET_LABELS["NG Photos"])
    ngs.append([excel_label(c) for c in ranking_cols])
    for r in sorted(results,key=lambda x:(x.club_name,x.rank)):
        if r.ng_flag:
            ev = r.eval_data
            group_index = list(by.keys()).index(r.club_name) + 1
            row = [ev.get("formality_score"),ev.get("beauty_score"),ev.get("expression_score"),ev.get("emotion_score"),ev.get("people_count_score"),ev.get("has_hand_gesture"),ev.get("has_exaggerated_expression"),ev.get("eyes_closed"),ev.get("face_obscured"),ev.get("subject_falling"),ev.get("eating_or_mouth_full"),ev.get("strange_obscene_posing"),ev.get("bad_pose"),ev.get("visible_underwear"),ev.get("severe_framing_issue"),ev.get("gesture_expression_penalty"),r.ng_flag,r.ng_reason,ev.get("short_comment"),r.path.name,str(r.path),group_index,r.total_score,r.rank]
            ngs.append([translate_display_value(v) for v in row])
    ews=wb.create_sheet(CLUB_SHEET_LABELS["Eye Closure Summary"]); ews.append([excel_label("club"),excel_label("file_name"),excel_label("person_count"),excel_label("closed_eye_faces"),excel_label("eyes_closed_photo")])
    for r in results: ews.append([r.club_name,r.path.name,r.person_count,r.closed_eye_face_count,translate_display_value(r.eyes_closed_photo)])
    fws=wb.create_sheet(CLUB_SHEET_LABELS["Face Detail"]); fws.append([excel_label("club"),excel_label("file_name"),excel_label("face_index"),excel_label("bbox"),excel_label("left_eye_ratio"),excel_label("right_eye_ratio"),excel_label("eye_closed")])
    for r in results:
        for f in r.face_details: fws.append([r.club_name,r.path.name,f.face_index,str(f.bbox),f.left_eye_ratio,f.right_eye_ratio,translate_display_value(f.eye_closed)])
    wb.save(excel)
    jsonp.write_text(json.dumps({"summary":{"club_count":len(by),"photo_count":len(results)},"items":[{"club_name":r.club_name,"original_file":r.path.name,"rank":r.rank,"renamed_file":r.renamed,"ng_flag":r.ng_flag,"ng_reason":r.ng_reason,"total_score":r.total_score,"evaluation":r.eval_data} for r in sorted(results,key=lambda x:(x.club_name,x.rank))]},ensure_ascii=False,indent=2),encoding="utf-8")
    return {"status":"completed","summary":{"club_count":len(by),"photo_count":len(results)},"excel_path":str(excel),"json_path":str(jsonp),"output_dir":str(club_out)}
>>>>>>> MayFourthWeek
