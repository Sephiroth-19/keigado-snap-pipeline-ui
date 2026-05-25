from __future__ import annotations

import base64
import io
import json
import os
import shutil
import zipfile
from dataclasses import dataclass, field
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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
RANK_TOKEN = "本"
RANK_PAD = 2
MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", os.getenv("MODEL_VISION", "gpt-5.4"))

PHOTO_EVAL_PROMPT = """
You are a professional photo editor selecting the best shot for a school club album.
Return STRICT JSON only.

Important pose rule:
- Thumbs up is acceptable and must NOT be treated as NG.
- NG hand/pose examples: middle finger, obscene/improper gestures, sexual/improper posing.

JSON schema:
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
  "ng_reason": "short reason",
  "short_comment": "short comment"
}
""".strip()

@dataclass
class FaceDetail:
    face_index: int
    bbox: tuple[int, int, int, int]
    left_eye_ratio: float
    right_eye_ratio: float
    eye_closed: bool


@dataclass
class ClubImageResult:
    club_name: str
    path: Path
    shooting_date: str
    person_count: int
    eyes_closed_photo: bool
    closed_eye_face_count: int
    face_details: list[FaceDetail] = field(default_factory=list)
    eval_data: dict[str, Any] = field(default_factory=dict)
    ng_flag: bool = False
    ng_reason: str = ""
    total_score: float = 0.0
    rank: int = 0
    renamed: str = ""


def _get_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key or OpenAI is None:
        return None
    return OpenAI(api_key=key)


def _collect_club_images(extracted_root: Path):
    out = {}
    for club_dir in sorted([p for p in extracted_root.iterdir() if p.is_dir()]):
        imgs = [p for p in sorted(club_dir.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if imgs:
            out[club_dir.name] = imgs
    return out


def _shooting_date(path: Path) -> str:
    try:
        exif = Image.open(path).getexif()
        if exif:
            for tag_id, value in exif.items():
                if ExifTags.TAGS.get(tag_id, tag_id) == "DateTimeOriginal" and value:
                    return str(value)[:10].replace(":", "")
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y%m%d")


def _analyze_faces(image_bgr: np.ndarray, face_app: Any | None):
    vis = image_bgr.copy(); details=[]
    if face_app is not None:
        faces = face_app.get(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        for i,f in enumerate(faces,1):
            x1,y1,x2,y2=[int(v) for v in f.bbox]
            kps = f.kps if hasattr(f, "kps") else None
            closed=False
            lr=rr=0.0
            if kps is not None and len(kps)>=2:
                lr=rr=0.3
            details.append(FaceDetail(i,(x1,y1,x2,y2),lr,rr,closed))
            cv2.rectangle(vis,(x1,y1),(x2,y2),(0,180,0),2)
    else:
        g=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        for i,(x,y,w,h) in enumerate(face.detectMultiScale(g,1.15,5,minSize=(50,50)),1):
            closed = len(eye.detectMultiScale(g[y:y+h, x:x+w],1.1,4,minSize=(10,10)))==0
            details.append(FaceDetail(i,(x,y,x+w,y+h),0.0,0.0,closed))
            cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255) if closed else (0,180,0),2)
    closed_count=sum(1 for d in details if d.eye_closed)
    cv2.putText(vis, f"People:{len(details)} ClosedEyes:{closed_count}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
    return len(details), closed_count, details, vis


def _parse_json_obj(text: str) -> dict[str, Any]:
    t=(text or "").strip(); s=t.find("{"); e=t.rfind("}")
    if s==-1 or e==-1: return {}
    return json.loads(t[s:e+1])


def _evaluate_with_gpt(client: Any, image_path: Path, person_count: int) -> dict[str, Any] | None:
    try:
        img=Image.open(image_path).convert("RGB")
        b=io.BytesIO(); img.save(b, format="JPEG", quality=92)
        b64=base64.b64encode(b.getvalue()).decode("utf-8")
        resp=client.responses.create(model=MODEL_VISION,input=[{"role":"user","content":[{"type":"input_text","text":PHOTO_EVAL_PROMPT+f"\nDetected people count: {person_count}"},{"type":"input_image","image_url":f"data:image/jpeg;base64,{b64}","detail":"high"}]}])
        return _parse_json_obj(resp.output_text)
    except Exception:
        return None


def _fallback_eval(person_count:int, closed_count:int)->dict[str,Any]:
    return {"formality_score":max(0,8-closed_count),"beauty_score":7,"expression_score":max(0,8-closed_count),"emotion_score":7,"people_count_score":min(10, max(1,person_count)),"has_hand_gesture":False,"has_exaggerated_expression":False,"eyes_closed":closed_count>0,"face_obscured":person_count==0,"subject_falling":False,"eating_or_mouth_full":False,"strange_obscene_posing":False,"bad_pose":False,"visible_underwear":False,"severe_framing_issue":False,"gesture_expression_penalty":0,"is_ng":person_count==0,"ng_reason":"no face detected" if person_count==0 else "","short_comment":"heuristic"}


def _score(eval_data: dict[str, Any], closed_count: int)->tuple[float,bool,str]:
    w = (float(eval_data.get("formality_score",0))*0.30 + float(eval_data.get("beauty_score",0))*0.25 + float(eval_data.get("expression_score",0))*0.15 + float(eval_data.get("emotion_score",0))*0.10 + float(eval_data.get("people_count_score",0))*0.20)
    total = w - float(eval_data.get("gesture_expression_penalty",0)) - (closed_count*0.75)
    ng = bool(eval_data.get("is_ng", False)) or bool(eval_data.get("eyes_closed", False) and closed_count>0)
    reason = str(eval_data.get("ng_reason", ""))
    return round(total,3), ng, reason


def run_club_pipeline(input_zip_path: str, output_dir: str) -> dict[str, Any]:
    out=Path(output_dir); extracted=out/"extracted"; club_out=out/"Club_Output"
    ranked=club_out/"ranked_photos"; ranked_marked=club_out/"ranked_photos_marked"; ng_root=club_out/"ng_photos"; marked=club_out/"marked_images"; clean=club_out/"clean_images"
    if out.exists(): shutil.rmtree(out)
    for d in [extracted,ranked,ranked_marked,ng_root,marked,clean]: d.mkdir(parents=True,exist_ok=True)
    zipfile.ZipFile(input_zip_path).extractall(extracted)
    clubs=_collect_club_images(extracted)
    face_app=None
    if FaceAnalysis is not None:
        try:
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
    for club,rows in by.items():
        rows.sort(key=lambda x:(x.ng_flag,-x.total_score))
        for i,r in enumerate(rows,1):
            r.rank=i; r.renamed=f"{club}_{r.shooting_date}_{r.path.stem}_{RANK_TOKEN}{i:02d}{r.path.suffix.lower()}"
            (ranked/club).mkdir(parents=True,exist_ok=True); (ranked_marked/club).mkdir(parents=True,exist_ok=True)
            shutil.copy2(clean / club / r.path.name, ranked / club / r.renamed)
            marked_source = marked_outputs.get((club, r.path.name), marked / club / r.path.name)
            shutil.copy2(marked_source, ranked_marked / club / r.renamed)
            if r.ng_flag:
                (ng_root/club).mkdir(parents=True,exist_ok=True); shutil.copy2(clean/club/r.path.name, ng_root/club/r.path.name)

    excel=club_out/"club_result.xlsx"; jsonp=club_out/"club_result.json"
    wb=Workbook(); ws=wb.active; ws.title=CLUB_SHEET_LABELS["Summary"]
    ws.append([excel_label("club_count"),excel_label("photo_count"),excel_label("closed_eye_photo_count"),excel_label("closed_eye_face_count"),excel_label("ranked_output_count")])
    ws.append([len(by),len(results),sum(1 for r in results if r.eyes_closed_photo),sum(r.closed_eye_face_count for r in results),len(results)])
    ews=wb.create_sheet(CLUB_SHEET_LABELS["Eye Closure Summary"]); ews.append([excel_label("club"),excel_label("file_name"),excel_label("person_count"),excel_label("closed_eye_faces"),excel_label("eyes_closed_photo")])
    for r in results: ews.append([r.club_name,r.path.name,r.person_count,r.closed_eye_face_count,translate_display_value(r.eyes_closed_photo)])
    fws=wb.create_sheet(CLUB_SHEET_LABELS["Face Detail"]); fws.append([excel_label("club"),excel_label("file_name"),excel_label("face_index"),excel_label("bbox"),excel_label("left_eye_ratio"),excel_label("right_eye_ratio"),excel_label("eye_closed")])
    for r in results:
        for f in r.face_details: fws.append([r.club_name,r.path.name,f.face_index,str(f.bbox),f.left_eye_ratio,f.right_eye_ratio,f.eye_closed])
    rws=wb.create_sheet(CLUB_SHEET_LABELS["Best Shot Ranking"])
    rws.append([excel_label("club"),excel_label("rank"),excel_label("file_name"),excel_label("formality"),excel_label("beauty_score"),excel_label("expression_score"),excel_label("emotion_score"),excel_label("people_count_score"),excel_label("gesture_expression_penalty"),excel_label("is_ng"),excel_label("ng_reason"),excel_label("short_comment"),excel_label("total_score")])
    for r in sorted(results,key=lambda x:(x.club_name,x.rank)):
        ev=r.eval_data
        rws.append([r.club_name,r.rank,r.path.name,ev.get("formality_score"),ev.get("beauty_score"),ev.get("expression_score"),ev.get("emotion_score"),ev.get("people_count_score"),ev.get("gesture_expression_penalty"),translate_display_value(r.ng_flag),translate_display_value(r.ng_reason),translate_display_value(ev.get("short_comment")),r.total_score])
    nws=wb.create_sheet(CLUB_SHEET_LABELS["Rename Output"]); nws.append([excel_label("club"),excel_label("original_file"),excel_label("renamed_file"),excel_label("shooting_date"),excel_label("rank")])
    for r in sorted(results,key=lambda x:(x.club_name,x.rank)): nws.append([r.club_name,r.path.name,r.renamed,r.shooting_date,r.rank])
    ngs=wb.create_sheet("NG写真・要確認")
    ngs.append([excel_label("club"),excel_label("file_name"),excel_label("ng_flag"),excel_label("reason")])
    for r in sorted(results,key=lambda x:(x.club_name,x.rank)):
        if r.ng_flag: ngs.append([r.club_name,r.path.name,translate_display_value(r.ng_flag),translate_display_value(r.ng_reason)])
    wb.save(excel)
    jsonp.write_text(json.dumps({"summary":{"club_count":len(by),"photo_count":len(results)},"items":[{"club_name":r.club_name,"original_file":r.path.name,"rank":r.rank,"renamed_file":r.renamed,"ng_flag":r.ng_flag,"ng_reason":r.ng_reason,"total_score":r.total_score,"evaluation":r.eval_data} for r in sorted(results,key=lambda x:(x.club_name,x.rank))]},ensure_ascii=False,indent=2),encoding="utf-8")
    return {"status":"completed","summary":{"club_count":len(by),"photo_count":len(results)},"excel_path":str(excel),"json_path":str(jsonp),"output_dir":str(club_out)}
