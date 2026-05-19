from __future__ import annotations

GENERAL_COLUMN_LABELS = {
    "Summary": "サマリー",
    "metric": "項目",
    "value": "値",
    "file_name": "ファイル名",
    "file": "ファイル名",
    "original_file": "元ファイル名",
    "renamed_file": "リネーム後ファイル名",
    "shooting_date": "撮影日",
    "rank": "順位",
    "status": "ステータス",
    "reason": "理由",
    "description": "説明",
    "comment": "コメント",
}

SNAP_SHEET_LABELS = {
    "all_events_summary": "全イベントサマリー",
    "summary": "サマリー",
    "clusters": "類似グループ",
    "selection": "選定結果",
}

SNAP_COLUMN_LABELS = {
    "event_name": "イベント名",
    "total_input_images": "入力画像数",
    "total_clusters": "類似グループ数",
    "total_representative_candidates": "代表候補写真数",
    "dedup_reduction_rate": "重複削減率",
    "best_shot_count": "ベストショット指定枚数",
    "final_selected_count": "ベストショット選定枚数",
    "ng_count_after_menna": "NG写真枚数",
    "other_passing_count": "通過写真枚数",
    "cluster_id": "類似グループID",
    "capture_time": "撮影日時",
    "is_representative": "代表写真",
    "bucket": "区分",
}

SNAP_BUCKET_LABELS = {
    "final_selected": "ベストショット",
    "ng": "NG写真",
    "other_passing": "通過写真",
}

CLUB_SHEET_LABELS = {
    "Summary": "サマリー",
    "Eye Closure Summary": "目つぶり確認サマリー",
    "Face Detail": "顔検出詳細",
    "Best Shot Ranking": "ベストショット順位",
    "Rename Output": "リネーム結果",
}

CLUB_COLUMN_LABELS = {
    "club": "部活動名",
    "club_count": "部活動数",
    "photo_count": "写真枚数",
    "closed_eye_photo_count": "目つぶり写真枚数",
    "closed_eye_face_count": "目つぶり人数",
    "ranked_output_count": "順位付け出力枚数",
    "person_count": "人数",
    "closed_eye_faces": "目つぶり人数",
    "eyes_closed_photo": "目つぶり写真",
    "face_index": "顔番号",
    "bbox": "顔領域",
    "left_eye_ratio": "左目開眼比率",
    "right_eye_ratio": "右目開眼比率",
    "eye_closed": "目つぶり",
    "formality": "整列・構図",
    "quality": "品質",
    "expression": "表情",
    "gesture_penalty": "ポーズ減点",
    "obscured_penalty": "遮蔽減点",
    "ng_flag": "NG判定",
    "total_score": "総合スコア",
}

REASON_DESCRIPTION_LABELS = {
    "eyes closed": "目つぶり",
    "unreadable": "読み取り不可",
    "placard visible": "札写り込み",
    "low-sharp": "ピント不足",
    "blurry": "ピント不足",
    "dark": "暗い",
    "no face detected": "顔検出なし",
    "gesture": "不適切なポーズの可能性",
    "middle finger": "中指を立てている",
    "obscene pose": "不適切なポーズ",
    "improper pose": "不適切なポーズ",
    "selected as best shot": "ベストショットとして選定",
    "passed but not selected": "通過写真（ベストショット未選定）",
    "NG photo": "NG写真",
}


def excel_label(value: str) -> str:
    return (
        GENERAL_COLUMN_LABELS.get(value)
        or SNAP_COLUMN_LABELS.get(value)
        or CLUB_COLUMN_LABELS.get(value)
        or REASON_DESCRIPTION_LABELS.get(value)
        or value
    )
