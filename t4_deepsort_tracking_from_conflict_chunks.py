import cv2
import os
import sys
import torch
import pandas as pd
from pathlib import Path

# Path to your DeepSORT repo
DEEPSORT_PATH = "/Users/erik/Downloads/code_csv_to_diagram/YOLOv9-DeepSORT-Object-Tracking_mycopy"
sys.path.append(str(Path(DEEPSORT_PATH) / "deep_sort_pytorch"))

from deep_sort_pytorch.deep_sort import DeepSort

# === Configuration ===
VIDEO_PATH = "/Users/erik/Downloads/code_csv_to_diagram/YOLOv9-DeepSORT-Object-Tracking_mycopy/event=test_testevent_2025_camip=10.0.0.111_date=2025-06-25_08-21-18.740.mkv"  # ⬅️ Set this
CONTEXT_CSV = "deepsort_conflict_context.csv"
START_IDS_CSV = "deepsort_conflict_start_ids.csv"
OUTPUT_CSV = "deepsort_conflict_fixed.csv"
FRAME_GAP_THRESHOLD = 10

# === Load data ===
context_df = pd.read_csv(CONTEXT_CSV)
start_ids_df = pd.read_csv(START_IDS_CSV)

# === Prepare batches ===
frame_groups = context_df['frame'].sort_values().unique()
batches = []
current_batch = []
last_frame = None

for frame in frame_groups:
    if last_frame is None or frame - last_frame <= FRAME_GAP_THRESHOLD:
        current_batch.append(frame)
    else:
        batches.append(current_batch)
        current_batch = [frame]
    last_frame = frame
if current_batch:
    batches.append(current_batch)

# === Load video frames ===
cap = cv2.VideoCapture(VIDEO_PATH)
frame_cache = {}
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    frame_cache[frame_num] = frame
cap.release()

# === Init DeepSORT ===
deepsort = DeepSort(
    model_path=os.path.join(DEEPSORT_PATH, "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"),
    max_dist=0.2, min_confidence=0.3, nms_max_overlap=0.5,
    max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100,
    use_cuda=torch.cuda.is_available()
)

# === Output storage ===
output_rows = []

# === Process each batch ===
for batch_frames in batches:
    batch_df = context_df[context_df['frame'].isin(batch_frames)].copy()
    batch_df = batch_df.sort_values("frame")

    first_frame = min(batch_frames)
    init_tracks = start_ids_df[start_ids_df["frame"] == first_frame]

    deepsort = DeepSort(
        model_path=os.path.join(DEEPSORT_PATH, "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"),
        max_dist=0.2, min_confidence=0.3, nms_max_overlap=0.5,
        max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100,
        use_cuda=torch.cuda.is_available()
    )

    for frame_num in sorted(batch_frames):
        frame = frame_cache.get(frame_num)
        if frame is None:
            continue

        detections = batch_df[batch_df["frame"] == frame_num]

        bbox_xywh = []
        confs = []

        for _, row in detections.iterrows():
            x1, y1, x2, y2 = row[["x1", "y1", "x2", "y2"]]
            w = x2 - x1
            h = y2 - y1
            bbox_xywh.append([x1 + w / 2, y1 + h / 2, w, h])
            confs.append(row["conf"])

        bbox_xywh = torch.Tensor(bbox_xywh)
        confs = torch.Tensor(confs)

        oids = [0] * len(bbox_xywh)  # Replace 0 with int(row["cls"]) if available
        outputs = deepsort.update(bbox_xywh, confs, oids, frame)



        for out in outputs:
            x1, y1, x2, y2, track_id, cls_id = out
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            output_rows.append({
                "frame": frame_num,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": 1.0,
                "cls": cls_id,
                "timestamp": "",
                "original_index": "",
                "track_id": track_id,
                "cx": cx, "cy": cy
            })


# === Save CSV ===
output_df = pd.DataFrame(output_rows)
output_df = output_df.sort_values(by=["frame", "track_id"])
output_df.to_csv(OUTPUT_CSV, index=False)

print(f"[✓] Done. Output saved to {OUTPUT_CSV}")
