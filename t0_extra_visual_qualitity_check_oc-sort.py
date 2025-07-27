import cv2
import pandas as pd
import os

# Dateinamen anpassen, falls nötig
VIDEO_IN      = 'event=test_testevent_2025_camip=10.0.0.111_date=2025-06-25_08-21-18.740.mkv'
imput_CSV    = '/Users/erik/Downloads/code_csv_to_diagram/YOLOv9-DeepSORT-Object-Tracking_mycopy/ocsort_tracked_filtered.csv'
VIDEO_OUT     = 'debug_overlay.mp4'

# CSVs laden
df_tr = pd.read_csv(imput_CSV)



# Gruppieren nach Frame
frames_data = {}
for _, row in df_tr.iterrows():
    key = int(row.frame)
    bbox = (int(row.x1), int(row.y1), int(row.x2), int(row.y2))
    tid  = int(row.track_id)
    frames_data.setdefault(key, []).append((bbox, tid))

# Video öffnen
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError(f"Kann Video nicht öffnen: {VIDEO_IN}")

# Video-Writer: gleiches Format + FPS + Größe
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (width, height))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Zeichne alle Boxes dieses Frames
    for bbox, tid in frames_data.get(frame_idx, []):
        x1, y1, x2, y2 = bbox
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'ID{tid}', (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"=> Overlay-Video gespeichert als {VIDEO_OUT}")
