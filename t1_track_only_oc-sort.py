import pandas as pd
import numpy as np
from ocsort.ocsort import OCSort
import torch
from pathlib import Path
import argparse

# Argumente einlesen
parser = argparse.ArgumentParser(description="OCSort Tracking für Excel-Datei")
parser.add_argument('--input', type=str, required=True, help='Pfad zur Eingabe-Excel- oder CSV-Datei')
parser.add_argument('--output', type=str, default='ocsort_tracked.csv', help='Pfad zur Ausgabedatei (CSV)')
parser.add_argument('--batch', type=int, default=1000, help='Batch-Größe für Zwischenspeicherung')
args = parser.parse_args()

INPUT_CSV = args.input
OUTPUT_OCSORT_CSV = args.output
BATCH_SIZE = args.batch

# Datei laden
if INPUT_CSV.endswith('.csv'):
    df = pd.read_csv(INPUT_CSV)
elif INPUT_CSV.endswith(('.xls', '.xlsx')):
    df = pd.read_excel(INPUT_CSV)
else:
    raise ValueError("Dateiformat nicht unterstützt. Bitte CSV oder Excel angeben.")

# Tracker initialisieren
tracker = OCSort()
tracked_rows = []

# Header-Check
header_written = Path(OUTPUT_OCSORT_CSV).exists()

# IoU-Funktion
def iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / (union + 1e-6)

# Tracking pro Frame
for frame_id, group in df.groupby('frame'):
    dets = []
    meta = []

    for i, row in group.iterrows():
        det = [row.x1, row.y1, row.x2, row.y2, row.conf, row.cls]
        dets.append(det)
        meta.append({
            'frame': int(row.frame),
            'x1': row.x1,
            'y1': row.y1,
            'x2': row.x2,
            'y2': row.y2,
            'conf': row.conf,
            'cls': row.cls,
            'timestamp': row.get('timestamp', None),
            'original_index': i
        })

    dets = torch.tensor(dets, dtype=torch.float32)
    outputs = tracker.update(dets, frame_id)

    for output in outputs:
        x1, y1, x2, y2, track_id = output[:5]
        best_match = None
        best_iou = 0

        for m in meta:
            box_gt = [m['x1'], m['y1'], m['x2'], m['y2']]
            iou_score = iou([x1, y1, x2, y2], box_gt)
            if iou_score > best_iou and iou_score > 0.9:
                best_match = m
                best_iou = iou_score

        if best_match:
            tracked_rows.append({**best_match, 'track_id': int(track_id)})

    if len(tracked_rows) >= BATCH_SIZE:
        df_tmp = pd.DataFrame(tracked_rows)
        df_tmp.to_csv(OUTPUT_OCSORT_CSV, mode='a', index=False, header=not header_written)
        header_written = True
        tracked_rows.clear()

# Rest speichern
if tracked_rows:
    df_tmp = pd.DataFrame(tracked_rows)
    df_tmp.to_csv(OUTPUT_OCSORT_CSV, mode='a', index=False, header=not header_written)

print(f"✅ Fertig: {OUTPUT_OCSORT_CSV} mit kontinuierlichem Schreiben.")

