import pandas as pd
import numpy as np
import argparse

# Argumente parsen
parser = argparse.ArgumentParser(description="Konfliktanalyse von Tracking-Daten")
parser.add_argument('--input', type=str, required=True, help='Pfad zur getrackten CSV-Datei')
parser.add_argument('--output', type=str, default='ocsort_conflicts.csv', help='Pfad zur Ausgabedatei für Konflikte')
parser.add_argument('--iou_thresh', type=float, default=0.3, help='IoU-Schwellenwert')
parser.add_argument('--dist_thresh', type=float, default=50, help='Distanzschwelle für Nähe')
parser.add_argument('--gap_thresh', type=int, default=3, help='Maximal erlaubte Frame-Lücke')
parser.add_argument('--margin', type=int, default=20, help='Bildrand-Marge')
args = parser.parse_args()

# Parameter
INPUT_TRACKED_CSV = args.input
OUTPUT_CONFLICT_CSV = args.output
IOU_THRESH = args.iou_thresh
DIST_THRESH = args.dist_thresh
GAP_THRESH = args.gap_thresh
MARGIN = args.margin

# Trackingdaten laden
df = pd.read_csv(INPUT_TRACKED_CSV)

# Bildgröße bestimmen (Fallback auf FullHD)
W = int(df['frame_width'].iloc[0]) if 'frame_width' in df.columns else 1920
H = int(df['frame_height'].iloc[0]) if 'frame_height' in df.columns else 1080

# Zentren berechnen
df['cx'] = (df.x1 + df.x2) / 2.0
df['cy'] = (df.y1 + df.y2) / 2.0

# IoU-Matrix
def compute_iou_matrix(boxes):
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    N = len(boxes)
    iou_mat = np.zeros((N, N))
    for i in range(N):
        xi1 = np.maximum(x1[i], x1)
        yi1 = np.maximum(y1[i], y1)
        xi2 = np.minimum(x2[i], x2)
        yi2 = np.minimum(y2[i], y2)
        inter_w = np.maximum(0, xi2 - xi1)
        inter_h = np.maximum(0, yi2 - yi1)
        inter_area = inter_w * inter_h
        union = areas[i] + areas - inter_area
        iou_mat[i] = inter_area / (union + 1e-6)
    return iou_mat

conflict_indices = set()

# Regel 3: Overlap oder Nähe innerhalb eines Frames
for frame_id, group in df.groupby('frame'):
    boxes = group[['x1', 'y1', 'x2', 'y2']].values
    centers = group[['cx', 'cy']].values
    iou_mat = compute_iou_matrix(boxes)
    dist_mat = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)

    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            if iou_mat[i, j] > IOU_THRESH or dist_mat[i, j] < DIST_THRESH:
                conflict_indices.update({group.index[i], group.index[j]})

# Regel 1: Lücken innerhalb von Track-IDs im Zentrum
for track_id, sub in df.groupby('track_id'):
    frames = sorted(sub.frame.tolist())
    for a, b in zip(frames, frames[1:] + [frames[-1] + GAP_THRESH + 1]):
        if 1 < b - a <= GAP_THRESH:
            row = sub[sub.frame == a].iloc[0]
            if (row.x1 > MARGIN and row.y1 > MARGIN and
                row.x2 < W - MARGIN and row.y2 < H - MARGIN):
                conflict_indices.add(row.name)

# Konflikte speichern
df_conflicts = df.loc[sorted(conflict_indices)]
df_conflicts.to_csv(OUTPUT_CONFLICT_CSV, index=False)

print(f"✅ Konfliktanalyse abgeschlossen: {len(conflict_indices)} Konflikte in {OUTPUT_CONFLICT_CSV}")
