#!/usr/bin/env python3
"""
Track-Only Script (DeepSORT)
----------------------------
Nutzt vorher berechnete Detections (CSV) + Originalvideo, um:
  • DeepSORT-Tracking durchzuführen (CPU oder GPU)
  • Optional Blur und Trails zu zeichnen
  • Ein finales Video und eine tracking_data.csv zu erzeugen

Aufruf-Beispiel:
python track_only.py \
  --video-path /pfad/zum/video.mkv \
  --detections-csv /pfad/zu/detections.csv \
  --device cpu \
  --blur \
  --log-level INFO

Voraussetzungen:
- deep_sort_pytorch Projektstruktur (cfg & ckpt)
- pandas, numpy, torch, opencv-python
"""
import argparse
import logging
import os
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import cv2
import torch
from utils.general import increment_path


# --- NumPy alias compatibility for >=1.24 ---
if not hasattr(np, "float"):
    np.float = float
    np.int = int
    np.bool = bool
    np.object = object
    np.str = str

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

# Simple drawing color map
COLORS = [(85,45,255),(222,82,175),(0,204,255),(0,149,255),(200,100,0)]

def color_for(cls_id: int):
    return COLORS[cls_id % len(COLORS)]

LOG = logging.getLogger("track_only")

# ----------------------------- Logger -----------------------------
def setup_logger(level: str = "INFO"):
    if LOG.handlers:
        LOG.setLevel(getattr(logging, level.upper(), logging.INFO))
        return
    fmt = "[%(asctime)s] [%(levelname)5s] %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(format=fmt, datefmt=datefmt, level=getattr(logging, level.upper(), logging.INFO))
    LOG.setLevel(getattr(logging, level.upper(), logging.INFO))

# --------------------------- DeepSORT init -------------------------
def init_deepsort(cfg_path: str, use_cuda: bool) -> DeepSort:
    cfg = get_config()
    cfg.merge_from_file(cfg_path)
    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=use_cuda,
    )
    return deepsort

# ------------------------ Trails Buffer ---------------------------
class TrailBuffer:
    def __init__(self, maxlen=64):
        self.buf = {}
        self.maxlen = maxlen
    def push(self, track_id, center):
        if track_id not in self.buf:
            self.buf[track_id] = deque(maxlen=self.maxlen)
        self.buf[track_id].appendleft(center)
    def draw(self, img, track_id, color):
        if track_id not in self.buf:
            return
        pts = self.buf[track_id]
        for i in range(1, len(pts)):
            if pts[i-1] is None or pts[i] is None:
                continue
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            cv2.line(img, pts[i-1], pts[i], color, thickness)

# --------------------------- Main Run -----------------------------
def run(video_path: str,
        detections_csv: str,
        project: str,
        name: str,
        exist_ok: bool,
        blur: bool,
        draw_trails: bool,
        cfg_deepsort: str,
        device: str,
        log_level: str):

    setup_logger(log_level)
    LOG.info("Starte Tracking-Only ...")

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    LOG.info(f"Output-Verzeichnis: {save_dir}")

    # Lade Detections
    df = pd.read_csv(detections_csv)
    if 'frame' not in df.columns:
        raise ValueError("detections.csv enthält keine 'frame'-Spalte")
    df.sort_values(['frame'], inplace=True)

    # Video öffnen
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Kann Video nicht öffnen: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = save_dir / (Path(video_path).stem + "_tracked.mp4")
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Torch/Device
    use_cuda = (device != 'cpu' and torch.cuda.is_available())
    if not use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_grad_enabled(False)

    # DeepSORT init
    deepsort = init_deepsort(cfg_deepsort, use_cuda)

    # Trails
    trails = TrailBuffer(maxlen=64) if draw_trails else None

    tracked_rows = []
    processed = 0

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # hole detections dieses Frames
        dets = df[df.frame == frame_idx]
        outputs = []
        if len(dets):
            xywhs = []
            confs = []
            clss  = []
            for _, row in dets.iterrows():
                x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
                w_box, h_box = x2 - x1, y2 - y1
                cx, cy = x1 + w_box / 2.0, y1 + h_box / 2.0
                xywhs.append([cx, cy, w_box, h_box])
                confs.append(row['conf'])
                clss.append(row['cls'])

            xywhs_t = torch.tensor(xywhs)
            confs_t = torch.tensor(confs)
            clss_t  = torch.tensor(clss)
            try:
                with torch.inference_mode(False):
                    outputs = deepsort.update(xywhs_t.clone(), confs_t.clone(), clss_t.clone(), frame)
            except Exception as e:
                LOG.error(f"DeepSORT.update Fehler (Frame {frame_idx}): {e}")
                outputs = []
        else:
            deepsort.increment_ages()

        if len(outputs) > 0:
            for j, out in enumerate(outputs):
                x1, y1, x2, y2, track_id, cls_id = out
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                # Blur optional
                if blur:
                    roi = frame[y1:y2, x1:x2]
                    if roi.size > 0:
                        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (35, 35), 0)

                # Draw box
                color = color_for(int(cls_id))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID {int(track_id)} | C {int(cls_id)}", (x1, y1-7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Trail
                if trails is not None:
                    cx_c = int((x1 + x2)//2)
                    cy_c = int((y1 + y2)//2)
                    trails.push(int(track_id), (cx_c, cy_c))
                    trails.draw(frame, int(track_id), color)

                # passende Detection-Row (optional)
                # Achtung: j entspricht nicht zwingend Index in dets (sortiere ggf.)
                conf_val = confs[j] if j < len(confs) else None
                row_det = dets.iloc[j] if j < len(dets) else None
                tracked_rows.append({
                    'video_name': Path(video_path).name,
                    'frame': frame_idx,
                    'object_id': int(track_id),
                    'class_id': int(cls_id),
                    'class_name': None,
                    'confidence': float(conf_val) if conf_val is not None else None,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'center_x': int((x1 + x2)//2), 'center_y': int((y1 + y2)//2),
                    'width': x2 - x1, 'height': y2 - y1,
                    'timestamp': row_det['timestamp'] if row_det is not None and 'timestamp' in row_det else None,
                    'camera_ip': row_det['camera_ip'] if row_det is not None and 'camera_ip' in row_det else None,
                    'frame_width': w, 'frame_height': h
                })

        writer.write(frame)
        processed += 1
        if processed % 100 == 0:
            LOG.info(f"{processed}/{total_frames} Frames verarbeitet")

    cap.release()
    writer.release()

    out_csv = save_dir / 'tracking_data.csv'
    pd.DataFrame(tracked_rows).to_csv(out_csv, index=False)
    LOG.info(f"Tracking CSV gespeichert: {out_csv}")
    LOG.info(f"Video gespeichert: {out_video}")
    LOG.info("Fertig.")

# --------------------------- Arg Parser ---------------------------
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True, help='Pfad zum Originalvideo')
    parser.add_argument('--detections-csv', type=str, required=True, help='CSV aus detect_only.py')
    parser.add_argument('--project', type=str, default='runs/track', help='Output-Projektordner')
    parser.add_argument('--name', type=str, default='exp', help='Run-Name')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--blur', action='store_true', help='Regionen innerhalb der BBox blurren')
    parser.add_argument('--draw-trails', action='store_true', help='Bewegungstrails zeichnen')
    parser.add_argument('--cfg-deepsort', type=str, default='deep_sort_pytorch/configs/deep_sort.yaml')
    parser.add_argument('--device', type=str, default='cpu', help='cpu oder cuda:0 etc.')
    parser.add_argument('--log-level', type=str, default='INFO')
    return parser.parse_args()

# ----------------------------- Utils -----------------------------
# import hier, um Zirkelimporte oben zu vermeiden
from utils.general import increment_path

# ------------------------------ Main ------------------------------
if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))
