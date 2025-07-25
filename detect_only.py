#!/usr/bin/env python3
"""
DETECT FAST PIPELINE
====================
Ziel: Forward-Pass minimieren & Overhead entfernen.

Features:
- Echte Batch-Inferenz (ein Forward pro Batch)
- Optionales Profiling (--profile) ohne Performance-Verlust im Normalmodus
- Half-Precision (--half) default an, falls HW/Modell kompatibel
- Kleinere Default-Auflösung (512) -> weniger FLOPs
- Optional: torch.compile() (--compile) für PyTorch 2.x
- Optionale Frameskip (--vid-stride > 1)
- Option: Video zuerst lokal kopieren (--copy-local) um Drive-Latenz zu vermeiden
- Progressbar mit reduzierter Update-Frequenz (--tqdm-update)

Ausgabe: runs/detect/exp*/detections.csv

Hinweis: TensorRT/ONNX Schritte sind separat (siehe unten in der Chat-Antwort Commands).
"""
import argparse
import re
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import shutil

import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (increment_path, check_img_size, check_file, check_imshow, non_max_suppression,
                           scale_boxes, print_args, cv2)
from utils.torch_utils import select_device

LOG = logging.getLogger("detect_fast")

# ------------------------ Logger ------------------------
def setup_logger(level: str = "INFO"):
    if LOG.handlers:
        LOG.setLevel(getattr(logging, level.upper(), logging.INFO))
        return
    fmt = "[%(asctime)s] [%(levelname)5s] %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(format=fmt, datefmt=datefmt, level=getattr(logging, level.upper(), logging.INFO))
    LOG.setLevel(getattr(logging, level.upper(), logging.INFO))

# ------------------------ Helpers ------------------------
RE_IP = re.compile(r"camip=(\d+\.\d+\.\d+\.\d+)")
RE_TS = re.compile(r"date=(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d+)")

def parse_name_meta(filename: str):
    p = Path(filename)
    ip_match = RE_IP.search(p.name)
    if not ip_match:
        raise ValueError(f"Keine Kamera-IP im Dateinamen {p.name}")
    cam_ip = ip_match.group(1)

    ts_match = RE_TS.search(p.name)
    if not ts_match:
        raise ValueError(f"Kein Timestamp im Dateinamen {p.name}")
    start_time = datetime.strptime(ts_match.group(1), "%Y-%m-%d_%H-%M-%S.%f")
    return cam_ip, start_time

# ------------------------ Main ------------------------
def run(weights, source, data, imgsz, conf_thres, iou_thres, max_det, device, agnostic_nms,
        augment, project, name, exist_ok, half, dnn, vid_stride, det_batch, log_level,
        tqdm_update, profile, compile_model, copy_local):
    setup_logger(log_level)
    LOG.info("Starte Detection FAST")

    torch.backends.cudnn.benchmark = True

    # optional: source lokal kopieren (nur Dateien, keine Streams)
    if copy_local:
        src_path = Path(source)
        if src_path.exists() and src_path.is_file():
            local_path = Path("/content") / src_path.name
            if not local_path.exists():
                LOG.info(f"Kopiere Video lokal nach {local_path} …")
                shutil.copy2(src_path, local_path)
            source = str(local_path)
        else:
            LOG.warning("--copy-local gesetzt, aber Source ist kein einzelnes File oder existiert nicht.")

    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    LOG.info(f"Output-Verzeichnis: {save_dir}")

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    LOG.info(f"YOLO geladen (stride={stride}, fp16={model.fp16})")

    if compile_model:
        try:
            LOG.info("torch.compile() aktiv …")
            model.model = torch.compile(model.model, mode="reduce-overhead")
        except Exception as e:
            LOG.warning(f"torch.compile fehlgeschlagen: {e}")

    # Dataloader
    if webcam:
        _ = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt,
                             vid_stride=vid_stride, batch_size=det_batch)

    # Warmup
    model.warmup(imgsz=(1 if pt or model.triton else det_batch, 3, *imgsz))

    detections = []
    CSV_BATCH_SIZE = 5000
    LOG.info("Laufe durch Frames …")

    total_frames = getattr(dataset, 'nframes', None)
    pbar = tqdm(total=total_frames, desc='Detection', unit='frame', dynamic_ncols=True)

    t_total_start = time.time()
    frames_done = 0
    frame_counters = defaultdict(int)

    # Profiling accumulators
    t_h2d = t_forward = t_nms = t_post = 0.0
    use_cuda = device != 'cpu' and torch.cuda.is_available()
    if profile and use_cuda:
        ev_fwd_s = torch.cuda.Event(enable_timing=True)
        ev_fwd_e = torch.cuda.Event(enable_timing=True)

    for paths, ims, im0s, vid_caps, texts in dataset:
        bs_act = len(paths)
        # H2D copy
        t0 = time.time()
        np_batch = np.stack(ims)
        im_tensor = torch.from_numpy(np_batch).to(model.device, non_blocking=True)
        im_tensor = im_tensor.half() if model.fp16 else im_tensor.float()
        im_tensor /= 255.0
        if profile and use_cuda:
            torch.cuda.synchronize()
        t_h2d += (time.time() - t0) if profile else 0.0

        # Forward
        if profile and use_cuda:
            torch.cuda.synchronize(); ev_fwd_s.record()
        pred_raw = model(im_tensor, augment=augment, visualize=False)[0]
        if profile and use_cuda:
            ev_fwd_e.record(); torch.cuda.synchronize()
            t_forward += ev_fwd_s.elapsed_time(ev_fwd_e)/1000.0

        # NMS
        t1 = time.time()
        preds = non_max_suppression(pred_raw, conf_thres, iou_thres, None, agnostic_nms, max_det=max_det)
        t_nms += (time.time() - t1) if profile else 0.0

        # Postproc
        t2 = time.time()
        for i in range(bs_act):
            path = paths[i]; im0 = im0s[i]; vid_cap = vid_caps[i]; det = preds[i]
            p = Path(path)
            cam_ip, start_time_video = parse_name_meta(p.name)

            fps_local = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
            if not fps_local or fps_local <= 1:
                fps_local = 30

            key = str(p)
            frame_idx = frame_counters[key]
            frame_counters[key] += 1

            timestamp = start_time_video + timedelta(seconds=frame_idx / fps_local) if start_time_video else None

            if len(det):
                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    detections_batch.append({
                        'video_name': p.name,
                        'frame': int(frame_idx),
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'conf': float(conf),
                        'cls': int(cls),
                        'timestamp': timestamp.isoformat() if timestamp else None,
                        'camera_ip': cam_ip,
                        'fps': fps_local,
                        'frame_width': im0.shape[1],
                        'frame_height': im0.shape[0]
                    })
                    
                # Batch-weise CSV-Schreiben
                if len(detections_batch) >= CSV_BATCH_SIZE:
                    df_tmp = pd.DataFrame(detections_batch)
                    df_tmp.to_csv(out_csv, mode='a', index=False, header=not Path(out_csv).exists())
                    detections_batch.clear()


            frames_done += 1
            if frames_done % tqdm_update == 0:
                elapsed = time.time() - t_total_start
                fps_now = frames_done / elapsed if elapsed > 0 else 0.0
                eta = (total_frames - frames_done) / fps_now if (total_frames and fps_now > 0) else float('nan')
                pbar.set_postfix({'fps': f"{fps_now:5.2f}", 'eta_s': f"{eta:6.1f}" if not np.isnan(eta) else 'n/a'})
                pbar.update(tqdm_update)
        t_post += (time.time() - t2) if profile else 0.0

    # finalize progress bar
    if total_frames:
        remainder = total_frames - frames_done
        if remainder > 0:
            pbar.update(remainder)
    pbar.close()

    t_total = time.time() - t_total_start
    avg_fps = frames_done / t_total if t_total > 0 else 0.0
    LOG.info(f"Gesamt: {frames_done} Frames | Zeit: {t_total:.2f}s | ∅FPS: {avg_fps:.2f}")

    if profile:
        def pct(x):
            return 100.0 * x / t_total if t_total > 0 else 0.0
        LOG.info("Zeitaufteilung:")
        LOG.info(f"  H2D copy:   {t_h2d:6.3f}s ({pct(t_h2d):5.1f}%)")
        LOG.info(f"  Forward:    {t_forward:6.3f}s ({pct(t_forward):5.1f}%)")
        LOG.info(f"  NMS:        {t_nms:6.3f}s ({pct(t_nms):5.1f}%)")
        LOG.info(f"  Postproc:   {t_post:6.3f}s ({pct(t_post):5.1f}%)")

    # CSV speichern
    df = pd.DataFrame(detections)
    out_csv = save_dir / 'detections.csv'
    df.to_csv(out_csv, index=False)
    LOG.info(f"Detections gespeichert: {out_csv}")
    LOG.info("Fertig.")

# ------------------------ CLI ------------------------
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, required=True, help='Pfad zum YOLO-Model (.pt)')
    parser.add_argument('--source', type=str, required=True, help='Video/Ordner/Stream')
    parser.add_argument('--data', type=str, default='data/coco128.yaml')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[512])
    parser.add_argument('--conf-thres', type=float, default=0.25)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--max-det', type=int, default=1000)
    parser.add_argument('--device', default='0')
    parser.add_argument('--agnostic-nms', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--project', default='runs/detect')
    parser.add_argument('--name', default='exp')
    parser.add_argument('--exist-ok', action='store_true')
    parser.add_argument('--half', action='store_true', help='FP16 Inferenz (empfohlen auf GPU)')
    parser.add_argument('--dnn', action='store_true')
    parser.add_argument('--vid-stride', type=int, default=1, help='Frame-Stride (>=2 skippt Frames)')
    parser.add_argument('--det-batch', type=int, default=16, help='Batchgröße für Loader & Forward')
    parser.add_argument('--tqdm-update', type=int, default=20, help='tqdm Update alle N Frames')
    parser.add_argument('--log-level', type=str, default='INFO')
    parser.add_argument('--profile', action='store_true', help='CUDA/CPU Zeiten messen (langsamer)')
    parser.add_argument('--compile', dest='compile_model', action='store_true', help='torch.compile verwenden')
    parser.add_argument('--copy-local', action='store_true', help='Source-Video erst nach /content kopieren')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    run(**vars(opt))
