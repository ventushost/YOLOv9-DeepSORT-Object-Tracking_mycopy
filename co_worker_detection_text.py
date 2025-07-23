import argparse
import os
import platform
import sys
import threading
import queue
import logging
import re
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta

import numpy as np
# --- NumPy alias compatibility for newer versions (>=1.24) ---
# DeepSORT/Scipy code may still call deprecated aliases like np.float, np.int, etc.
if not hasattr(np, 'float'):
    np.float = float
    np.int = int
    np.bool = bool
    np.object = object
    np.str = str

import torch
import pandas as pd

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer)
from utils.torch_utils import select_device

"""
Asynchrones YOLO (+v9 kompatibel) + DeepSORT mit:
- Producer/Consumer-Queue (Backpressure per maxsize & blockierendes put)
- Fix für "Inplace update to inference tensor outside InferenceMode"
- Stabiler VideoWriter-Key
- Ausführlichem Logging

Dateiname: co_worker_detection_text.py (gepatcht)
"""

# =============================
# Logging
# =============================
LOG = logging.getLogger("yolo_deepsort_async")

def setup_logger(level: str = "INFO"):
    if LOG.handlers:
        LOG.setLevel(getattr(logging, level.upper(), logging.INFO))
        return
    fmt = "[%(asctime)s] [%(levelname)5s] %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(format=fmt, datefmt=datefmt, level=getattr(logging, level.upper(), logging.INFO))
    LOG.setLevel(getattr(logging, level.upper(), logging.INFO))

# =============================
# Globals / Helpers
# =============================
tracked_data = []
data_deque = {}

def classNames():
    return ["human_head"]

className = classNames()

def colorLabels(classid: int):
    if classid == 0:  # human_head
        return (85, 45, 255)
    elif classid == 2:
        return (222, 82, 175)
    elif classid == 3:
        return (0, 204, 255)
    elif classid == 5:
        return (0, 149, 255)
    return (200, 100, 0)

def draw_boxes(frame, bbox_xyxy, draw_trails, identities=None, categories=None, offset=(0, 0)):
    # remove stale IDs
    for key in list(data_deque):
        if identities is None or key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(v) for v in box]
        x1 += offset[0]; x2 += offset[0]
        y1 += offset[1]; y2 += offset[1]
        center = int((x1 + x2) / 2), int((y1 + y2) / 2)

        cat = int(categories[i]) if categories is not None else 0
        color = colorLabels(cat)
        track_id = int(identities[i]) if identities is not None else 0

        if track_id not in data_deque:
            data_deque[track_id] = deque(maxlen=64)
        data_deque[track_id].appendleft(center)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        name = className[cat] if cat < len(className) else "object"
        label = f"{track_id}:{name}"
        text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=2)[0]
        c2 = x1 + text_size[0], y1 - text_size[1] - 3
        cv2.rectangle(frame, (x1, y1), c2, color, -1)
        cv2.putText(frame, label, (x1, y1 - 2), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, center, 2, (0, 255, 0), cv2.FILLED)

        if draw_trails:
            for k in range(1, len(data_deque[track_id])):
                if data_deque[track_id][k - 1] is None or data_deque[track_id][k] is None:
                    continue
                thickness = int(np.sqrt(64 / float(k + k)) * 1.5)
                cv2.line(frame, data_deque[track_id][k - 1], data_deque[track_id][k], color, thickness)
    return frame

def initialize_deepsort(cfg_path="deep_sort_pytorch/configs/deep_sort.yaml"):
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
        use_cuda=True
    )
    LOG.info("DeepSORT initialisiert")
    return deepsort

# =============================
# Queue-Item
# =============================
class DetectionItem:
    __slots__ = ("frame_idx", "path", "im0", "det", "cam_ip", "start_time", "fps",
                 "save_dir", "frame_num_global", "vid_key", "in_shape")
    def __init__(self, frame_idx, path, im0, det, cam_ip, start_time, fps, save_dir, frame_num_global, vid_key, in_shape):
        self.frame_idx = frame_idx
        self.path = path
        self.im0 = im0
        self.det = det
        self.cam_ip = cam_ip
        self.start_time = start_time
        self.fps = fps
        self.save_dir = save_dir
        self.frame_num_global = frame_num_global
        self.vid_key = vid_key
        self.in_shape = in_shape  # (h,w)

# =============================
# Detection Worker
# =============================

def detection_worker(det_queue: queue.Queue, dataset, model, conf_thres, iou_thres, classes, agnostic_nms,
                     augment, visualize, save_dir, pt, det_batch):
    LOG.info("[DET] Worker gestartet")
    frame_counter_global = 0

    batch_tensors = []
    batch_paths = []
    batch_im0s = []
    batch_vidcaps = []
    batch_frameidx = []
    batch_cam_ips = []
    batch_start_times = []
    batch_fps = []
    batch_vid_keys = []
    batch_in_shapes = []

    for paths, ims, im0s, vid_caps, texts in dataset:
        for idx, (path, im, im0, vid_cap, s) in enumerate(zip(paths, ims, im0s, vid_caps, texts)):
            p = Path(path)
            ip_match = re.search(r'camip=(\d+\.\d+\.\d+\.\d+)', p.name)
            if not ip_match:
                raise ValueError(f"❌ Keine Kamera-IP im Dateinamen {p.name}")
            cam_ip = ip_match.group(1)

            ts_match = re.search(r'date=(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.\d+)', p.name)
            if not ts_match:
                raise ValueError(f"❌ Kein Timestamp im Dateinamen {p.name}")
            start_time = datetime.strptime(ts_match.group(1), "%Y-%m-%d_%H-%M-%S.%f")

            fps_local = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30
            if not fps_local or fps_local <= 1:
                fps_local = 30

            t = torch.from_numpy(im).to(model.device)
            t = t.half() if model.fp16 else t.float()
            t /= 255
            if t.ndim == 3:
                t = t[None]

            batch_tensors.append(t)
            batch_paths.append(path)
            batch_im0s.append(im0.copy())
            batch_vidcaps.append(vid_cap)
            batch_frameidx.append(getattr(dataset, 'frame', 0))
            batch_cam_ips.append(cam_ip)
            batch_start_times.append(start_time)
            batch_fps.append(fps_local)
            batch_vid_keys.append(str(save_dir / p.name))
            batch_in_shapes.append(t.shape[2:])

            if len(batch_tensors) == det_batch:
                _process_batch_and_enqueue(batch_tensors, batch_paths, batch_im0s, batch_vidcaps, batch_frameidx,
                                           batch_cam_ips, batch_start_times, batch_fps, batch_vid_keys, batch_in_shapes,
                                           model, conf_thres, iou_thres, classes, agnostic_nms, augment, visualize,
                                           save_dir, pt, det_queue, frame_counter_global)
                frame_counter_global += len(batch_tensors)

                batch_tensors.clear(); batch_paths.clear(); batch_im0s.clear(); batch_vidcaps.clear(); batch_frameidx.clear()
                batch_cam_ips.clear(); batch_start_times.clear(); batch_fps.clear(); batch_vid_keys.clear(); batch_in_shapes.clear()

    if batch_tensors:
        _process_batch_and_enqueue(batch_tensors, batch_paths, batch_im0s, batch_vidcaps, batch_frameidx,
                                   batch_cam_ips, batch_start_times, batch_fps, batch_vid_keys, batch_in_shapes,
                                   model, conf_thres, iou_thres, classes, agnostic_nms, augment, visualize,
                                   save_dir, pt, det_queue, frame_counter_global)

    det_queue.put(None)
    LOG.info("[DET] Worker beendet")

def _process_batch_and_enqueue(batch_tensors, batch_paths, batch_im0s, batch_vidcaps, batch_frameidx,
                               batch_cam_ips, batch_start_times, batch_fps, batch_vid_keys, batch_in_shapes,
                               model, conf_thres, iou_thres, classes, agnostic_nms, augment, visualize,
                               save_dir, pt, det_queue, frame_counter_global):
    im_tensor = torch.cat(batch_tensors, dim=0)
    preds = model(im_tensor, augment=augment, visualize=False)[0]
    preds = non_max_suppression(preds, conf_thres, iou_thres, classes, agnostic_nms, max_det=1000)

    for i, det in enumerate(preds):
        item = DetectionItem(
            frame_idx=batch_frameidx[i],
            path=batch_paths[i],
            im0=batch_im0s[i],
            det=det,
            cam_ip=batch_cam_ips[i],
            start_time=batch_start_times[i],
            fps=batch_fps[i],
            save_dir=save_dir,
            frame_num_global=frame_counter_global + i,
            vid_key=batch_vid_keys[i],
            in_shape=batch_in_shapes[i]
        )
        det_queue.put(item)  # blockiert, wenn voll
    LOG.debug(f"[DET] Batch -> {len(preds)} Frames enqueued. Queue={det_queue.qsize()}")

# =============================
# Tracking Worker
# =============================

def tracking_worker(det_queue: queue.Queue, deepsort: DeepSort, draw_trails: bool, view_img: bool,
                    nosave: bool, save_img_default: bool):
    LOG.info("[TRK] Worker gestartet")
    vid_writers = {}
    processed = 0

    while True:
        item = det_queue.get()
        if item is None:
            LOG.info("[TRK] Endsignal erhalten")
            break

        det = item.det
        im0 = item.im0
        p = Path(item.path)
        frame_w, frame_h = im0.shape[1], im0.shape[0]

        bbox_xyxy, confs, oids = [], [], []
        if det is not None and len(det):
            # Skaliere YOLO-Boxen zurück auf Originalauflösung
            det[:, :4] = scale_boxes(item.in_shape, det[:, :4], im0.shape).round()
            for box in det:
                if len(box) == 6:
                    *xyxy, conf, cls = box
                    cls_int = int(cls)
                else:
                    *xyxy, conf = box
                    cls_int = 0
                x1, y1, x2, y2 = map(int, xyxy)
                if (x2 - x1) > 0 and (y2 - y1) > 0:
                    bbox_xyxy.append([x1, y1, x2, y2])
                    confs.append(float(conf))
                    oids.append(cls_int)
        else:
            deepsort.increment_ages()

        ims = im0.copy()
        outputs = []
        if bbox_xyxy:
            xywhs = torch.tensor([[ (x1+x2)//2, (y1+y2)//2, (x2-x1), (y2-y1) ] for x1, y1, x2, y2 in bbox_xyxy])
            confss = torch.tensor(confs)
            oids_t = torch.tensor(oids)
            try:
                # raus aus inference_mode
                with torch.inference_mode(False):
                    outputs = deepsort.update(xywhs.clone(), confss.clone(), oids_t.clone(), ims)
            except Exception as e:
                LOG.error(f"[TRK] DeepSORT.update Fehler: {e}")
                outputs = []

        if len(outputs) > 0:
            bbox_xyxy_out = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            draw_boxes(ims, bbox_xyxy_out, draw_trails, identities, object_id)

            for j, box in enumerate(bbox_xyxy_out):
                x1, y1, x2, y2 = map(int, box)
                x1 = max(0, x1); y1 = max(0, y1)
                x2 = min(im0.shape[1], x2); y2 = min(im0.shape[0], y2)
                roi = im0[y1:y2, x1:x2]
                ims[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (35, 35), 0)

                width, height = x2 - x1, y2 - y1
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                timestamp = item.start_time + timedelta(seconds=item.frame_idx / item.fps) if item.start_time else None
                conf_val = confs[j] if j < len(confs) else None
                class_val = int(object_id[j]) if j < len(object_id) else 0

                tracked_data.append({
                    "video_name": p.name,
                    "frame": int(item.frame_idx),
                    "object_id": int(identities[j]),
                    "class_id": class_val,
                    "class_name": className[class_val] if class_val < len(className) else "object",
                    "confidence": conf_val,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "center_x": center_x, "center_y": center_y,
                    "width": width, "height": height,
                    "timestamp": timestamp.isoformat() if timestamp else None,
                    "camera_ip": item.cam_ip,
                    "frame_width": frame_w, "frame_height": frame_h
                })

        # anzeigen
        if view_img:
            if platform.system() == 'Linux' and item.vid_key not in vid_writers:
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(str(p), ims.shape[1], ims.shape[0])
            cv2.imshow(str(p), ims)
            cv2.waitKey(1)

        # speichern
        save_img = save_img_default and not nosave and not str(item.path).endswith('.txt')
        if save_img:
            save_path_raw = item.vid_key
            if save_path_raw not in vid_writers:
                fps = item.fps if item.fps else 30
                h, w = ims.shape[:2]
                save_path = str(Path(save_path_raw).with_suffix('.mp4'))
                os.makedirs(Path(save_path).parent, exist_ok=True)
                vid_writers[save_path_raw] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                LOG.info(f"[TRK] VideoWriter geöffnet: {save_path} (fps={fps}, {w}x{h})")
            if item.frame_num_global < 3:
                ims = cv2.GaussianBlur(ims, (101, 101), 0)
            vid_writers[save_path_raw].write(ims)

        processed += 1
        if processed % 50 == 0:
            LOG.info(f"[TRK] {processed} Frames verarbeitet. Queue={det_queue.qsize()}")

    # cleanup
    for writer in vid_writers.values():
        if isinstance(writer, cv2.VideoWriter):
            writer.release()
    LOG.info("[TRK] Writer geschlossen")

# =============================
# Main
# =============================

def run(
        weights=Path('yolo.pt'),
        source=Path('data/images'),
        data=Path('data/coco.yaml'),
        imgsz=(640, 640),
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=1000,
        device='',
        view_img=False,
        nosave=False,
        classes=None,
        agnostic_nms=False,
        augment=False,
        visualize=False,
        update=False,
        project=Path('runs/detect'),
        name='exp',
        exist_ok=False,
        half=False,
        dnn=False,
        vid_stride=1,
        draw_trails=False,
        det_batch=8,
        log_level='INFO'
):
    setup_logger(log_level)
    LOG.info("Parameter:")
    LOG.info(str({k: v for k, v in locals().items() if k not in ['model', 'dataset']}))

    source = str(source)
    save_img_default = not nosave and not source.endswith('.txt')
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
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

    # Dataloader
    bs = 128
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
        LOG.info(f"Webcam/Streams: {bs}")
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        LOG.info("Screenshot-Modus")
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride, batch_size=bs)
        LOG.info("Datei/Ordner-Modus")

    # Warmup
    model.warmup(imgsz=(1 if pt or model.triton else det_batch, 3, *imgsz))
    LOG.info("Warmup fertig")

    deepsort = initialize_deepsort()

    # Queue mit Backpressure
    det_queue = queue.Queue(maxsize=det_batch * 2)

    t_det = threading.Thread(target=detection_worker, args=(det_queue, dataset, model, conf_thres, iou_thres, classes,
                                                            agnostic_nms, augment, visualize, save_dir, pt, det_batch),
                             daemon=True, name="DetectionThread")
    t_track = threading.Thread(target=tracking_worker, args=(det_queue, deepsort, draw_trails, view_img, nosave,
                                                             save_img_default), daemon=True, name="TrackingThread")

    LOG.info("Starte Threads...")
    t_det.start(); t_track.start()
    t_det.join(); t_track.join()
    LOG.info("Threads beendet")

    if tracked_data:
        df = pd.DataFrame(tracked_data)
        csv_output_path = save_dir / 'tracking_data.csv'
        df.to_csv(csv_output_path, index=False)
        LOG.info(f"Trackingdaten gespeichert: {csv_output_path}")

    if update:
        strip_optimizer(weights[0] if isinstance(weights, (list, tuple)) else weights)
        LOG.info("Optimizer-Stripping durchgeführt")

# =============================
# CLI
# =============================

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolo.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--draw-trails', action='store_true', help='draw trails')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--det-batch', type=int, default=8, help='Batchgröße für Detection-Worker')
    parser.add_argument('--log-level', type=str, default='INFO', help='DEBUG, INFO, WARNING, ERROR')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1
    print_args(vars(opt))
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
