import numpy as np
# Legacy numpy compatibility
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import cv2
import os
import subprocess
import torch
import pandas as pd
import gc
import argparse
import sys
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# === Logging konfigurieren ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def process_single_batch(task_args):
    """
    Worker-Funktion für parallele Batch-Verarbeitung.
    task_args: Tuple mit (idx, batch, DEEPSORT_PATH, VIDEO_PATH, CONTEXT_CSV, ds_params, fps)
    """
    idx, batch, DEEPSORT_PATH, VIDEO_PATH, CONTEXT_CSV, START_IDS_CSV, ds_params, fps = task_args
    logging.info("Worker startet Batch %d: Frames %d bis %d", idx, batch[0], batch[-1])

    # CSV nur für diesen Worker laden
    context_df = pd.read_csv(CONTEXT_CSV)

    start_f, end_f = batch[0], batch[-1]
    start_s = start_f / fps
    duration_s = (end_f - start_f + 1) / fps
    clip_file = f"temp_batch_{idx}.mp4"

    # 1) Clip mit FFmpeg extrahieren
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-i", VIDEO_PATH,
        "-t", str(duration_s),
        "-c", "copy",
        clip_file
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    logging.info("Worker Batch %d: Clip erstellt (%s)", idx, clip_file)

    # 2) Neuer Tracker instanziieren
    sys.path.append(str(Path(DEEPSORT_PATH) / "deep_sort_pytorch"))
    from deep_sort_pytorch.deep_sort import DeepSort
    deepsort = DeepSort(**ds_params)

    # 3) Clip frameweise tracken
    cap = cv2.VideoCapture(clip_file)
    batch_out = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx_seg = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        abs_frame = start_f + idx_seg

        dets = context_df[context_df['frame'] == abs_frame]
        if dets.empty:
            continue

        bbox_xywh, confs = [], []
        for _, r in dets.iterrows():
            x1, y1, x2, y2 = r[['x1','y1','x2','y2']]
            w, h = x2 - x1, y2 - y1
            bbox_xywh.append([x1 + w/2, y1 + h/2, w, h])
            confs.append(r['conf'])

        if not bbox_xywh:
            continue

        outs = deepsort.update(
            torch.Tensor(bbox_xywh),
            torch.Tensor(confs),
            [0] * len(bbox_xywh),
            frame
        )
        for x1, y1, x2, y2, tid, cls in outs:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            batch_out.append({
                'frame': abs_frame,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'conf': 1.0, 'cls': cls, 'timestamp': '',
                'original_index': '', 'track_id': tid,
                'cx': cx, 'cy': cy
            })
    cap.release()
    os.remove(clip_file)
    logging.info("Worker Batch %d: Clip gelöscht", idx)

    # Speicher freigeben
    del context_df
    gc.collect()

    # === Korrektur der IDs aus start_ids_csv ===
    start_ids_df = pd.read_csv(START_IDS_CSV)

    first_frame_num = batch[0]
    batch_out_df = pd.DataFrame(batch_out)

    first_frame_detections = batch_out_df[batch_out_df['frame'] == first_frame_num]

    for _, det in first_frame_detections.iterrows():
        cx, cy, assigned_tid = det['cx'], det['cy'], det['track_id']
        candidates = start_ids_df[start_ids_df['frame'] == first_frame_num]
        if candidates.empty:
            continue
        candidates['distance'] = np.sqrt((candidates['cx'] - cx)**2 + (candidates['cy'] - cy)**2)
        closest_match = candidates.loc[candidates['distance'].idxmin()]
        if closest_match['distance'] < 10:
            correct_tid = closest_match['track_id']
            batch_out_df.loc[batch_out_df['track_id'] == assigned_tid, 'track_id'] = correct_tid

    batch_out = batch_out_df.to_dict(orient='records')


    return idx, batch_out


def main():
    parser = argparse.ArgumentParser(
        description="DeepSORT-Tracking in Chunks via FFmpeg mit parallel Processing"
    )
    parser.add_argument("--deepsort-path", type=str, required=True, help="Pfad zum lokalen DeepSORT-Repository")
    parser.add_argument("--video", type=str, required=True, help="Pfad zum Eingabe-Video")
    parser.add_argument("--context-csv", type=str, required=True, help="CSV mit vorverarbeiteten Detektionen")
    parser.add_argument("--start-ids-csv", type=str, required=True, help="CSV mit Start-Track-IDs")
    parser.add_argument("--output-csv", type=str, required=True, help="Ziel-CSV für die Tracking-Ergebnisse")
    parser.add_argument("--frame-gap-threshold", type=int, default=10, help="Maximaler Frame-Abstand für Batch-Bildung")
    parser.add_argument("--workers", type=int, default=10, help="Anzahl paralleler Worker")
    args = parser.parse_args()

    DEEPSORT_PATH = args.deepsort_path
    VIDEO_PATH = args.video
    CONTEXT_CSV = args.context_csv
    START_IDS_CSV = args.start_ids_csv
    OUTPUT_CSV = args.output_csv
    FRAME_GAP_THRESHOLD = args.frame_gap_threshold
    NUM_WORKERS = args.workers

    logging.info("Starte paralleles Tracking mit Video: %s und %d Worker", VIDEO_PATH, NUM_WORKERS)

    # CSVs einlesen / Batches bilden
    context_df = pd.read_csv(CONTEXT_CSV)
    frames = context_df['frame'].sort_values().unique()
    batches, curr, last = [], [], None
    for f in frames:
        if last is None or f - last <= FRAME_GAP_THRESHOLD:
            curr.append(f)
        else:
            batches.append(curr)
            curr = [f]
        last = f
    if curr:
        batches.append(curr)
    logging.info("Insgesamt %d Batches erstellt", len(batches))

    # FPS bestimmen
    cap_probe = cv2.VideoCapture(VIDEO_PATH)
    fps = cap_probe.get(cv2.CAP_PROP_FPS)
    cap_probe.release()
    logging.info("Video-FPS: %.2f", fps)

    # DeepSORT Parameter
    ds_params = {
        "model_path": os.path.join(
            DEEPSORT_PATH,
            "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
        ),
        "max_dist": 0.2,
        "min_confidence": 0.3,
        "nms_max_overlap": 0.5,
        "max_iou_distance": 0.7,
        "max_age": 70,
        "n_init": 3,
        "nn_budget": 100,
        "use_cuda": torch.cuda.is_available()
    }

    # Output CSV initialisieren
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)
        logging.info("Vorhandene Output-CSV entfernt: %s", OUTPUT_CSV)
    cols = ["frame","x1","y1","x2","y2","conf","cls",
            "timestamp","original_index","track_id","cx","cy"]
    pd.DataFrame(columns=cols).to_csv(OUTPUT_CSV, index=False)
    logging.info("Output-CSV initialisiert: %s", OUTPUT_CSV)

    # Aufgabenliste vorbereiten
    tasks = [
        (idx, batch, DEEPSORT_PATH, VIDEO_PATH, CONTEXT_CSV, START_IDS_CSV, ds_params, fps)
        for idx, batch in enumerate(batches, start=1)
    ]


    # Paralleles Tracking starten
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_batch, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            idx, batch_out = fut.result()
            if batch_out:
                pd.DataFrame(batch_out).to_csv(OUTPUT_CSV, mode='a', header=False, index=False)
                logging.info("Batch %d: %d Zeilen zu CSV hinzugefügt", idx, len(batch_out))

    logging.info("Paralleles Tracking abgeschlossen. Ergebnisse in %s", OUTPUT_CSV)


if __name__ == '__main__':
    main()


