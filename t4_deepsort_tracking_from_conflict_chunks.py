#!/usr/bin/env python3
import os
import sys
import logging
import argparse
import traceback
import time
import threading
import queue
from pathlib import Path

import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
try:
    from decord import VideoReader, cpu
except ImportError:
    logging.basicConfig(level=logging.ERROR)
    LOG = logging.getLogger("deep_sort")
    LOG.error("Required library 'decord' not found. Install it via 'pip install decord' and retry.")
    sys.exit(1)

# Fix deprecated numpy aliases
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)5s] %(message)s",
    datefmt="%H:%M:%S"
)
LOG = logging.getLogger("deep_sort")

# === Argparser ===
def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepSORT Tracking mit Loader- und Consumer-Threads"
    )
    parser.add_argument("-d", "--deepsort-path", required=True,
                        help="Pfad zum YOLOv9-DeepSORT-Object-Tracking-Repo")
    parser.add_argument("-v", "--video", required=True,
                        help="Pfad oder URL zur Videodatei")
    parser.add_argument("-c", "--context-csv", required=True,
                        help="Pfad zur Context-CSV")
    parser.add_argument("-s", "--start-ids-csv", required=True,
                        help="Pfad zur Start-IDs-CSV")
    parser.add_argument("-o", "--output-csv", required=True,
                        help="Pfad zur Ausgabe-CSV")
    parser.add_argument("-g", "--frame-gap", type=int, default=10,
                        help="Maximaler Frame-Abstand für Batches")
    parser.add_argument("--half", action="store_true",
                        help="FP16 für ReID-Modell nutzen, falls möglich")
    parser.add_argument("--compile", dest="compile_model", action="store_true",
                        help="torch.compile auf update() anwenden")
    parser.add_argument("--profile", action="store_true",
                        help="Profiling der Update-Zeiten aktivieren")
    parser.add_argument("--tqdm-update", type=int, default=20,
                        help="tqdm-Update-Intervall")
    parser.add_argument("--expected-fps", type=float, default=None,
                        help="Erwartete Framerate prüfen (Warnung bei Abweichung)")
    parser.add_argument("--mem-cap-gb", type=float, default=10.0,
                        help="Maximaler Speicher für Frames und Context in GB")
    return parser.parse_args()

# === DeepSORT factory ===
def make_deepsort(repo_path, half=False, compile_model=False):
    from deep_sort_pytorch.deep_sort.deep_sort import DeepSort
    ckpt = os.path.join(repo_path, "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7")
    ds = DeepSort(
        model_path=ckpt,
        max_dist=0.2, min_confidence=0.3, nms_max_overlap=0.5,
        max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100,
        use_cuda=torch.cuda.is_available()
    )
    if half:
        try:
            ds.model = ds.model.half()
            LOG.info("DeepSort ReID model auf FP16 umgestellt")
        except Exception:
            LOG.warning("FP16-Konvertierung fehlgeschlagen")
    if compile_model:
        try:
            ds.update = torch.compile(ds.update, mode="reduce-overhead")
            LOG.info("torch.compile auf DeepSort.update angewendet")
        except Exception as e:
            LOG.warning(f"torch.compile fehlgeschlagen: {e}")
    return ds

# === Batch Buffer Loader ===
class BatchLoader:
    def __init__(self, batches, vr, context_df, mem_cap_bytes):
        self.batches = batches
        self.vr = vr
        self.context_df = context_df
        self.mem_cap = mem_cap_bytes
        self.current_mem = 0
        self.queue = queue.Queue()
        self.lock = threading.Condition()
        self.finished = False
        # start thread
        self.thread = threading.Thread(target=self._load_batches, daemon=True)
        self.thread.start()

    def _load_batches(self):
        for batch in self.batches:
            idxs = np.array(batch, dtype=np.int32) - 1
            frames = self.vr.get_batch(idxs).asnumpy()
            ctx = self.context_df[self.context_df['frame'].isin(batch)].copy()
            # estimate memory
            size = frames.nbytes + ctx.memory_usage(deep=True).sum()
            with self.lock:
                while self.current_mem + size > self.mem_cap:
                    self.lock.wait(timeout=0.1)
                # enqueue
                self.queue.put((batch, frames, ctx))
                self.current_mem += size
        with self.lock:
            self.finished = True
            self.lock.notify_all()

    def get_batch(self, timeout=None):
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def batch_done(self, batch, frames, ctx):
        # free memory
        size = frames.nbytes + ctx.memory_usage(deep=True).sum()
        with self.lock:
            self.current_mem -= size
            self.lock.notify_all()

    def has_more(self):
        with self.lock:
            return not (self.finished and self.queue.empty())

# === Main ===
def main():
    args = parse_args()
    logging.getLogger().setLevel(logging.DEBUG)
    LOG.info("=== Script Start ===")

    # Prepare DeepSORT imports
    ds_root = Path(args.deepsort_path) / "deep_sort_pytorch"
    sys.path.append(str(ds_root))
    from deep_sort_pytorch.deep_sort.deep_sort import DeepSort
    from deep_sort_pytorch.deep_sort.sort.tracker import Tracker

    # Load CSVs once
    t0 = time.time()
    context_df = pd.read_csv(args.context_csv)
    start_ids_df = pd.read_csv(args.start_ids_csv)
    LOG.info(f"CSV geladen in {time.time()-t0:.2f}s: context={len(context_df)}, start_ids={len(start_ids_df)}")

    # Compute batches
    frames = sorted(context_df['frame'].unique())
    batches, curr, last = [], [], None
    for f in frames:
        if last is None or f - last <= args.frame_gap:
            curr.append(f)
        else:
            batches.append(curr); curr=[f]
        last = f
    if curr: batches.append(curr)
    LOG.info(f"{len(batches)} Batches erstellt (gap≤{args.frame_gap})")

    # Open VideoReader
    vr = VideoReader(args.video, ctx=cpu(0), num_threads=4)
    total_frames = len(vr)
    LOG.info(f"Video geöffnet: total_frames={total_frames}")

    # FPS check
    actual_fps = vr.get_avg_fps()
    LOG.info(f"Detected video FPS: {actual_fps:.2f}")
    if args.expected_fps and abs(actual_fps - args.expected_fps)>0.1:
        LOG.warning(f"Erwartete FPS {args.expected_fps}, aber {actual_fps:.2f} erkannt.")

    # Start batch loader
    mem_cap = int(args.mem_cap_gb * 1024**3)
    loader = BatchLoader(batches, vr, context_df, mem_cap)

    # Instantiate DeepSort
    deepsort = make_deepsort(args.deepsort_path, half=args.half, compile_model=args.compile_model)

    # Prepare output CSV writer
    out_f = open(args.output_csv, 'w', buffering=1)
    header = 'frame,x1,y1,x2,y2,conf,cls,track_id' + '\n'
    out_f.write(header)

        # Compute total frames across all batches for progress
    total_batch_frames = sum(len(batch) for batch in batches)
    LOG.info(f"Total frames to process (all batches): {total_batch_frames}")
    # Progress bar over total batch frames
    pbar = tqdm(total=total_batch_frames, desc='Batch 1/{}'.format(len(batches)), unit='frame', miniters=args.tqdm_update, dynamic_ncols=True)
    pbar = tqdm(total=total_batch_frames, desc=f'Batch 1/{len(batches)}', unit='frame', miniters=args.tqdm_update, dynamic_ncols=True)
    update_times = []

        # Consumer loop
    processed_frames = 0
    while loader.has_more():
        item = loader.get_batch(timeout=1.0)
        if item is None:
            continue
        batch, frames_arr, ctx = item
        bi = batches.index(batch) + 1
        LOG.info(f"--- Consuming Batch {bi}/{len(batches)}: Frames {batch[0]}–{batch[-1]} ({len(batch)}) ---")
        # reset tracker
        deepsort.tracker = Tracker(
            metric=deepsort.tracker.metric,
            max_iou_distance=deepsort.tracker.max_iou_distance,
            max_age=deepsort.tracker.max_age,
            n_init=deepsort.tracker.n_init
        )
        # process frames
        for fn, frame in zip(batch, frames_arr):
            # Update progress bar description for current batch
            pbar.set_description(f"Batch {bi}/{len(batches)}")
            # process detections
            ...
            pbar.update(1)
            processed_frames += 1
            processed_frames += 1
            processed_frames += 1
            continue
            arr = dets[['x1','y1','x2','y2','conf']].values
            xy1 = arr[:,:2]; wh = arr[:,2:4] - xy1; centers = xy1 + wh/2
            bbox_xywh = torch.from_numpy(np.hstack((centers, wh)))
            confs = torch.from_numpy(arr[:,4])
            t1 = time.time()
            with torch.no_grad():
                outs = deepsort.update(bbox_xywh, confs, [0]*len(dets), frame)
            if args.profile:
                update_times.append(time.time()-t1)
            for x1,y1,x2,y2,tid,cls in outs:
                out_f.write(f"{fn},{x1},{y1},{x2},{y2},1.0,{cls},{tid}\n")
            pbar.update(1)
            processed_frames += 1
        # batch done, free memory
        loader.batch_done(batch, frames_arr, ctx)
        LOG.info(f"Overall progress: {processed_frames}/{total_batch_frames} frames ({processed_frames/total_batch_frames:.2%})")

    pbar.close(); out_f.close()
    if args.profile and update_times:
        avg = sum(update_times)/len(update_times)
        LOG.info(f"Update calls: {len(update_times)}, avg time: {avg:.4f}s")
    LOG.info("Processing complete.")

if __name__=='__main__':
    try: main()
    except Exception:
        LOG.error("Unhandled error:\n"+traceback.format_exc())
        sys.exit(1)



