import pandas as pd
from pathlib import Path

# === Step 1: Load data ===
detections_df = pd.read_csv("detections-3.csv")
conflicts_df = pd.read_csv("ocsort_conflicts.csv")
tracked_df = pd.read_csv("ocsort_tracked.csv")

# === Step 2: Group conflicts into batches with ±10 frame context ===
conflicts_df = conflicts_df.sort_values("frame").reset_index(drop=True)
conflicts_df["frame_diff"] = conflicts_df["frame"].diff().fillna(0)
frame_gap_threshold = 10

batches = []
current_batch = []
last_frame = None

for _, row in conflicts_df.iterrows():
    frame = int(row["frame"])
    if last_frame is None or frame - last_frame <= frame_gap_threshold:
        current_batch.append(frame)
    else:
        batches.append(current_batch)
        current_batch = [frame]
    last_frame = frame
if current_batch:
    batches.append(current_batch)

# Expand each batch by ±10 frames and remove duplicates
expanded_batches = []
for batch in batches:
    min_frame = max(min(batch) - 10, 0)
    max_frame = max(batch) + 10
    expanded_batches.append(list(range(min_frame, max_frame + 1)))

# === Step 3: Extract relevant detection and tracked entries ===
context_detections_list = []
initial_tracked_list = []

for frame_range in expanded_batches:
    # Pull context detections from original YOLO output
    detections = detections_df[detections_df["frame"].isin(frame_range)]
    context_detections_list.append(detections)

    # Get track IDs from first frame in batch
    first_frame = min(frame_range)
    tracked = tracked_df[tracked_df["frame"] == first_frame]
    initial_tracked_list.append(tracked)

# Combine all results
context_detections_df = pd.concat(context_detections_list, ignore_index=True)
initial_tracked_df = pd.concat(initial_tracked_list, ignore_index=True)

# === Output CSVs ===
context_detections_df.to_csv("deepsort_conflict_context.csv", index=False)
initial_tracked_df.to_csv("deepsort_conflict_start_ids.csv", index=False)

print("Exported:")
print("- deepsort_conflict_context.csv")
print("- deepsort_conflict_start_ids.csv")
