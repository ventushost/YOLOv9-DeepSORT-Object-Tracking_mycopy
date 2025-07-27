import pandas as pd

# === File paths ===
deepsort_csv = "deepsort_conflict_fixed.csv"
ocsort_csv = "ocsort_tracked.csv"
conflict_csv = "ocsort_conflicts.csv"
output_csv = "ocsort_tracked_merged.csv"

# === Load data ===
deepsort_df = pd.read_csv(deepsort_csv)
ocsort_df = pd.read_csv(ocsort_csv)
conflict_df = pd.read_csv(conflict_csv)

# === Round bbox values for matching ===
bbox_cols = ["x1", "y1", "x2", "y2"]
deepsort_df[bbox_cols] = deepsort_df[bbox_cols].round(1)
ocsort_df[bbox_cols] = ocsort_df[bbox_cols].round(1)
conflict_df[bbox_cols] = conflict_df[bbox_cols].round(1)

# === Detect batches in DeepSORT ===
deepsort_df = deepsort_df.sort_values("frame").reset_index(drop=True)
deepsort_df["frame_diff"] = deepsort_df["frame"].diff().fillna(1)
deepsort_df["new_batch"] = deepsort_df["frame_diff"] > 10
deepsort_df["batch_id"] = deepsort_df["new_batch"].cumsum()

# === Build ID mapping from first frame of each batch ===
first_frames_map = deepsort_df.groupby("batch_id")["frame"].min().reset_index()
id_mapping = []

for _, row in first_frames_map.iterrows():
    frame_num = row["frame"]

    ds_frame = deepsort_df[deepsort_df["frame"] == frame_num][["x1", "y1", "x2", "y2", "track_id"]]
    oc_frame = ocsort_df[ocsort_df["frame"] == frame_num][["x1", "y1", "x2", "y2", "track_id"]]

    merged = pd.merge(ds_frame, oc_frame, on=["x1", "y1", "x2", "y2"], suffixes=("_deepsort", "_ocsort"))

    for _, m in merged.iterrows():
        id_mapping.append((m["track_id_deepsort"], m["track_id_ocsort"]))

# Apply ID mapping
id_mapping_dict = dict(id_mapping)
deepsort_df["track_id"] = deepsort_df["track_id"].map(id_mapping_dict).fillna(deepsort_df["track_id"])

# === Remove conflicting rows from OC-SORT ===
conflict_keys = conflict_df[["frame", "x1", "y1", "x2", "y2"]].apply(tuple, axis=1)
ocsort_keys = ocsort_df[["frame", "x1", "y1", "x2", "y2"]].apply(tuple, axis=1)
ocsort_df_cleaned = ocsort_df[~ocsort_keys.isin(conflict_keys)].copy()

# === Merge and sort final result ===
merged_df = pd.concat([ocsort_df_cleaned, deepsort_df], ignore_index=True)
merged_df = merged_df.sort_values(by=["frame", "track_id"])
merged_df.to_csv(output_csv, index=False)

print(f"[✓] Merged file written to: {output_csv}")
