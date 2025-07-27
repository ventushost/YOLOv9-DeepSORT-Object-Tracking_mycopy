import cv2
import numpy as np
import pandas as pd

# === KONFIGURATION ===
VIDEO_PATH = "/Users/erik/Downloads/code_csv_to_diagram/YOLOv9-DeepSORT-Object-Tracking_mycopy/event=test_testevent_2025_camip=10.0.0.111_date=2025-06-25_08-21-18.740.mkv"  # <- Hier den echten Pfad zur Videodatei einfügen
CSV_PATH = "ocsort_tracked.csv"
OUTPUT_CSV_PATH = "ocsort_tracked_filtered.csv"
SECONDS_OFFSET = 5  # Wir überspringen die ersten 5 Sekunden

# === VIDEOFRAME LADEN ===
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_number = int(fps * SECONDS_OFFSET)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Konnte kein Frame aus dem Video laden – überprüfe den Videopfad.")

# === POLYGON MIT MAUS ZEICHNEN ===
polygon_points = []

def draw_polygon(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        cv2.circle(frame_copy, (x, y), 3, (0, 255, 0), -1)

cv2.namedWindow("Wähle Polygonbereich (ESC zum Beenden)")
cv2.setMouseCallback("Wähle Polygonbereich (ESC zum Beenden)", draw_polygon)

frame_copy = frame.copy()
while True:
    display_frame = frame_copy.copy()
    if polygon_points:
        cv2.polylines(display_frame, [np.array(polygon_points)], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imshow("Wähle Polygonbereich (ESC zum Beenden)", display_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()

if len(polygon_points) < 3:
    raise ValueError("Du musst mindestens drei Punkte für ein gültiges Polygon wählen.")

# === CSV LADEN UND FILTERN ===
df = pd.read_csv(CSV_PATH)

# Mittelpunkt der Bounding Box berechnen
df["cx"] = (df["x1"] + df["x2"]) / 2
df["cy"] = (df["y1"] + df["y2"]) / 2

# Prüfen, ob Mittelpunkt im Polygon liegt
polygon_np = np.array(polygon_points, np.int32)
polygon_mask = df.apply(lambda row: cv2.pointPolygonTest(polygon_np, (row["cx"], row["cy"]), False) >= 0, axis=1)

# Nur IDs behalten, die mindestens einmal im Polygon auftauchen
valid_ids = df.loc[polygon_mask, "track_id"].unique()
filtered_df = df[df["track_id"].isin(valid_ids)].drop(columns=["cx", "cy"])

# === SPEICHERN ===
filtered_df.to_csv(OUTPUT_CSV_PATH, index=False)
print(f"[✓] Gefilterte CSV gespeichert unter: {OUTPUT_CSV_PATH}")
