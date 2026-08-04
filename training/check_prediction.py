import cv2
import pandas as pd

VIDEO_PATH = "data/rally_video/29_set1_1.mp4"
CSV_PATH = "data/29_set1_1_hit_predicted.csv"
OUTPUT_PATH = "labelled_video.mp4"

# Read labels (single-column CSV)
labels = pd.read_csv(CSV_PATH)["hit"].to_numpy(dtype=int)
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx < len(labels):
        label = labels[frame_idx]
    else:
        label = 0

    if label == 0:
        text = "NO HIT"
        colour = (255, 255, 255)      # white
    elif label == 1:
        text = "BOTTOM HIT"
        colour = (0, 0, 255)          # red
    elif label == 2:
        text = "TOP HIT"
        colour = (0, 255, 0)          # green
    else:
        text = f"UNKNOWN ({label})"
        colour = (0, 255, 255)

    # Background box
    cv2.rectangle(frame, (15, 15), (300, 75), (0, 0, 0), -1)

    # Label
    cv2.putText(
        frame,
        text,
        (25, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        colour,
        2,
        cv2.LINE_AA,
    )

    # Frame number (optional)
    cv2.putText(
        frame,
        f"Frame {frame_idx}",
        (25, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()

print(f"Saved labelled video to {OUTPUT_PATH}")
