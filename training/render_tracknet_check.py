"""
render_tracknet_check.py
--------------------------
Overlay TrackNet's predicted shuttle position onto a rally video, so you
can visually verify tracking quality frame by frame -- useful for
diagnosing whether weak model performance traces back to noisy/missing
shuttle tracking rather than the model itself.

Usage:
    python render_tracknet_check.py \
        --video rally_video/29_set1_3.mp4 \
        --ball ball_trajectory/29_set1_3_ball_predicted.csv \
        --output 29_set1_3_tracknet_check.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd


def load_ball_predictions(csv_path: Path) -> dict[int, tuple[float, float, bool]]:
    """Load TrackNet's per-frame shuttle predictions.

    Args:
        csv_path: Path to a _ball_predicted.csv (Frame,Visibility,X,Y,Time).

    Returns:
        Dict mapping frame -> (x, y, visible), one entry per row in the file.
    """
    df = pd.read_csv(csv_path)
    predictions: dict[int, tuple[float, float, bool]] = {}
    for _, row in df.iterrows():
        frame = int(row["Frame"])
        visible = bool(int(row["Visibility"]))
        predictions[frame] = (float(row["X"]), float(row["Y"]), visible)
    return predictions


def render_overlay(
    video_path: Path,
    ball_csv_path: Path,
    out_path: Path,
    trail_length: int = 5,
) -> dict[str, int]:
    """Render a copy of the video with TrackNet's shuttle predictions overlaid.

    Draws a bright marker at the shuttle's predicted position on every
    frame it was detected, plus a short fading trail of recent positions
    (helps make trajectory smoothness/jitter visually obvious -- an
    erratic, jumpy trail is a strong visual signal of noisy tracking).
    Frames where the shuttle was NOT detected get an explicit on-screen
    warning rather than just showing nothing, so gaps in tracking are as
    easy to spot as bad positions are.

    Args:
        video_path: Path to the rally video clip.
        ball_csv_path: Path to the matching _ball_predicted.csv.
        out_path: Where to write the annotated output video.
        trail_length: Number of previous visible positions to draw as a
            fading trail.

    Returns:
        Dict with counts: {"total_frames", "visible", "not_detected",
        "no_csv_row"} -- summary stats, also printed to stdout.

    Raises:
        FileNotFoundError: If video_path can't be opened.
    """
    predictions = load_ball_predictions(ball_csv_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    trail: list[tuple[float, float]] = []
    n_visible = n_not_visible = n_no_prediction = 0

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx in predictions:
            x, y, visible = predictions[frame_idx]
            if visible:
                n_visible += 1
                trail.append((x, y))
                if len(trail) > trail_length:
                    trail.pop(0)

                for i, (tx, ty) in enumerate(trail):
                    fade = (i + 1) / len(trail)
                    radius = max(2, int(3 + 4 * fade))
                    colour = (0, int(180 * fade + 75), int(255 * fade))
                    cv2.circle(frame, (int(tx), int(ty)), radius, colour, -1)

                cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 0), 2)
                cv2.circle(frame, (int(x), int(y)), 6, (0, 255, 255), -1)
            else:
                n_not_visible += 1
                trail.clear()
                cv2.putText(frame, "SHUTTLE NOT DETECTED", (20, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            n_no_prediction += 1
            trail.clear()

        cv2.putText(frame, f"frame {frame_idx}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    stats = {
        "total_frames": frame_idx,
        "visible": n_visible,
        "not_detected": n_not_visible,
        "no_csv_row": n_no_prediction,
    }
    print(f"[INFO] Rendered {frame_idx} frames -> {out_path}")
    print(f"[INFO] Visible: {n_visible}, Not detected: {n_not_visible}, "
          f"No CSV row at all: {n_no_prediction}")
    if n_no_prediction > 0:
        print(f"[WARN] {n_no_prediction} frame(s) had no corresponding row "
              f"in the CSV at all -- check that the video and CSV are from "
              f"the same rally and haven't drifted out of frame-alignment.")
    return stats


def main() -> None:
    """CLI entry point.

    Returns:
        None. Writes the annotated video via render_overlay().
    """
    parser = argparse.ArgumentParser(
        description="Overlay TrackNet's predicted shuttle position onto a video, for visual QA."
    )
    parser.add_argument("--video", required=True, help="Path to the rally video clip")
    parser.add_argument("--ball", required=True, help="Path to the matching _ball_predicted.csv")
    parser.add_argument("--output", default=None,
                        help="Output video path (default: {video}_tracknet_check.mp4)")
    parser.add_argument("--trail_length", type=int, default=5,
                        help="Number of previous positions shown as a fading trail (default: 5)")
    args = parser.parse_args()

    video_path = Path(args.video)
    output = Path(args.output) if args.output else video_path.parent / f"{video_path.stem}_tracknet_check.mp4"

    render_overlay(video_path, Path(args.ball), output, trail_length=args.trail_length)


if __name__ == "__main__":
    main()