"""
pose_quality_check.py
-----------------------
Tools for checking whether the top player's pose tracking is
systematically worse than the bottom player's -- directly testing the
leading hypothesis for the top-hit recall weakness found in
evaluate_model()'s confusion matrix (recall 0.623 bottom vs 0.331 top).

1. compute_pose_detection_rates(): aggregate, quantitative comparison
   across every rally's pose CSVs -- per-keypoint detection rate for
   bottom vs top, across the whole dataset. This is the decisive check --
   if the top player's detection rate is systematically lower across many
   rallies, that's real evidence, not one rally's bad camera angle.
2. render_pose_overlay(): visual, per-rally overlay of both players'
   skeletons on the video, for spot-checking specific rallies flagged by
   the aggregate stats.

Usage:
    # Aggregate stats across your whole dataset:
    python pose_quality_check.py --stats --poses_dir data/poses

    # Visual overlay for one specific rally:
    python pose_quality_check.py --overlay \
        --video data/rally_video/35_set1_2.mp4 \
        --pose_bottom data/poses/35_set1_2_player_bottom.csv \
        --pose_top data/poses/35_set1_2_player_top.csv \
        --output 35_set1_2_pose_check.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# Standard COCO-17 keypoint order and skeleton connectivity -- matches the
# convention already used throughout this pipeline's pose CSVs (kp_0..kp_16).
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

SKELETON_BONES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms/shoulders
    (6, 12), (5, 11), (11, 12),               # torso
    (11, 13), (12, 14), (13, 15), (14, 16),   # legs
]


def load_pose_csv(csv_path: Path) -> np.ndarray:
    """Load a _player_bottom.csv or _player_top.csv into a (frames, 17, 2) array.

    Args:
        csv_path: Path to a pose CSV (34 columns: x0,y0,x1,y1,...,x16,y16,
            one row per frame -- matches format_data.py's
            pose_df_to_dense_bottom_top() output).

    Returns:
        Array of shape (n_frames, 17, 2).
    """
    df = pd.read_csv(csv_path)
    return df.to_numpy(dtype=np.float64).reshape(-1, 17, 2)


def compute_pose_detection_rates(poses_dir: Path, eps: float = 1e-6) -> pd.DataFrame:
    """Compute per-keypoint detection rates for bottom vs top players, across the whole dataset.

    A keypoint is counted as "detected" for a given (rally, frame) if its
    (x, y) isn't (0, 0) -- matching this pipeline's "undetected" sentinel
    convention throughout.

    Args:
        poses_dir: Directory containing *_player_bottom.csv /
            *_player_top.csv files (e.g. INPUT_DATA_DIR/poses).
        eps: Threshold below which a coordinate is treated as "undetected".

    Returns:
        DataFrame with one row per COCO keypoint, columns: keypoint,
        bottom_detection_rate, top_detection_rate, gap (bottom - top) --
        sorted by gap descending, so the most-affected keypoints show first.
    """
    bottom_files = sorted(poses_dir.glob("*_player_bottom.csv"))

    bottom_total = np.zeros(17)
    bottom_detected = np.zeros(17)
    top_total = np.zeros(17)
    top_detected = np.zeros(17)

    n_rallies = 0
    for bottom_path in bottom_files:
        stem = bottom_path.name.replace("_player_bottom.csv", "")
        top_path = poses_dir / f"{stem}_player_top.csv"
        if not top_path.is_file():
            print(f"[WARN] {stem}: no matching _player_top.csv, skipping")
            continue

        bottom_kp = load_pose_csv(bottom_path)
        top_kp = load_pose_csv(top_path)
        n_rallies += 1

        bottom_present = ~((np.abs(bottom_kp[:, :, 0]) < eps) & (np.abs(bottom_kp[:, :, 1]) < eps))
        top_present = ~((np.abs(top_kp[:, :, 0]) < eps) & (np.abs(top_kp[:, :, 1]) < eps))

        bottom_total += bottom_kp.shape[0]
        bottom_detected += bottom_present.sum(axis=0)
        top_total += top_kp.shape[0]
        top_detected += top_present.sum(axis=0)

    print(f"[INFO] Aggregated over {n_rallies} rallies")

    bottom_rate = bottom_detected / np.maximum(bottom_total, 1)
    top_rate = top_detected / np.maximum(top_total, 1)

    result = pd.DataFrame({
        "keypoint": COCO_KEYPOINT_NAMES,
        "bottom_detection_rate": bottom_rate,
        "top_detection_rate": top_rate,
        "gap": bottom_rate - top_rate,
    })
    return result.sort_values("gap", ascending=False).reset_index(drop=True)


def render_pose_overlay(
    video_path: Path,
    pose_bottom_path: Path,
    pose_top_path: Path,
    out_path: Path,
    bottom_colour: tuple[int, int, int] = (255, 150, 0),
    top_colour: tuple[int, int, int] = (0, 100, 255),
) -> dict[str, int]:
    """Overlay both players' pose skeletons onto a rally video, for visual QA.

    Args:
        video_path: Path to the rally video clip.
        pose_bottom_path: Path to the matching _player_bottom.csv.
        pose_top_path: Path to the matching _player_top.csv.
        out_path: Where to write the annotated output video.
        bottom_colour: BGR colour for the bottom player's skeleton.
        top_colour: BGR colour for the top player's skeleton.

    Returns:
        Dict with counts: {"total_frames", "bottom_fully_missing",
        "top_fully_missing"} -- "fully_missing" means every keypoint for
        that player was (0,0) on that frame (a total tracking dropout).

    Raises:
        FileNotFoundError: If video_path can't be opened.
    """
    bottom_kp = load_pose_csv(pose_bottom_path)
    top_kp = load_pose_csv(pose_top_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    def draw_skeleton(frame, kp, colour, eps=1e-6):
        present = ~((np.abs(kp[:, 0]) < eps) & (np.abs(kp[:, 1]) < eps))
        for a, b in SKELETON_BONES:
            if present[a] and present[b]:
                pa = (int(kp[a, 0]), int(kp[a, 1]))
                pb = (int(kp[b, 0]), int(kp[b, 1]))
                cv2.line(frame, pa, pb, colour, 2)
        for i in range(17):
            if present[i]:
                cv2.circle(frame, (int(kp[i, 0]), int(kp[i, 1])), 4, colour, -1)
        return int(present.sum())

    n_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), bottom_kp.shape[0], top_kp.shape[0])
    bottom_fully_missing = top_fully_missing = 0

    frame_idx = 0
    while frame_idx < n_frames:
        ok, frame = cap.read()
        if not ok:
            break

        n_bottom_present = draw_skeleton(frame, bottom_kp[frame_idx], bottom_colour)
        n_top_present = draw_skeleton(frame, top_kp[frame_idx], top_colour)
        if n_bottom_present == 0:
            bottom_fully_missing += 1
        if n_top_present == 0:
            top_fully_missing += 1

        cv2.putText(
            frame,
            f"frame {frame_idx}  bottom(blue): {n_bottom_present}/17  top(red): {n_top_present}/17",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    stats = {
        "total_frames": frame_idx,
        "bottom_fully_missing": bottom_fully_missing,
        "top_fully_missing": top_fully_missing,
    }
    print(f"[INFO] Rendered {frame_idx} frames -> {out_path}")
    print(f"[INFO] Bottom player fully missing (all 17 kp undetected): {bottom_fully_missing} frames")
    print(f"[INFO] Top player fully missing (all 17 kp undetected):    {top_fully_missing} frames")
    return stats


def main() -> None:
    """CLI entry point.

    Returns:
        None. Runs --stats and/or --overlay depending on the flags given.
    """
    parser = argparse.ArgumentParser(
        description="Check pose tracking quality for bottom vs top players."
    )
    parser.add_argument("--stats", action="store_true",
                        help="Compute aggregate detection-rate stats across all rallies")
    parser.add_argument("--overlay", action="store_true",
                        help="Render a visual pose overlay for one rally")
    parser.add_argument("--poses_dir", default=None, help="Directory of pose CSVs (for --stats)")
    parser.add_argument("--video", default=None, help="Rally video path (for --overlay)")
    parser.add_argument("--pose_bottom", default=None, help="_player_bottom.csv path (for --overlay)")
    parser.add_argument("--pose_top", default=None, help="_player_top.csv path (for --overlay)")
    parser.add_argument("--output", default=None, help="Output video path (for --overlay)")
    args = parser.parse_args()

    if args.stats:
        if not args.poses_dir:
            parser.error("--stats requires --poses_dir")
        result = compute_pose_detection_rates(Path(args.poses_dir))
        print("\n" + result.to_string(index=False))

    if args.overlay:
        if not (args.video and args.pose_bottom and args.pose_top):
            parser.error("--overlay requires --video, --pose_bottom, --pose_top")
        video_path = Path(args.video)
        output = Path(args.output) if args.output else video_path.parent / f"{video_path.stem}_pose_check.mp4"
        render_pose_overlay(video_path, Path(args.pose_bottom), Path(args.pose_top), output)


if __name__ == "__main__":
    main()
