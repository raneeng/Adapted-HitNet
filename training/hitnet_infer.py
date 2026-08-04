"""
hitnet_infer.py
-----------------
Run a trained HitNet model (hitnet_model.h5) on a rally to predict hit
events -- this is the actual integration point connecting the trained
model back into the rest of the pipeline.

Windows overlap (stride 1, 12 frames each) -- a rally with N frames
produces N-11 overlapping predictions, and a given frame typically gets
"voted on" by up to 6 different windows (since only the last 6 frames of
each window's label window mattered during training -- see
build_rally_windows() in train_hitnet.py). 

Usage:
    python hitnet_infer.py \
        --model hitnet_model.h5 \
        --stem 29_set1_5 \
        --data_dir ~/hitnet/training/data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_hitnet import (
    NUM_CONSEC,
    FEATURES_PER_FRAME,
    read_court,
    flatten_court_corners,
    read_ball_trajectory,
    read_pose_csv,
    scale_data,
    stem_to_match_id,
)


def build_windows_from_arrays(
    ball_x: np.ndarray,
    ball_y: np.ndarray,
    bottom: np.ndarray,
    top: np.ndarray,
    corners: np.ndarray,
    num_consec: int = NUM_CONSEC,
) -> tuple[np.ndarray, int] | None:
    """Build windowed features directly from in-memory arrays, no file I/O.

    Core windowing logic shared by build_inference_windows() (which reads
    these same arrays from disk) and any caller that already has this
    data in memory -- e.g. bgl_to_hitnet.slice_rally_data()'s output,
    which would otherwise need writing to CSV and immediately reading it
    back for no reason.

    Args:
        ball_x, ball_y: Shuttle position per frame, shape (n_frames,) each.
        bottom, top: Dense pose arrays, shape (n_frames, 34) each --
            e.g. from bgl_to_hitnet.slice_rally_data()'s bottom_rally/top_rally.
        corners: Flattened court corners, shape (8,) -- e.g.
            flatten_court_corners(reorder_corners_for_hitnet(webapp_corners)).
        num_consec: Window size in frames (must match what the model was
            trained with).

    Returns:
        (x, n_frames): x has shape (n_windows, num_consec *
        FEATURES_PER_FRAME), already scaled to [1, 2]; n_frames is the
        rally's total frame count (needed to map window predictions back
        to frame indices). Returns None if the rally is too short to
        produce at least one window.
    """
    n = min(len(ball_x), bottom.shape[0], top.shape[0])
    if n < num_consec:
        print(f"[SKIP] rally has only {n} frames, need at least {num_consec}")
        return None

    ball_x, ball_y = ball_x[:n], ball_y[:n]
    bottom, top = bottom[:n], top[:n]

    x_list = []
    for i in range(num_consec):
        end = n - num_consec + i + 1
        x_bird = np.column_stack([ball_x[i:end], ball_y[i:end]])
        x_pose = np.hstack([bottom[i:end], top[i:end]])
        x_corners = np.tile(corners, (end - i, 1))
        x_list.append(np.hstack([x_bird, x_pose, x_corners]))

    x_t = np.hstack(x_list)
    return scale_data(x_t), n


def build_inference_windows(
    stem: str,
    data_dir: Path,
    num_consec: int = NUM_CONSEC,
) -> tuple[np.ndarray, int] | None:
    """Build windowed features for a rally by reading its files from disk.

    Thin wrapper around build_windows_from_arrays() -- reads the 4 files
    format_data.py produces per rally (no ground-truth _hit.csv needed,
    unlike training), then delegates the actual windowing to the shared
    in-memory function. Use build_windows_from_arrays() directly instead
    when the data is already in memory, to skip the file round-trip.

    Reuses the exact same readers already tested in train_hitnet.py
    (read_court, flatten_court_corners, read_ball_trajectory,
    read_pose_csv, scale_data), so this stays consistent with however
    training data was built.

    Args:
        stem: Rally stem, e.g. "29_set1_5".
        data_dir: Root data directory (contains ball_trajectory/, poses/,
            court/ subfolders -- i.e. format_data.py's INPUT_DATA_DIR).
        num_consec: Window size in frames (must match what the model was
            trained with).

    Returns:
        Same as build_windows_from_arrays(). Also returns None (with a
        [SKIP] message) if any of the 4 required files is missing.
    """
    match_id = stem_to_match_id(stem)
    ball_path = data_dir / "ball_trajectory" / f"{stem}_ball_predicted.csv"
    pose_bottom_path = data_dir / "poses" / f"{stem}_player_bottom.csv"
    pose_top_path = data_dir / "poses" / f"{stem}_player_top.csv"
    court_path = data_dir / "court" / f"{match_id}.out"

    for p in (ball_path, pose_bottom_path, pose_top_path, court_path):
        if not p.is_file():
            print(f"[SKIP] {stem}: missing {p}")
            return None

    ball_x, ball_y = read_ball_trajectory(ball_path)
    bottom = read_pose_csv(pose_bottom_path)
    top = read_pose_csv(pose_top_path)
    corners = flatten_court_corners(read_court(court_path))

    return build_windows_from_arrays(ball_x, ball_y, bottom, top, corners, num_consec=num_consec)


def aggregate_window_predictions(
    window_probs: np.ndarray,
    n_frames: int,
    num_consec: int = NUM_CONSEC,
    left_window: int = 6,
) -> np.ndarray:
    """Combine overlapping window-level predictions into one probability vector per frame.

    Window i (i = 0 .. n_windows-1) covers original frames [i, i + num_consec),
    and its label (during training) only looked at frames
    [i + left_window, i + num_consec) -- so window i's prediction is
    attributed here to exactly that frame range, matching training's own
    windowing convention. A given frame is typically covered by several
    windows; their probability vectors are averaged.

    Args:
        window_probs: Model output, shape (n_windows, 3) -- per-window
            class probabilities (no-hit / bottom-hit / top-hit).
        n_frames: The rally's total frame count.
        num_consec: Window size in frames (must match training).
        left_window: Matches train_hitnet.py's windowing convention --
            only frames [left_window:num_consec) of each window
            contributed to that window's label during training.

    Returns:
        Array of shape (n_frames, 3): averaged probability vector per
        frame. Frames with no covering window (can happen very close to
        the rally's start, before the first window's label range begins)
        get a pure [1, 0, 0] no-hit vector as a neutral default.
    """
    frame_probs_sum = np.zeros((n_frames, 3))
    frame_probs_count = np.zeros(n_frames)

    n_windows = window_probs.shape[0]
    for i in range(n_windows):
        start = i + left_window
        end = i + num_consec
        frame_probs_sum[start:end] += window_probs[i]
        frame_probs_count[start:end] += 1

    covered = frame_probs_count > 0
    result = np.zeros((n_frames, 3))
    result[covered] = frame_probs_sum[covered] / frame_probs_count[covered, None]
    result[~covered] = [1.0, 0.0, 0.0]  # no coverage -> default to no-hit
    return result


def predict_hits_from_windows(
    x: np.ndarray,
    n_frames: int,
    model: "tf.keras.Model",
    temperature: float = 1.0,
) -> np.ndarray:
    """Run the model on already-built windowed features and aggregate to per-frame predictions.

    Core prediction logic shared by predict_hits_for_rally() (file-based)
    and any caller with windows already built in memory -- e.g. via
    build_windows_from_arrays() on bgl_to_hitnet.slice_rally_data()'s output.

    Args:
        x: Windowed features -- from build_windows_from_arrays() or
            build_inference_windows().
        n_frames: The rally's total frame count (from the same source as x).
        model: A trained tf.keras Model (e.g. loaded from hitnet_model.h5).
        temperature: Optional calibration temperature (from
            train_hitnet.temp_scaling()) -- divides pre-softmax logits
            before the final softmax if not 1.0. Pass 1.0 (default) to
            skip calibration.

    Returns:
        Array of shape (n_frames,): the argmax class prediction per frame
        (0 = no-hit, 1 = bottom-hit, 2 = top-hit) -- same convention as
        ground-truth _hit.csv.
    """
    if temperature == 1.0:
        window_probs = model.predict(x, verbose=0)
    else:
        import tensorflow as tf
        logits_model = tf.keras.Model(model.input, model.layers[-2].output)
        logits = logits_model.predict(x, verbose=0)
        window_probs = tf.nn.softmax(logits / temperature).numpy()

    frame_probs = aggregate_window_predictions(window_probs, n_frames)
    return np.argmax(frame_probs, axis=1)


def predict_hits_for_rally(
    stem: str,
    data_dir: Path,
    model: "tf.keras.Model",
    temperature: float = 1.0,
) -> np.ndarray | None:
    """Run the full inference pipeline for one rally, reading its files from disk.

    Thin wrapper: build_inference_windows() (file-based) +
    predict_hits_from_windows(). Use predict_hits_from_windows() directly
    instead when the windowed features are already in memory, to skip the
    file round-trip.

    Args:
        stem: Rally stem, e.g. "29_set1_5".
        data_dir: Root data directory (format_data.py's INPUT_DATA_DIR).
        model: A trained tf.keras Model (e.g. loaded from hitnet_model.h5).
        temperature: Optional calibration temperature -- see
            predict_hits_from_windows().

    Returns:
        Same as predict_hits_from_windows(). Returns None if the rally's
        input files are missing/too short (see build_inference_windows()).
    """
    result = build_inference_windows(stem, data_dir)
    if result is None:
        return None
    x, n_frames = result
    return predict_hits_from_windows(x, n_frames, model, temperature=temperature)


def write_predicted_hit_csv(hit_predictions: np.ndarray, out_path: Path) -> None:
    """Write predicted per-frame hit labels in the same format as ground-truth _hit.csv.

    Args:
        hit_predictions: Array of shape (n_frames,), values in {0, 1, 2}.
        out_path: Where to write the CSV (e.g. {stem}_hit_predicted.csv).

    Returns:
        None.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"hit": hit_predictions}).to_csv(out_path, index=False)
    print(f"[INFO] Wrote {out_path} ({int((hit_predictions != 0).sum())} predicted hits "
          f"/ {len(hit_predictions)} frames)")


def main() -> None:
    """CLI entry point.

    Returns:
        None.
    """
    import tensorflow as tf

    parser = argparse.ArgumentParser(description="Run HitNet inference on a rally.")
    parser.add_argument("--model", required=True, help="Path to hitnet_model.h5")
    parser.add_argument("--stem", required=True, help="Rally stem, e.g. 29_set1_5")
    parser.add_argument("--data_dir", required=True, help="Root data directory")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Calibration temperature from temp_scaling() (default: 1.0, no calibration)")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: {data_dir}/shot/{stem}_hit_predicted.csv)")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    data_dir = Path(args.data_dir)
    predictions = predict_hits_for_rally(args.stem, data_dir, model, temperature=args.temperature)

    if predictions is None:
        print(f"[ERROR] Could not run inference for {args.stem} -- see [SKIP] message above.")
        return

    output = Path(args.output) if args.output else data_dir / f"{args.stem}_hit_predicted.csv"
    write_predicted_hit_csv(predictions, output)


if __name__ == "__main__":
    main()