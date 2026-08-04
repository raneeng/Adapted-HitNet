"""
train_hitnet.py
----------------
ADAPTED FROM:
https://github.com/jhwang7628/monotrack/blob/main/modified-tracknet/train-hitnet.ipynb

Trains a HitNet-style GRU hit-detector on your own ShuttleSet-derived data
(produced by format_data.py), replacing the original notebook's ai_badminton
dependency, per-match-subfolder assumption, and pose-distance hitter
inference with direct reads of your actual files and ground-truth labels.
 
Court corner convention (must match your .out files exactly):
    A -------- D
    |          |
    G -------- X
    Order in the file: 1. Top-left (A)  2. Top-right (D)
                        3. Bottom-left (G)  4. Bottom-right (X)
    These are the OUTER (doubles) corners -- see the corner-order discussion
    from this conversation for why singles corners would silently distort
    every downstream court-relative coordinate.

References:
    1. https://github.com/jhwang7628/monotrack
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import tensorflow as tf
import math

from pathlib import Path
import gc
import os
import pandas as pd
import numpy as np
import cv2
import random
from scipy.stats import mode
from scipy.ndimage import shift
from skimage.transform import rescale, resize

from court_side import resolve_bottom_players_for_set  # noqa: F401 (kept for reference)


# ====== DIRECTORY STRUCTURE ==================================================

BASE_DIR = Path.cwd().parent
DATA_PREP_DIR = BASE_DIR / "data_prep"
SHUTTLESET_DIR = DATA_PREP_DIR / "shuttleset"
WEIGHTS_DIR = DATA_PREP_DIR / "weights"
YOLO_WEIGHTS_PATH = WEIGHTS_DIR / "yolov8s-pose.pt"
TRACKNET_WEIGHTS_DIR = WEIGHTS_DIR

TRAINING_DIR = BASE_DIR / "training"
INPUT_DATA_DIR = TRAINING_DIR / "data"
VIDEO_DIR = INPUT_DATA_DIR / "videos"
RALLY_VID_DIR = INPUT_DATA_DIR / "rally_video"
COURT_DIR = INPUT_DATA_DIR / "court"
POSES_DIR = INPUT_DATA_DIR / "poses"
BALL_TRAJ_DIR = INPUT_DATA_DIR / "ball_trajectory"
SHOT_DIR = INPUT_DATA_DIR / "shot"

MODEL_OUT_PATH = TRAINING_DIR / "hitnet_model.h5"


# ====== HYPERPARAMS ==========================================================

NUM_CONSEC = 12       # frames per window
# window label looks at frames 7-12 only (see paper)
# AKA: 12-frame window
# 1 2 3 4 5 6 | 7 8 9 10 11 12
#              ^
#       prediction region
LEFT_WINDOW = 6
N_KEYPOINTS = 17
FEATURES_PER_FRAME = 2 + (N_KEYPOINTS * 2) * 2 + 8  # shuttle (2) + bottom (34) + top (34) + corners (8) = 78
# over a 12-frame window: 12 × 78 = 936 -> every training sample is a vector of length 936.
 
BATCH_SIZE = 64
COURT_W = 6.1   # doubles width (m) -- see corner-order discussion
COURT_L = 13.4  # doubles length (m)

# ====== TRAIN / VALIDATION SPLIT =============================================
# explicit split; split by match -> unseen
# matches in ./data:
# 1 to 7, 21 to 26, 28 to 44 -- id 27 video unavailable on youtube
# total of 30 matches
VAL_MATCH_IDS = {"7", "21", "29", "30", "31", "36"} # 5 matches for 80/20 split


# ====== FILE READERS ==========================================================
# matching format_data.py's ACTUAL output (Deliberately not reusing ai_badminton.
# trajectory.Trajectory / ai_badminton.pose.read_player_poses -- those have their 
# own quirks (positional hit-column access, a required `frame` column on pose CSVs)
# that don't match what format_data.py currently writes.

def read_court(court_path: Path) -> list[list[float]]:
    """Read court corner coordinates from a semicolon-delimited .out file.
 
    Matches the reference notebook's read_court() convention exactly --
    returns raw points, not a flattened feature vector -- so this stays
    reusable for anything (visualization, sanity-checking, other consumers),
    not committed to one specific downstream use.
 
    Args:
        court_path: Path to a {match_id}.out file -- 4 lines, "x;y" per line,
            in A (top-left), D (top-right), G (bottom-left), X (bottom-right)
            order (the doubles/outer corners).
 
    Returns:
        List of 4 [x, y] float pairs, one per line, in file order (A, D, G, X).
        A -------- D
        |          |
        |          |
        G -------- X
        
    Raises:
        ValueError: If the file doesn't contain exactly 4 lines.
    """
    with open(court_path) as f:
        pts = [[float(v) for v in line.strip().split(";")] for line in f if line.strip()]
    if len(pts) != 4:
        raise ValueError(f"{court_path} has {len(pts)} lines, expected 4 (A, D, G, X)")
    return pts 


def flatten_court_corners(court_pts: list[list[float]]) -> np.ndarray:
    """Flatten a list of 4 court corner points into the 8-value feature vector layout.
 
    No reindexing needed -- unlike the reference notebook's
    corners = np.array([court_pts[1], court_pts[2], court_pts[0], court_pts[3]]).flatten(),
    which reorders because its .out convention differs from ours. Since our
    .out files are already written in A, D, G, X order (matching the world
    corner order build_rally_windows() expects), this is a straight flatten.
 
    Args:
        court_pts: Output of read_court() -- 4 [x, y] pairs, in A, D, G, X order.
 
    Returns:
        Flat array of 8 values: [Ax, Ay, Dx, Dy, Gx, Gy, Xx, Xy]. --- corners in train-hitnet
    """
    return np.array(court_pts, dtype=np.float64).flatten()
 
 
def read_ball_trajectory(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a _ball_predicted.csv's X/Y columns.
 
    Args:
        csv_path: Path to a {stem}_ball_predicted.csv (Frame,Visibility,X,Y,Time).
 
    Returns:
        (X, Y) arrays, one value per frame, in file order. No interpolation
        or cleanup -- raw values, including 0s at undetected frames, matching
        write_prediction_csv()'s output exactly.
    """
    df = pd.read_csv(csv_path)
    return df["X"].to_numpy(dtype=np.float64), df["Y"].to_numpy(dtype=np.float64)
 
 
def read_hit_labels(csv_path: Path) -> np.ndarray:
    """Read a _hit.csv's 3-way label column, by name (not position).
 
    Args:
        csv_path: Path to a {stem}_hit.csv (single `hit` column, values in
            {0, 1, 2}: no-hit / bottom-player-hit / top-player-hit).
 
    Returns:
        Array of int, one value per frame.
    """
    return pd.read_csv(csv_path)["hit"].to_numpy(dtype=np.int64)
 
 
def read_pose_csv(csv_path: Path) -> np.ndarray:
    """Read a _player_bottom.csv or _player_top.csv (34 columns, no frame column).
 
    Args:
        csv_path: Path to a {stem}_player_bottom.csv or _player_top.csv --
            columns x0,y0,x1,y1,...,x16,y16, one row per frame (matches
            format_data.py's pose_df_to_dense_bottom_top() output exactly).
 
    Returns:
        Array of shape (n_frames, 34).
    """
    df = pd.read_csv(csv_path)
    return df.to_numpy(dtype=np.float64)
    

# ====== RALLY DISCOVERY =======================================================
 
def discover_rally_stems() -> list[str]:
    """ 
    Returns:
        List of stems (e.g. "29_set1_3"), one per {stem}_hit.csv found,
        sorted.
    """
    return sorted(p.name.removesuffix("_hit.csv") for p in SHOT_DIR.glob("*_hit.csv"))

 
def stem_to_match_id(stem: str) -> str:
    """Extract the match id from a rally stem.
 
    Args:
        stem: e.g. "29_set1_3".
 
    Returns:
        The match id, e.g. "29" -- used to look up the per-match court file
        and to decide train-vs-validation membership.
    """
    return stem.split("_set")[0]
 
 
def rally_file_paths(stem: str) -> dict[str, Path]:
    """Compute every input file path for one rally stem.
 
    Args:
        stem: e.g. "29_set1_3".
 
    Returns:
        Dict with keys "ball", "hit", "pose_bottom", "pose_top", "court",
        "video" -- "court" uses the match-level id (stem_to_match_id), every
        other path uses the full rally stem.
    """
    match_id = stem_to_match_id(stem)
    return {
        "ball": BALL_TRAJ_DIR / f"{stem}_ball_predicted.csv",
        "hit": SHOT_DIR / f"{stem}_hit.csv",
        "pose_bottom": POSES_DIR / f"{stem}_player_bottom.csv",
        "pose_top": POSES_DIR / f"{stem}_player_top.csv",
        "court": COURT_DIR / f"{match_id}.out",
        "video": RALLY_VID_DIR / f"{stem}.mp4",
    }


# ====== FEATURE EXTRACTION: WINDOWS =========================================
# from the reference train-hitnet notebook, confirmed against the paper's 
# num_consec=12 / left_window=6 spec
 
def build_rally_windows(
    stem: str,
    num_consec: int = NUM_CONSEC,
    left_window: int = LEFT_WINDOW,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build windowed (x, y) training examples for one rally.
 
    Args:
        stem: Rally stem, e.g. "29_set1_3".
        num_consec: Window size in frames.
        left_window: Only frames [left_window:num_consec) of each window
            contribute to that window's label (paper: predicts a hit within
            the last 6 of 12 frames).
 
    Returns:
        (x_t, y_t): x_t has shape (n_windows, num_consec * FEATURES_PER_FRAME),
        y_t has shape (n_windows,) with values in {0, 1, 2}. Returns None if
        any required file is missing or the rally is too short to produce
        at least one window.
    """
    paths = rally_file_paths(stem)
    for key in ("ball", "hit", "pose_bottom", "pose_top", "court"):
        if not paths[key].is_file():
            print(f"[SKIP] {stem}: missing {paths[key]}")
            return None
 
    ball_x, ball_y = read_ball_trajectory(paths["ball"])
    hit = read_hit_labels(paths["hit"])
    bottom = read_pose_csv(paths["pose_bottom"])
    top = read_pose_csv(paths["pose_top"])
    court_pts = read_court(paths["court"])
    corners = flatten_court_corners(court_pts) # ADGX
 
    n = min(len(ball_x), len(hit), bottom.shape[0], top.shape[0])
    if n < num_consec:
        print(f"[SKIP] {stem}: only {n} frames, need at least {num_consec}")
        return None
 
    ball_x, ball_y, hit = ball_x[:n], ball_y[:n], hit[:n]
    bottom, top = bottom[:n], top[:n]
 
    x_list, y_list = [], []
    for i in range(num_consec):
        end = n - num_consec + i + 1
        x_bird = np.column_stack([ball_x[i:end], ball_y[i:end]])
        x_pose = np.hstack([bottom[i:end], top[i:end]])
        x_corners = np.tile(corners, (end - i, 1))
        x = np.hstack([x_bird, x_pose, x_corners])
        x_list.append(x)
        y_list.append(hit[i:end])
 
    x_t = np.hstack(x_list)
    y_t = np.max(np.column_stack(y_list[left_window:]), axis=1)
    return x_t, y_t


# ===== DATASET ASSEMBLY =======================================================
 
def build_dataset(stems: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Build a full (scaled) feature/label dataset from a list of rally stems.
 
    Args:
        stems: Rally stems to include.
 
    Returns:
        (x, y): x has shape (N, NUM_CONSEC * FEATURES_PER_FRAME), scaled to
        [1, 2] (undetected entries stay 0). y has shape (N,), values in
        {0, 1, 2}. Rallies that fail to load (see build_rally_windows) are
        skipped, not fatal to the whole run.
    """
    x_all, y_all = [], []
    for stem in stems:
        result = build_rally_windows(stem)
        if result is None:
            continue
        x_t, y_t = result
        x_all.append(scale_data(x_t))
        y_all.append(y_t)
 
    if not x_all:
        raise RuntimeError(f"No usable rallies found among {len(stems)} stems given.")
 
    return np.vstack(x_all), np.hstack(y_all)
 
 
def rebalance_classes(x: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Subsample no-hit examples so hit/non-hit classes is roughly equally represented.
 
    Args:
        x: Feature array, shape (N, ...).
        y: Label array, shape (N,), values in {0, 1, 2}.
        seed: RNG seed for reproducibility.
 
    Returns:
        (x_balanced, y_balanced), shuffled, with the no-hit (0) class
        subsampled down to roughly match the combined size of the two hit
        classes (1 and 2).
    """
    rng = np.random.default_rng(seed)
    idx_no_hit = np.where(y == 0)[0]
    idx_hit = np.where(y != 0)[0]
 
    if len(idx_hit) == 0:
        raise RuntimeError("No hit examples found at all -- check your _hit.csv files.")
 
    keep_no_hit = rng.choice(idx_no_hit, size=min(len(idx_no_hit), len(idx_hit)), replace=False)
    keep = np.concatenate([keep_no_hit, idx_hit])
    rng.shuffle(keep)
    return x[keep], y[keep]
 
 
# ===== TRAIN/VAL SPLIT =========================================================
 
def split_stems_by_match(
    stems: list[str],
    val_match_ids: set[str] = VAL_MATCH_IDS,
) -> tuple[list[str], list[str]]:
    """Split rally stems into train/validation sets by match id.
 
    Args:
        stems: All discovered rally stems.
        val_match_ids: Match ids to hold out for validation -- every rally
            belonging to these matches goes to validation, every other
            rally goes to training. No individual rally is ever split
            across both sets, and no two rallies from the same match ever
            land on opposite sides -- see the leave-one-match-out
            discussion for why this matters.
 
    Returns:
        (train_stems, val_stems).
    """
    train, val = [], []
    for stem in stems:
        (val if stem_to_match_id(stem) in val_match_ids else train).append(stem)
    return train, val
 

# copied functinos from train-hitnet.ipynb from hitnet repo

def visualize(x, y):
    """Scatter-plot windowed feature vectors, colored by their 3-way hit label.

    Args:
        x: Feature array; only the last 4 timesteps' shuttle (x, y)
            coordinates are actually plotted.
        y: Label array, values in {0, 1, 2} (no-hit / bottom-hit / top-hit).

    Returns:
        None. Displays a matplotlib scatter plot.
    """
    print(x.shape, y.shape)
    cdict = {0: 'red', 1: 'blue', 2: 'green'}
    plt.figure()
    for g in np.unique(y):
        ix = np.where(y == g)
        plt.scatter(*x[ix, -4:, :].T, c=cdict[g], label=g)
        plt.scatter(*x[ix, 0, :].T, c=cdict[g], label=g)
    plt.show()

    
def resample(series, s):
    """Resample a 1D or 2D time series to a different playback speed.

    Args:
        series: Array to resample (e.g. shuttle X or Y coordinates).
        s: Speed multiplier -- s<1 slows down (more samples out), s>1
            speeds up (fewer samples out).

    Returns:
        Resampled array, length scaled by s.
    """
    flatten = False
    if len(series.shape) == 1:
        series.resize((series.shape[0], 1))
        series = series.astype('float64')
        flatten = True
    series = resize(
        series, (int(s * series.shape[0]), series.shape[1]),
    )
    if flatten:
        series = series.flatten()
    return series   

 
# ====== DATA AUGMENTATION ============================================

def reflect(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """ Augmentation: Mirror "video" horizontally 
 
    Negates every x-coordinate column, leaving y-coordinates and
    "undetected" (near-zero) entries untouched. Applied consistently
    across every point in a row (shuttle, both players' poses, court
    corners)
    
    Args:
        x: Feature array with alternating x/y coordinate columns, already
            scaled to [1, 2] (see scale_data()) with 0 marking "undetected".
        eps: Threshold below which a value is treated as "undetected".
 
    Returns:
        A new array with every x-coordinate column negated, except where
        the original value was ~0, left untouched.
    """
    x = x.copy()
    even_cols = list(range(0, x.shape[1], 2))
    block = x[:, even_cols]
    undetected = np.abs(block) < eps
    x[:, even_cols] = np.where(undetected, block, -block)
    return x
 
 
def drop_consecutive(
    x: np.ndarray,
    rep_value: float = 0.0,
    num_consec: int = NUM_CONSEC,
    features_per_frame: int = FEATURES_PER_FRAME,
) -> np.ndarray:
    """Augmentation: Erase several consecutive frames. 
    Simulates: occlusion motion blur temporary tracking failure
    
    Zero out one random contiguous run of frames per row.
    picks a random start frame and length (both within the window) 
    independently per row, and overwrites every feature belonging to those 
    frames.
    
    Args:
        x: Feature array, shape (N, num_consec * features_per_frame).
        rep_value: Value used to overwrite the dropped frames (default 0,
            matching the "undetected" sentinel elsewhere in this pipeline).
        num_consec: Frames per window.
        features_per_frame: Feature columns per frame.
 
    Returns:
        Augmented copy of x, one random contiguous frame-run per row
        overwritten with rep_value.
    """
    x = x.copy()
    n_rows = x.shape[0]
    start_frames = np.random.randint(0, num_consec, size=n_rows)
    for i in range(n_rows):
        start = start_frames[i]
        length = np.random.randint(1, num_consec - start + 1)
        col_lo = start * features_per_frame
        col_hi = (start + length) * features_per_frame
        x[i, col_lo:col_hi] = rep_value
    return x
 
 
def corrupt_consecutive(
    x: np.ndarray,
    num_consec: int = NUM_CONSEC,
    features_per_frame: int = FEATURES_PER_FRAME,
    noise_range: tuple[float, float] = (1.0, 2.0),
) -> np.ndarray:
    """Augmentation: replace one random contiguous run of frames with 
    plausible noise.
    Simulates: a tracker confidently reporting a wrong position, rather 
    than reporting nothing.
    
    Same frame-selection logic as drop_consecutive(), but instead of
    zeroing (which would look like "undetected"), fills with random values
    drawn from the same [1, 2] range real detections live in after
    scale_data()simulating 
 
    Args:
        x: Feature array, shape (N, num_consec * features_per_frame),
            already scaled to [1, 2] with 0 marking "undetected".
        num_consec: Frames per window.
        features_per_frame: Feature columns per frame.
        noise_range: (min, max) to draw replacement noise from.
 
    Returns:
        Augmented copy of x, one random contiguous frame-run per row
        overwritten with uniform random noise in noise_range.
    """
    x = x.copy()
    n_rows = x.shape[0]
    start_frames = np.random.randint(0, num_consec, size=n_rows)
    for i in range(n_rows):
        start = start_frames[i]
        length = np.random.randint(1, num_consec - start + 1)
        col_lo = start * features_per_frame
        col_hi = (start + length) * features_per_frame
        n_cols = col_hi - col_lo
        x[i, col_lo:col_hi] = np.random.uniform(*noise_range, size=n_cols)
    return x
 
 
def drop_data(x: np.ndarray, rep_value: float = 0.0, keep_prob: float = 0.95) -> np.ndarray:
    """Augmentation: zero out a random scattered subset of individual values.
 
    Unlike drop_consecutive(), entries are chosen independently across the
    whole array (any feature, any frame), not as a contiguous block --
    simulates isolated single-value noise rather than a sustained dropout.
 
    Args:
        x: Feature array.
        rep_value: Value used to overwrite dropped entries (default 0).
        keep_prob: Fraction of values left untouched (default 0.95).
 
    Returns:
        Augmented copy of x with (1 - keep_prob) of its entries, chosen
        independently at random, replaced by rep_value.
    """
    x = x.copy()
    mask = np.random.random(x.shape) >= keep_prob
    x[mask] = rep_value
    return x
 
 
def corrupt_data(
    x: np.ndarray,
    keep_prob: float = 0.95,
    noise_range: tuple[float, float] = (1.0, 2.0),
    eps: float = 1e-6,
) -> np.ndarray:
    """Augmentation: replace a random scattered subset of values with plausible
    noise.
 
    Same scattered-selection logic as drop_data(), but fills with random
    values in noise_range instead of zeroing -- and specifically leaves
    already-"undetected" (near-zero) entries alone, since there's nothing
    real there to corrupt into a fake detection.
 
    Args:
        x: Feature array, already scaled to [1, 2] with 0 marking "undetected".
        keep_prob: Fraction of values left untouched (default 0.95).
        noise_range: (min, max) to draw replacement noise from.
        eps: Threshold below which a value is treated as "undetected" and
            excluded from corruption.
 
    Returns:
        Augmented copy of x with (1 - keep_prob) of its non-"undetected"
        entries replaced by random noise in noise_range.
    """
    x = x.copy()
    corrupt_mask = (np.random.random(x.shape) >= keep_prob) & (np.abs(x) >= eps)
    x[corrupt_mask] = np.random.uniform(*noise_range, size=int(corrupt_mask.sum()))
    return x
 
 
def jiggle_and_rotate(
    x: np.ndarray,
    max_angle_deg: float = 15.0,
    max_shear: float = 0.05,
    eps: float = 1e-6,
) -> np.ndarray:
    """Augmentation: Apply a small rotation and shear. 
    Simulates: a very slightly different camera angle 
    
    The same transform is applied to every (x, y) pair in a row (shuttle, both
    players' poses, court corners) so the perturbed scene stays spatially
    coherent, rather than applying independent random rotations per point
    (which would just be incoherent noise, not a plausible camera jiggle).
 
    Args:
        x: Feature array with alternating x/y coordinate columns, already
            scaled to [1, 2] with 0 marking "undetected".
        max_angle_deg: Maximum rotation magnitude, in degrees.
        max_shear: Maximum shear magnitude.
        eps: Threshold below which a value is treated as "undetected" and
            left untouched.
 
    Returns:
        Augmented copy of x.
    """
    x = x.copy()
    n_rows, n_cols = x.shape
    even_cols = np.arange(0, n_cols, 2)
    odd_cols = np.arange(1, n_cols, 2)
 
    for i in range(n_rows):
        xs, ys = x[i, even_cols], x[i, odd_cols]
        undetected = (np.abs(xs) < eps) & (np.abs(ys) < eps)
 
        angle = np.random.uniform(-max_angle_deg, max_angle_deg) * np.pi / 180
        shear = np.random.uniform(-max_shear, max_shear)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
 
        # Rotation, then a small shear along x, both centred on this row's
        # own mean position so the transform perturbs shape, not location.
        cx, cy = xs[~undetected].mean() if (~undetected).any() else 1.5, \
                 ys[~undetected].mean() if (~undetected).any() else 1.5
        xs_c, ys_c = xs - cx, ys - cy
        new_xs = xs_c * cos_a - ys_c * sin_a + shear * ys_c
        new_ys = xs_c * sin_a + ys_c * cos_a
 
        x[i, even_cols] = np.where(undetected, xs, new_xs + cx)
        x[i, odd_cols] = np.where(undetected, ys, new_ys + cy)
 
    return x
 
 
def identity(x: np.ndarray) -> np.ndarray:
    """No-op augmentation -- returns the input unchanged.
 
    Args:
        x: Feature array.
 
    Returns:
        x, unmodified.
    """
    return x
 
 
def drop_random_and_jiggle(x: np.ndarray) -> np.ndarray:
    """Composed augmentation: jiggle/rotate, then randomly drop scattered values.
 
    Args:
        x: Feature array.
 
    Returns:
        Augmented copy of x.
    """
    return drop_data(jiggle_and_rotate(x))
 
 
def corrupt_random_and_jiggle(x: np.ndarray) -> np.ndarray:
    """Composed augmentation: jiggle/rotate, then randomly corrupt scattered values.
 
    Args:
        x: Feature array.
 
    Returns:
        Augmented copy of x.
    """
    return corrupt_data(jiggle_and_rotate(x))
 
 
def drop_consecutive_and_jiggle(x: np.ndarray) -> np.ndarray:
    """Composed augmentation: jiggle/rotate, then drop one contiguous frame-run.
 
    Args:
        x: Feature array.
 
    Returns:
        Augmented copy of x.
    """
    return drop_consecutive(jiggle_and_rotate(x))
 
 
def corrupt_consecutive_and_jiggle(x: np.ndarray) -> np.ndarray:
    """Composed augmentation: jiggle/rotate, then corrupt one contiguous frame-run.
 
    Args:
        x: Feature array.
 
    Returns:
        Augmented copy of x.
    """
    return corrupt_consecutive(jiggle_and_rotate(x))
 
 
AUGMENTATION_CHOICES = [
    identity, # no augmentation
    drop_random_and_jiggle,
    corrupt_random_and_jiggle,
    drop_consecutive_and_jiggle,
    corrupt_consecutive_and_jiggle,
]
AUGMENTATION_PROBS = [0.3, 0.175, 0.175, 0.175, 0.175]
 
 
def augment(x: np.ndarray) -> np.ndarray:
    """Apply one randomly-chosen augmentation to a batch, plus a 50% chance reflection.
 
    Deliberately does NOT re-scale the result via scale_data() -- unlike
    the reference notebook's version of this function. x is already scaled
    to [1, 2] once, at dataset-build time (build_dataset() ->
    scale_data()); re-scaling a small batch again here, based on that
    batch's own min/max, would introduce unintended scale drift between
    batches rather than a deliberate augmentation. See this module's
    "augmentation pipeline" discussion for the reasoning.
 
    Args:
        x: Batch of feature vectors, already scaled to [1, 2].
 
    Returns:
        Augmented copy of x -- one transform from AUGMENTATION_CHOICES is
        picked per call (weighted by AUGMENTATION_PROBS), then a 50%
        chance horizontal reflection is applied on top.
    """
    transform = np.random.choice(AUGMENTATION_CHOICES, p=AUGMENTATION_PROBS)
    x = transform(x)
    if np.random.random() < 0.5:
        x = reflect(x)
    return x
 
 
def scale_data(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize x/y coordinate columns to [1, 2], preserving "undetected" zeros.
 
    Args:
        x: Feature array with alternating x/y coordinate columns
            (shuttle, pose, and court corners are all laid out this way).
        eps: Threshold below which a value is treated as "undetected".
 
    Returns:
        Scaled copy of x -- near-zero entries stay 0, every other value in
        each coordinate group is linearly mapped to [1, 2].
    """
    x = x.astype(np.float64).copy()
 
    def scale_by_col(x, cols):
        x_ = x[:, cols]
        idx = np.abs(x_) < eps
        if (~idx).sum() == 0:
            return x
        m, M = np.min(x_[~idx]), np.max(x_[~idx])
        if M - m < eps:
            return x
        x_[~idx] = (x_[~idx] - m) / (M - m) + 1
        x[:, cols] = x_
        return x
 
    even_cols = list(range(0, x.shape[1], 2))
    odd_cols = list(range(1, x.shape[1], 2))
    x = scale_by_col(x, even_cols)
    x = scale_by_col(x, odd_cols)
    return x


def fetch_data(x_train: np.ndarray, y_train: np.ndarray, batch_size: int = BATCH_SIZE):
    """Infinite generator yielding shuffled, augmented training batches.
 
    Args:
        x_train: Training features, already scaled to [1, 2].
        y_train: Training labels, same row order as x_train.
        batch_size: Number of examples per yielded batch.
 
    Yields:
        (x_batch, y_batch) tuples -- x_batch is augmented via augment(),
        y_batch is the corresponding (unmodified) label slice. Reshuffles
        the full dataset at the start of every complete pass.
    """
    n = x_train.shape[0]
    while True:
        order = np.random.permutation(n)
        x_shuffled, y_shuffled = x_train[order], y_train[order]
        for start in range(0, n - batch_size + 1, batch_size):
            end = start + batch_size
            yield augment(x_shuffled[start:end]), y_shuffled[start:end]


# Composed augmentation: jiggle/rotate, then randomly drop scattered values.
drop_random_and_jiggle = lambda x: drop_data(jiggle_and_rotate(x), 0, 0.95)
# Composed augmentation: jiggle/rotate, then randomly corrupt scattered values.
corrupt_random_and_jiggle = lambda x: corrupt_data(jiggle_and_rotate(x), 0.95)
# Composed augmentation: jiggle/rotate, then drop one contiguous block.
drop_consecutive_and_jiggle = lambda x: drop_consecutive(jiggle_and_rotate(x))
# Composed augmentation: jiggle/rotate, then corrupt one contiguous block.
corrupt_consecutive_and_jiggle = lambda x: corrupt_consecutive(jiggle_and_rotate(x))

 
def _build_macro_f1_metric():
    """Build a MacroF1Score Keras metric class, defined lazily since it needs tf.keras.metrics.Metric.
 
    Macro-averaged F1 across the 3 hit classes -- used as the Keras Tuner
    search objective instead of raw val_accuracy. This matters
    specifically because x_val/y_val is NOT class-rebalanced,
    so it's dominated by the no-hit class (~80%); a raw val_accuracy
    objective can't distinguish a genuinely useful model from one that's
    just good at exploiting that imbalance. Macro-F1 weights each class equally
    regardless of frequency, so it can't be gamed by class imbalance the
    way accuracy can.
 
    Returns:
        The MacroF1Score class (not an instance) -- instantiate it
        yourself, e.g. MacroF1Score() when compiling a model.
    """
    import tensorflow as tf
 
    class MacroF1Score(tf.keras.metrics.Metric):
        """Macro-averaged F1 score across `num_classes` sparse-labeled classes.
 
        Maintains a running confusion matrix across batches (via
        tf.math.confusion_matrix), and computes macro-F1 from it in
        result() -- precision/recall/F1 per class, then averaged
        unweighted across classes, so a rare class contributes equally to
        a common one rather than being drowned out.
        """
 
        def __init__(self, num_classes: int = 3, name: str = "macro_f1", **kwargs):
            """
            Args:
                num_classes: Number of classes (3: no-hit, bottom-hit, top-hit).
                name: Metric name, as shown in training logs / used to
                    reference this metric as a Keras Tuner objective
                    (e.g. "val_macro_f1").
            """
            super().__init__(name=name, **kwargs)
            self.num_classes = num_classes
            self.confusion = self.add_weight(
                name="confusion", shape=(num_classes, num_classes), initializer="zeros"
            )
 
        def update_state(self, y_true, y_pred, sample_weight=None):
            """Accumulate this batch's contribution to the running confusion matrix.
 
            Args:
                y_true: Sparse integer labels, any shape that flattens to (N,).
                y_pred: Per-class probabilities/logits, shape (N, num_classes).
                sample_weight: Unused (accepted for Keras Metric API compatibility).
 
            Returns:
                None. Updates self.confusion in place.
            """
            y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
            y_pred_labels = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
            batch_cm = tf.math.confusion_matrix(
                y_true, y_pred_labels, num_classes=self.num_classes, dtype=tf.float32
            )
            self.confusion.assign_add(batch_cm)
 
        def result(self):
            """Compute macro-F1 from the running confusion matrix.
 
            Returns:
                Scalar tensor: the unweighted mean of per-class F1 scores.
            """
            cm = self.confusion
            tp = tf.linalg.diag_part(cm)
            fp = tf.reduce_sum(cm, axis=0) - tp
            fn = tf.reduce_sum(cm, axis=1) - tp
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            return tf.reduce_mean(f1)
 
        def reset_state(self):
            """Zero out the running confusion matrix at the start of each epoch.
 
            Returns:
                None.
            """
            self.confusion.assign(tf.zeros((self.num_classes, self.num_classes)))
 
    return MacroF1Score
    

def build_model(hp) -> "tf.keras.Model":
    """Build and compile a HitNet-style bidirectional GRU classifier.
 
    Architecture matches what we confirmed against the MonoTrack paper
    earlier in this conversation -- reimplemented here as our own code
    rather than copied from the reference notebook (see the "should we
    copy the notebook wholesale" discussion): a stack of bidirectional GRU
    layers over NUM_CONSEC frames, predicting a 3-way softmax (no-hit /
    bottom-hit / top-hit) using only the label window the paper specifies
    (frames left_window:NUM_CONSEC).
 
    Args:
        hp: A keras_tuner HyperParameters object, used to choose:
            - number of GRU layers: 1, 2, or 4
            - units per GRU layer: 16, 64, or 128
            - L2 regularization strength
            - dropout rate
 
    Returns:
        A compiled tf.keras Model. Input shape is
        (NUM_CONSEC * FEATURES_PER_FRAME,); internally reshaped to
        (NUM_CONSEC, FEATURES_PER_FRAME) before the GRU stack.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers
 
    # clear_session() resets Keras's global state
    # gc.collect() nudges Python's own collector to actually
    # free what clear_session() made collectable, since TF/Keras objects
    # often involve reference cycles the collector doesn't clean up
    # immediately from reference counting alone.
    tf.keras.backend.clear_session()
    gc.collect()
 
    l2_strength = hp.Float("l2_strength", min_value=1e-5, max_value=1e-2, sampling="log")
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.5, step=0.1)
    n_gru_layers = hp.Choice("gru_layers", [1, 2, 4])
    gru_units = hp.Choice("gru_units", [16, 64, 128])
 
    inputs = tf.keras.Input(shape=(NUM_CONSEC * FEATURES_PER_FRAME,))
    x = layers.Reshape((NUM_CONSEC, FEATURES_PER_FRAME))(inputs)
 
    for i in range(n_gru_layers):
        return_sequences = i < n_gru_layers - 1
        x = layers.Bidirectional(
            layers.GRU(
                gru_units,
                return_sequences=return_sequences,
                kernel_regularizer=regularizers.l2(l2_strength),
            )
        )(x)
        x = layers.Dropout(dropout_rate)(x)
 
    outputs = layers.Dense(3, activation="softmax")(x)
 
    MacroF1Score = _build_macro_f1_metric()
 
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", MacroF1Score()],
    )
    return model
    
 
def search_and_train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    max_trials: int = 40, # 50 in original notebook
    epochs: int = 60, # epochs=600 in original notebook
    tuner_dir: Path = TRAINING_DIR / "tuner",
) -> "tf.keras.Model":
    """Run a Bayesian hyperparameter search over build_model(), then return the best model.
 
    Trains via fetch_data()'s augmented generator.
 
    Ranks trials by val_macro_f1, NOT val_accuracy. x_val/y_val is
    (correctly) not class-rebalanced, so it's dominated by the no-hit
    class -- raw val_accuracy can't distinguish a genuinely useful model
    from one that's just good at exploiting that imbalance.
 
    Args:
        x_train, y_train: Training data (from build_dataset() + rebalance_classes()).
        x_val, y_val: Validation data (from build_dataset(), NOT rebalanced;
            not augmented -- validation should reflect real data as-is).
        max_trials: Number of hyperparameter combinations to try.
        epochs: Max epochs per trial (EarlyStopping will typically stop sooner).
        tuner_dir: Where Keras Tuner stores trial results/checkpoints.
 
    Returns:
        The best-performing tf.keras Model found by the search, already
    Returns:
        Tuple (best_model, tuner):
          best_model: The best-performing tf.keras Model found by the
              search, already retrained on the full training set with the
              winning hyperparameters.
          tuner: The kt.BayesianOptimization tuner object itself -- kept
              accessible to the caller for post-hoc inspection (e.g.
              tuner.oracle.get_best_trials(...) to check whether the
              search was still improving near the end, or had plateaued).
    """
    import tensorflow as tf
    import keras_tuner as kt
 
    steps_per_epoch = x_train.shape[0] // BATCH_SIZE
 
    tuner = kt.BayesianOptimization(
        build_model,
        objective=kt.Objective("val_macro_f1", direction="max"),
        max_trials=max_trials,
        directory=str(tuner_dir),
        project_name="hitnet_search",
    )
 
    callbacks = [
        # Monitor val_macro_f1 (mode="max"), NOT val_loss -- must stay
        # consistent with the tuner's own objective above. Leaving these on
        # val_loss (as an earlier version of this file did) meant
        # restore_best_weights would roll each trial's model back to
        # whichever epoch had the best LOSS, not the best macro-F1 --
        # reintroducing the exact imbalance-vulnerable selection we
        # switched the tuner's objective away from, just one level deeper
        # (within-trial instead of cross-trial). Confirmed this was a real
        # problem, not theoretical: fixing only the tuner's objective and
        # leaving this mismatched produced a run where both hit classes
        # got WORSE, including one collapsing to zero recall entirely.
        tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=8, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1", mode="max", factor=0.5, patience=6),
    ]
 
    tuner.search(
        fetch_data(x_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=steps_per_epoch,
        validation_data=(x_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
 
    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"[INFO] Best hyperparameters found: {best_hp.values}")
 
    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(
        fetch_data(x_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=steps_per_epoch,
        validation_data=(x_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    return best_model, tuner
 
 
def evaluate_model(model: "tf.keras.Model", x_val: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    """Evaluate a trained model on validation data with per-class metrics.
 
    Deliberately leads with per-class precision/recall/f1 rather than a
    single overall accuracy number -- x_val/y_val is NOT class-rebalanced
 
    Args:
        model: A trained tf.keras Model (e.g. from build_model()) whose
            predict() returns per-window class probabilities, shape
            (N, 3).
        x_val: Validation features, shape (N, NUM_CONSEC * FEATURES_PER_FRAME).
        y_val: True validation labels, shape (N,), values in {0, 1, 2}.
 
    Returns:
        y_pred: The model's predicted class per window, shape (N,) --
        handed back so the caller can feed it into further analysis (e.g.
        building the confusion-matrix-driven error inspection described in
        this function's docstring, or a tolerance-based hit-alignment check
        against the original rally timeline).
 
    Prints:
        - A classification_report (precision/recall/f1 per class, plus
          macro/weighted averages).
        - A confusion matrix -- rows are true class, columns are predicted
          class. Off-diagonal entries between class 1 and 2 specifically
          suggest a bottom/top labeling or court-orientation issue upstream
          rather than a model problem; entries in column 0 (predicted
          no-hit) for true classes 1/2 are missed hits (false negatives);
          entries in rows 0, columns 1/2 are false-positive hits.
    """
    from sklearn.metrics import classification_report, confusion_matrix
 
    y_prob = model.predict(x_val)
    y_pred = np.argmax(y_prob, axis=1)
 
    print("[INFO] Classification report (per class 0=no_hit, 1=bottom_hit, 2=top_hit):")
    print(classification_report(y_val, y_pred, digits=3, zero_division=0))
 
    cm = confusion_matrix(y_val, y_pred, labels=[0, 1, 2])
    print("[INFO] Confusion matrix (rows=true, cols=predicted):")
    print("            pred_0   pred_1   pred_2")
    for true_label, row in zip([0, 1, 2], cm):
        print(f"  true_{true_label}:   {row[0]:6d}   {row[1]:6d}   {row[2]:6d}")
 
    n_true_hits = int((y_val != 0).sum())
    n_missed = int(((y_val != 0) & (y_pred == 0)).sum())
    if n_true_hits > 0:
        print(f"[INFO] Missed {n_missed}/{n_true_hits} true hits "
              f"({100 * n_missed / n_true_hits:.1f}%) -- these are the windows "
              f"where a real hit occurred but the model predicted no_hit.")
 
    return y_pred

    
if __name__ == "__main__":
    stems = discover_rally_stems()
    print(f"[INFO] Found {len(stems)} rally clips")

    # explicitly split training and validation data
    # edit VAL_MATCH_IDS to change validation matches
    train_stems, val_stems = split_stems_by_match(stems)
    train_matches = set(stem_to_match_id(s) for s in train_stems)
    val_matches = set(stem_to_match_id(s) for s in val_stems)
    print(f"[INFO] Train: {len(train_stems)} rallies "
          f"({len(train_matches)} matches: {train_matches})")
    print(f"[INFO] Val:   {len(val_stems)} rallies "
          f"({len(val_matches)} matches: {val_matches})")
    
    # build training and validation datasets 
    x_train, y_train = build_dataset(train_stems)
    x_val, y_val = build_dataset(val_stems)
    print(f"[INFO] Raw windows -- train: {x_train.shape}, val: {x_val.shape}")

    # Subsample the no-hit windows in the !!training set only!! down to roughly match 
    # the combined count of the two hit classes (otherwise we can get high accuracy 
    # trivially from the nonhit events)
    x_train, y_train = rebalance_classes(x_train, y_train)
    print(f"[INFO] After class rebalancing -- train: {x_train.shape}")
    print(f"[INFO] Train label distribution: "
          f"{dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Model training
    model, tuner = search_and_train(x_train, y_train, x_val, y_val)
    evaluate_model(model, x_val, y_val)
    model.save(MODEL_OUT_PATH)
    print(f"[INFO] Saved model -> {MODEL_OUT_PATH}")
    
    # best_trials = tuner.oracle.get_best_trials(num_trials=max_trials)
    # scores = [t.score for t in best_trials]
    # print(scores)  # or plot them in trial order
