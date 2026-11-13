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

Training flow (see search_and_train()):
    1. Fast hyperparameter search -- short "epochs" (SEARCH_STEPS_PER_EPOCH
       batches), validated on a random SEARCH_VAL_SAMPLES-window subsample
       of the validation set. Ranks configs cheaply.
    2. Full training of the single winning config -- full-length epochs,
       full validation set, longer patience.
    3. Evaluation on BOTH validation (used for selection -> optimistic) and
       a separate TEST set of matches that nothing above ever touches
       (-> the honest number to report).

NOTE: changing anything in build_model()'s search space makes an existing
tuner/ directory incompatible -- delete it (rm -rf tuner/) before rerunning.

References:
    1. https://github.com/jhwang7628/monotrack
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import tensorflow as tf

# Only allocate GPU memory as needed, instead of TF's default of reserving
# nearly all free VRAM at startup -- other processes (data prep, webapps)
# share this GPU. Must run before anything touches the GPU.
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

import math
import time
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
INPUT_DATA_DIR = TRAINING_DIR / "data_pad15"
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

BATCH_SIZE = 256          # was 64 -- small GRU on a 48 GB GPU; 64 left it idle
VAL_BATCH_SIZE = 4096     # prediction-only; any size gives identical results, bigger = faster

COURT_W = 6.1   # doubles width (m) -- see corner-order discussion
COURT_L = 13.4  # doubles length (m)

# Settled by two searches (Oct 2026). Set to None to run the Bayesian search again.
FIXED_HP = {"learning_rate": 5.6e-4, "l2_strength": 4.7e-7, "dropout_rate": 0.0,
            "gru_layers": 4, "gru_units": 256}

# ---- Stage 1: hyperparameter search (fast) ----
MAX_TRIALS = 30                  # was 50; Bayesian search usually plateaus well before
SEARCH_EPOCHS = 40               # ceiling -- EarlyStopping ends most trials sooner
SEARCH_STEPS_PER_EPOCH = 1500    # ~384k windows per "epoch" at batch 256
SEARCH_VAL_SAMPLES = 100_000     # random validation subsample used to rank trials
SEARCH_ES_PATIENCE = 6
SEARCH_LR_PATIENCE = 2

# ---- Stage 2: final training of the winning config (thorough) ----
FINAL_EPOCHS = 100               # ceiling; each epoch = one full pass over train
FINAL_ES_PATIENCE = 10
FINAL_LR_PATIENCE = 4
MIN_LR = 1e-5


# ====== TRAIN / VALIDATION / TEST SPLIT ======================================
# Split by match -> every split contains only unseen matches.
# matches in ./data: 1 to 96 -- excluding 9, 10, 12, 27 (92 matches)
#
#   TRAIN -- fits the weights
#   VAL   -- hyperparameter selection, EarlyStopping, ReduceLROnPlateau and
#            (per the README) temperature calibration. Because models are
#            *selected* on it, its scores are optimistic.
#   TEST  -- never used for any decision; evaluated once at the very end.
#            This is the number to report / compare between model versions.
#
# Tagged-app matches (format_tagged.py, ids like "tag-YEO-AN") can be added to
# TEST_MATCH_IDS -- they're the footage the model is actually deployed on, so
# they make the most honest test. Anything not listed in VAL or TEST trains.

TAGGED_PREFIX = "tag-"   # format_tagged.py prefixes every tagged id with this

VAL_MATCH_IDS = {
    "2", "7", "11", "18", "21", "29", "30", "31", "36", "40",
    "43", "49", "53", "57", "60", "65", "67", "71", "76", "84",
    "89", "93", "96", "tag-1"
}  # 23 matches

TEST_MATCH_IDS = {
    "5", "16", "24", "38", "47", "58", "69", "80", "91", "tag-3",
}  # 9 matches

assert not (VAL_MATCH_IDS & TEST_MATCH_IDS), "a match can't be in both VAL and TEST"


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
        and to decide train/val/test membership.
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

    Window layout is FRAME-MAJOR: columns [f*78 : (f+1)*78] hold frame f's
    78 features. The vectorised augmentations below rely on this.

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
    corners = flatten_court_corners(court_pts)  # ADGX

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

def build_dataset(
    stems: list[str],
    label: str = "",
    rebalance: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a full (scaled) feature/label dataset from a list of rally stems.

    Args:
        stems: Rally stems to include.
        label: Name shown in progress lines ("train" / "val" / "test").
        rebalance: If True, subsample each rally's no-hit windows down to
            that rally's hit-window count (training set only). Done per
            rally so the full set of no-hit windows is never held in memory.
        seed: RNG seed for the rebalancing subsample.

    Returns:
        (x, y): x float32, shape (N, NUM_CONSEC * FEATURES_PER_FRAME), scaled
        to [1, 2] (undetected entries stay 0). y int8, shape (N,), values in
        {0, 1, 2}. Rallies that fail to load are skipped, not fatal.
    """
    rng = np.random.default_rng(seed)
    x_all, y_all = [], []
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        result = build_rally_windows(stem)
        if result is not None:
            x_t, y_t = result
            x_t = scale_data(x_t).astype(np.float32)   # scale BEFORE subsampling
            if rebalance:
                hit_idx = np.flatnonzero(y_t != 0)
                no_idx = np.flatnonzero(y_t == 0)
                keep_no = rng.choice(no_idx, size=min(len(no_idx), len(hit_idx)), replace=False)
                keep = np.sort(np.concatenate([hit_idx, keep_no]))
                x_t, y_t = x_t[keep], y_t[keep]
            x_all.append(x_t)
            y_all.append(y_t.astype(np.int8))
        if i % 250 == 0 or i == len(stems):
            elapsed = time.time() - t0
            print(f"[INFO] build_dataset {label}: {i}/{len(stems)} rallies "
                  f"({elapsed:.0f}s, ~{elapsed / i * (len(stems) - i):.0f}s left)", flush=True)

    if not x_all:
        raise RuntimeError(f"No usable rallies found among {len(stems)} stems given.")
    return np.vstack(x_all), np.hstack(y_all)


def rebalance_classes(x: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Subsample no-hit examples so hit/non-hit classes is roughly equally represented.

    With build_dataset(rebalance=True) this is effectively just a shuffle --
    every rally already has at most as many no-hit as hit windows.

    Args:
        x: Feature array, shape (N, ...).
        y: Label array, shape (N,), values in {0, 1, 2}.
        seed: RNG seed for reproducibility.

    Returns:
        (x_balanced, y_balanced), shuffled.
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


# ===== TRAIN/VAL/TEST SPLIT ====================================================

def split_stems_by_match(
    stems: list[str],
    val_match_ids: set[str] = VAL_MATCH_IDS,
    test_match_ids: set[str] = TEST_MATCH_IDS,
) -> tuple[list[str], list[str], list[str]]:
    """Split rally stems into train/validation/test sets by match id.

    No individual rally is ever split across sets, and no two rallies from
    the same match ever land in different sets.

    Args:
        stems: All discovered rally stems.
        val_match_ids: Matches held out for validation (model selection).
        test_match_ids: Matches held out for final, untouched evaluation.

    Returns:
        (train_stems, val_stems, test_stems).
    """
    train, val, test = [], [], []
    for stem in stems:
        match_id = stem_to_match_id(stem)
        if match_id in test_match_ids:
            test.append(stem)
        elif match_id in val_match_ids:
            val.append(stem)
        else:
            train.append(stem)
    return train, val, test


# copied functions from train-hitnet.ipynb from hitnet repo

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
# All augmentations operate on a whole batch with array operations (no
# per-row Python loops) -- the per-row loops were the main reason the GPU
# sat idle waiting for batches.

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
    block = x[:, 0::2]
    undetected = np.abs(block) < eps
    x[:, 0::2] = np.where(undetected, block, -block)
    return x


def _consecutive_frame_mask(
    n_rows: int,
    num_consec: int = NUM_CONSEC,
    features_per_frame: int = FEATURES_PER_FRAME,
) -> np.ndarray:
    """Boolean mask selecting one random contiguous run of frames per row.

    Same sampling as the original per-row loop: start frame uniform in
    [0, num_consec), run length uniform in [1, num_consec - start], drawn
    independently per row. Relies on the frame-major window layout built
    by build_rally_windows().

    Returns:
        Bool array, shape (n_rows, num_consec * features_per_frame).
    """
    start = np.random.randint(0, num_consec, size=n_rows)
    length = np.random.randint(1, num_consec - start + 1)   # per-row upper bound
    frames = np.arange(num_consec)
    frame_mask = (frames >= start[:, None]) & (frames < (start + length)[:, None])
    return np.repeat(frame_mask, features_per_frame, axis=1)


def drop_consecutive(
    x: np.ndarray,
    rep_value: float = 0.0,
    num_consec: int = NUM_CONSEC,
    features_per_frame: int = FEATURES_PER_FRAME,
) -> np.ndarray:
    """Augmentation: Erase several consecutive frames.
    Simulates: occlusion, motion blur, temporary tracking failure.

    Zeroes out one random contiguous run of frames per row (start frame and
    length chosen independently per row).

    Args:
        x: Feature array, shape (N, num_consec * features_per_frame).
        rep_value: Value used to overwrite the dropped frames (default 0,
            matching the "undetected" sentinel elsewhere in this pipeline).
        num_consec: Frames per window.
        features_per_frame: Feature columns per frame.

    Returns:
        Augmented copy of x.
    """
    x = x.copy()
    x[_consecutive_frame_mask(x.shape[0], num_consec, features_per_frame)] = rep_value
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

    Same frame selection as drop_consecutive(), but fills with random values
    from the same [1, 2] range real detections live in after scale_data().

    Args:
        x: Feature array, shape (N, num_consec * features_per_frame),
            already scaled to [1, 2] with 0 marking "undetected".
        num_consec: Frames per window.
        features_per_frame: Feature columns per frame.
        noise_range: (min, max) to draw replacement noise from.

    Returns:
        Augmented copy of x.
    """
    x = x.copy()
    mask = _consecutive_frame_mask(x.shape[0], num_consec, features_per_frame)
    x[mask] = np.random.uniform(*noise_range, size=int(mask.sum()))
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
        Augmented copy of x.
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

    Same scattered selection as drop_data(), but fills with random values in
    noise_range instead of zeroing -- and leaves already-"undetected"
    (near-zero) entries alone, since there's nothing real there to corrupt.

    Args:
        x: Feature array, already scaled to [1, 2] with 0 marking "undetected".
        keep_prob: Fraction of values left untouched (default 0.95).
        noise_range: (min, max) to draw replacement noise from.
        eps: Threshold below which a value is treated as "undetected".

    Returns:
        Augmented copy of x.
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
    Simulates: a very slightly different camera angle.

    One random rotation + shear per row, applied to every (x, y) pair in that
    row (shuttle, both players' poses, court corners) so the perturbed scene
    stays spatially coherent. Centred on the row's mean detected position so
    the transform perturbs shape, not location. Undetected points (both
    coordinates ~0) are left untouched.

    Args:
        x: Feature array with alternating x/y coordinate columns, already
            scaled to [1, 2] with 0 marking "undetected".
        max_angle_deg: Maximum rotation magnitude, in degrees.
        max_shear: Maximum shear magnitude.
        eps: Threshold below which a value is treated as "undetected".

    Returns:
        Augmented copy of x.
    """
    x = x.copy()
    n = x.shape[0]
    xs, ys = x[:, 0::2].copy(), x[:, 1::2].copy()
    undetected = (np.abs(xs) < eps) & (np.abs(ys) < eps)
    det = ~undetected
    n_det = det.sum(axis=1, keepdims=True)
    safe = np.maximum(n_det, 1)
    cx = np.where(n_det > 0, (xs * det).sum(axis=1, keepdims=True) / safe, 1.5)
    cy = np.where(n_det > 0, (ys * det).sum(axis=1, keepdims=True) / safe, 1.5)

    angle = np.random.uniform(-max_angle_deg, max_angle_deg, (n, 1)) * np.pi / 180
    shear = np.random.uniform(-max_shear, max_shear, (n, 1))
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    xs_c, ys_c = xs - cx, ys - cy
    new_xs = xs_c * cos_a - ys_c * sin_a + shear * ys_c + cx
    new_ys = xs_c * sin_a + ys_c * cos_a + cy
    x[:, 0::2] = np.where(undetected, xs, new_xs)
    x[:, 1::2] = np.where(undetected, ys, new_ys)
    return x


def identity(x: np.ndarray) -> np.ndarray:
    """No-op augmentation -- returns the input unchanged."""
    return x


def drop_random_and_jiggle(x: np.ndarray) -> np.ndarray:
    """Composed augmentation: jiggle/rotate, then randomly drop scattered values."""
    return drop_data(jiggle_and_rotate(x), 0.0, 0.95)


def corrupt_random_and_jiggle(x: np.ndarray) -> np.ndarray:
    """Composed augmentation: jiggle/rotate, then randomly corrupt scattered values."""
    return corrupt_data(jiggle_and_rotate(x), 0.95)


def drop_consecutive_and_jiggle(x: np.ndarray) -> np.ndarray:
    """Composed augmentation: jiggle/rotate, then drop one contiguous frame-run."""
    return drop_consecutive(jiggle_and_rotate(x))


def corrupt_consecutive_and_jiggle(x: np.ndarray) -> np.ndarray:
    """Composed augmentation: jiggle/rotate, then corrupt one contiguous frame-run."""
    return corrupt_consecutive(jiggle_and_rotate(x))


AUGMENTATION_CHOICES = [
    identity,  # no augmentation
    drop_random_and_jiggle,
    corrupt_random_and_jiggle,
    drop_consecutive_and_jiggle,
    corrupt_consecutive_and_jiggle,
]
AUGMENTATION_PROBS = [0.3, 0.175, 0.175, 0.175, 0.175]


def augment(x: np.ndarray) -> np.ndarray:
    """Apply one randomly-chosen augmentation to a batch, plus a 50% chance reflection.

    Deliberately does NOT re-scale the result via scale_data() -- x is
    already scaled to [1, 2] once, at dataset-build time; re-scaling each
    batch by its own min/max would introduce scale drift between batches.

    Args:
        x: Batch of feature vectors, already scaled to [1, 2].

    Returns:
        Augmented copy of x (float32).
    """
    transform = AUGMENTATION_CHOICES[np.random.choice(len(AUGMENTATION_CHOICES), p=AUGMENTATION_PROBS)]
    x = transform(x)
    if np.random.random() < 0.5:
        x = reflect(x)
    return x.astype(np.float32, copy=False)


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

    Shuffles an index array each pass and gathers rows per batch, rather than
    copying the whole (multi-GB) training array into shuffled order.

    Args:
        x_train: Training features, already scaled to [1, 2].
        y_train: Training labels, same row order as x_train.
        batch_size: Number of examples per yielded batch.

    Yields:
        (x_batch float32, y_batch int8).
    """
    n = x_train.shape[0]
    while True:
        order = np.random.permutation(n)
        for start in range(0, n - batch_size + 1, batch_size):
            idx = order[start:start + batch_size]
            yield augment(x_train[idx]), y_train[idx]


def make_train_dataset(x_train: np.ndarray, y_train: np.ndarray,
                       batch_size: int = BATCH_SIZE) -> "tf.data.Dataset":
    """Wrap fetch_data() in a tf.data pipeline that prefetches in the background.

    The CPU prepares (augments) the next batches while the GPU trains on the
    current one, instead of the GPU waiting on every batch.
    """
    return tf.data.Dataset.from_generator(
        lambda: fetch_data(x_train, y_train, batch_size=batch_size),
        output_signature=(
            tf.TensorSpec(shape=(None, NUM_CONSEC * FEATURES_PER_FRAME), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int8),
        ),
    ).prefetch(tf.data.AUTOTUNE)


def _build_macro_f1_metric():
    """Build a MacroF1Score Keras metric class, defined lazily since it needs tf.keras.metrics.Metric.

    Macro-averaged F1 across the 3 hit classes -- used as the Keras Tuner
    search objective instead of raw val_accuracy, because x_val/y_val is NOT
    class-rebalanced (dominated by no-hit) and accuracy can be gamed by that
    imbalance. Macro-F1 weights each class equally regardless of frequency.

    Returns:
        The MacroF1Score class (not an instance).
    """
    class MacroF1Score(tf.keras.metrics.Metric):
        """Macro-averaged F1 score across `num_classes` sparse-labeled classes."""

        def __init__(self, num_classes: int = 3, name: str = "macro_f1", **kwargs):
            super().__init__(name=name, **kwargs)
            self.num_classes = num_classes
            self.confusion = self.add_weight(
                name="confusion", shape=(num_classes, num_classes), initializer="zeros"
            )

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.reshape(tf.cast(y_true, tf.int32), [-1])
            y_pred_labels = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
            batch_cm = tf.math.confusion_matrix(
                y_true, y_pred_labels, num_classes=self.num_classes, dtype=tf.float32
            )
            self.confusion.assign_add(batch_cm)

        def result(self):
            cm = self.confusion
            tp = tf.linalg.diag_part(cm)
            fp = tf.reduce_sum(cm, axis=0) - tp
            fn = tf.reduce_sum(cm, axis=1) - tp
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * precision * recall / (precision + recall + 1e-7)
            return tf.reduce_mean(f1)

        def reset_state(self):
            self.confusion.assign(tf.zeros((self.num_classes, self.num_classes)))

    return MacroF1Score


def build_model(hp) -> "tf.keras.Model":
    """Build and compile a HitNet-style bidirectional GRU classifier.

    A stack of bidirectional GRU layers over NUM_CONSEC frames, predicting a
    3-way softmax (no-hit / bottom-hit / top-hit).

    Searched hyperparameters:
        - learning_rate: 1e-4 .. 3e-3 (log)   [new -- was fixed at 1e-3]
        - gru_layers:    1, 2, 3              [was 1, 2, 4]
        - gru_units:     32, 64, 128, 256     [was 16, 64, 128]
        - l2_strength:   1e-5 .. 1e-2 (log)
        - dropout_rate:  0.0 .. 0.5

    Returns:
        A compiled tf.keras Model. Input shape is
        (NUM_CONSEC * FEATURES_PER_FRAME,); internally reshaped to
        (NUM_CONSEC, FEATURES_PER_FRAME) before the GRU stack.
    """
    from tensorflow.keras import layers, regularizers

    # clear_session() resets Keras's global state; gc.collect() actually
    # frees what it made collectable (TF/Keras objects often form cycles).
    tf.keras.backend.clear_session()
    gc.collect()

    # hitnet_model_19oct.h5
    # Best hyperparameters found: 
    #   Hyperparam          Best        Search range 
    #   gru_units	        256	        32–256* (the maximum)
    #   gru_layers	        3	        1–3* (the maximum)
    #   dropout_rate	    0.0	        *0.0–0.5 (the minimum)
    #   l2_strength	        1.05e-5	    *1e-5–1e-2 (essentially the minimum)
    #   learning_rate	    5.1e-4	    1e-4–3e-3 (within range)

    # Since many land at the edge of a range, true optimum probably lies beyond it
    # -> Widen the search ranges past the edges it hit

    learning_rate = hp.Float("learning_rate", min_value=1e-4, max_value=3e-3, sampling="log")
    l2_strength = hp.Float("l2_strength", min_value=1e-7, max_value=1e-4, sampling="log")
    dropout_rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.3, step=0.1)
    n_gru_layers = hp.Choice("gru_layers", [2, 3, 4])
    gru_units = hp.Choice("gru_units", [128, 256, 384, 512])

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
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", MacroF1Score()],
    )
    return model


def _callbacks(es_patience: int, lr_patience: int) -> list:
    """EarlyStopping + ReduceLROnPlateau, both on val_macro_f1 (the tuner's objective).

    Both monitor val_macro_f1 (mode="max"), NOT val_loss -- must stay
    consistent with the tuner's objective; monitoring loss here previously
    rolled trials back to the best-LOSS epoch and made both hit classes worse.
    ES patience is ~2-3x LR patience so each LR reduction gets several epochs
    to pay off before the trial is stopped.
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1", mode="max", patience=es_patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1", mode="max", factor=0.5, patience=lr_patience, min_lr=MIN_LR),
    ]


def search_and_train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    max_trials: int = MAX_TRIALS,
    tuner_dir: Path = TRAINING_DIR / "tuner",
):
    """Two-stage training: fast Bayesian search, then full training of the winner.

    Stage 1 (search): each trial trains on SEARCH_STEPS_PER_EPOCH batches per
    "epoch" and is validated on a fixed random SEARCH_VAL_SAMPLES-window
    subsample of the validation set -- enough to rank configs reliably, at a
    fraction of the cost.

    Stage 2 (final): the winning hyperparameters are trained from scratch with
    full-length epochs, the FULL validation set, and longer patience.

    Args:
        x_train, y_train: Training data (rebalanced).
        x_val, y_val: Validation data (NOT rebalanced, NOT augmented).
        max_trials: Number of hyperparameter combinations to try.
        tuner_dir: Where Keras Tuner stores trial results/checkpoints.

    Returns:
        (best_model, tuner).
    """
    import keras_tuner as kt

    train_ds = make_train_dataset(x_train, y_train, batch_size=BATCH_SIZE)

    if FIXED_HP is not None:
        hp = kt.HyperParameters()
        for name, value in FIXED_HP.items():
            hp.Fixed(name, value)        # build_model's hp.Float/hp.Choice then return these values
        best_hp, tuner = hp, None
        print(f"[INFO] Using fixed hyperparameters (search skipped): {FIXED_HP}")        
    else:
        # ---- Stage 1: fast search ------------------------------------------
        rng = np.random.default_rng(0)
        sub = rng.choice(len(x_val), size=min(SEARCH_VAL_SAMPLES, len(x_val)), replace=False)
        x_val_s, y_val_s = x_val[sub], y_val[sub]
        print(f"[INFO] Search: {max_trials} trials, {SEARCH_STEPS_PER_EPOCH} steps/epoch "
            f"(batch {BATCH_SIZE}), validating on {len(sub)} of {len(x_val)} val windows")

        tuner = kt.BayesianOptimization(
            build_model,
            objective=kt.Objective("val_macro_f1", direction="max"),
            max_trials=max_trials,
            directory=str(tuner_dir),
            project_name="hitnet_search",
        )
        tuner.search(
            train_ds,
            steps_per_epoch=SEARCH_STEPS_PER_EPOCH,
            validation_data=(x_val_s, y_val_s),
            validation_batch_size=VAL_BATCH_SIZE,
            epochs=SEARCH_EPOCHS,
            callbacks=_callbacks(SEARCH_ES_PATIENCE, SEARCH_LR_PATIENCE),
            verbose=2,
        )

        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"[INFO] Best hyperparameters found: {best_hp.values}")

    # ---- Stage 2: full training of the winner --------------------------
    steps_per_epoch = x_train.shape[0] // BATCH_SIZE
    print(f"[INFO] Final training: up to {FINAL_EPOCHS} epochs x {steps_per_epoch} steps, "
          f"validating on all {len(x_val)} val windows")
    best_model = build_model(best_hp)
    best_model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=(x_val, y_val),
        validation_batch_size=VAL_BATCH_SIZE,
        epochs=FINAL_EPOCHS,
        callbacks=_callbacks(FINAL_ES_PATIENCE, FINAL_LR_PATIENCE),
        verbose=2,
    )
    return best_model, tuner


def evaluate_model(
    model: "tf.keras.Model",
    x: np.ndarray,
    y: np.ndarray,
    name: str = "validation",
) -> np.ndarray:
    """Evaluate a trained model with per-class metrics.

    Leads with per-class precision/recall/f1 rather than overall accuracy --
    the data is NOT class-rebalanced, so accuracy is dominated by no-hit.

    Args:
        model: A trained tf.keras Model whose predict() returns per-window
            class probabilities, shape (N, 3).
        x: Features, shape (N, NUM_CONSEC * FEATURES_PER_FRAME).
        y: True labels, shape (N,), values in {0, 1, 2}.
        name: Split name shown in the printout ("validation" / "TEST").

    Returns:
        y_pred: Predicted class per window, shape (N,).

    Prints:
        - classification_report (per class, plus macro/weighted averages).
        - Confusion matrix (rows=true, cols=predicted). Off-diagonal 1<->2
          entries suggest a bottom/top labeling or court-orientation issue
          upstream; column 0 for true 1/2 = missed hits; row 0, cols 1/2 =
          false-positive hits.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    y_prob = model.predict(x, batch_size=VAL_BATCH_SIZE, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    print(f"\n[INFO] ===== {name} set: {len(y)} windows =====")
    print("[INFO] Classification report (per class 0=no_hit, 1=bottom_hit, 2=top_hit):")
    print(classification_report(y, y_pred, digits=3, zero_division=0))

    cm = confusion_matrix(y, y_pred, labels=[0, 1, 2])
    print("[INFO] Confusion matrix (rows=true, cols=predicted):")
    print("            pred_0   pred_1   pred_2")
    for true_label, row in zip([0, 1, 2], cm):
        print(f"  true_{true_label}:   {row[0]:6d}   {row[1]:6d}   {row[2]:6d}")

    n_true_hits = int((y != 0).sum())
    n_missed = int(((y != 0) & (y_pred == 0)).sum())
    if n_true_hits > 0:
        print(f"[INFO] Missed {n_missed}/{n_true_hits} true hits "
              f"({100 * n_missed / n_true_hits:.1f}%) -- windows where a real hit "
              f"occurred but the model predicted no_hit.")

    return y_pred


if __name__ == "__main__":
    stems = discover_rally_stems()
    print(f"[INFO] Found {len(stems)} rally clips")

    # Three-way split by match -- edit VAL_MATCH_IDS / TEST_MATCH_IDS above.
    train_stems, val_stems, test_stems = split_stems_by_match(stems)
    for split_name, split_stems in (("Train", train_stems), ("Val", val_stems), ("Test", test_stems)):
        matches = sorted({stem_to_match_id(s) for s in split_stems},
                         key=lambda m: (not m.isdigit(), int(m) if m.isdigit() else 0, m))
        print(f"[INFO] {split_name + ':':6} {len(split_stems)} rallies "
              f"({len(matches)} matches: {matches})")

    missing_test = TEST_MATCH_IDS - {stem_to_match_id(s) for s in test_stems}
    if missing_test:
        print(f"[WARN] TEST_MATCH_IDS with no rally data found: {sorted(missing_test)}")

    # Build datasets. Only TRAIN is rebalanced (no-hit subsampled per rally);
    # VAL and TEST keep their natural class distribution.
    x_train, y_train = build_dataset(train_stems, "train", rebalance=True)
    x_val, y_val = build_dataset(val_stems, "val")
    x_test, y_test = build_dataset(test_stems, "test")
    print(f"[INFO] Windows -- train: {x_train.shape}, val: {x_val.shape}, test: {x_test.shape}")

    # Already balanced per rally above; this just shuffles.
    x_train, y_train = rebalance_classes(x_train, y_train)
    print(f"[INFO] Train label distribution: "
          f"{dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Model training
    model, tuner = search_and_train(x_train, y_train, x_val, y_val)

    # Validation scores are optimistic (val was used for model selection);
    # TEST is the honest number to report and compare between versions.
    evaluate_model(model, x_val, y_val, name="validation (used for selection)")
    evaluate_model(model, x_test, y_test, name="TEST (held out)")

    model.save(MODEL_OUT_PATH)
    print(f"[INFO] Saved model -> {MODEL_OUT_PATH}")

    if tuner is not None:
        print([t.score for t in tuner.oracle.get_best_trials(num_trials=MAX_TRIALS)])