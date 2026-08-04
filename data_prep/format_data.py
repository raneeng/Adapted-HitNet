"""
format_data.py
---------------
Given a ShuttleSet match --id, this script:
  1. Looks up the match's YouTube URL in {SHUTTLESET_DIR}/match.csv
  2. Downloads the match video
  3. Finds every per-set ShuttleSet CSV: {SHUTTLESET_DIR}/{id}_set{n}.csv
  4. For each rally in each set (grouped by the CSV's `rally` column):
       - Slices the match video down to that rally's frame range
       - Runs shuttle tracking (TrackNet+InpaintNet) on the rally clip
       - Runs pose estimation (YOLOv8-Pose) on the rally clip
       - Writes the four HitNet training inputs this script is responsible
         for (court/*.out is created manually, per your workflow):
           ball_trajectory/{id}_{set}_{rally}_ball_predicted.csv
           poses/{id}_{set}_{rally}_player_bottom.csv
           poses/{id}_{set}_{rally}_player_top.csv
           shot/{id}_{set}_{rally}_hit.csv

Usage:
    python format_data.py --id 29

Directory layout expected/produced (see the constants below to relocate
any of these):

    SHUTTLESET_DIR/
        match.csv
        29_set1.csv
        29_set2.csv
        ...

    VIDEO_DIR/
        29.mp4                          <- downloaded match video

    INPUT_DATA_DIR/
        ball_trajectory/29_set1_1_ball_predicted.csv
        poses/29_set1_1_player_bottom.csv
        poses/29_set1_1_player_top.csv
        shot/29_set1_1_hit.csv
        court/29.out                      <- YOU create this manually, once per match
        ...

ASSUMPTIONS:
  - match.csv (taken from ShuttleSet) has exactly one row per match id; that row's 
    `set` column gives the number of sets in the match, which main() uses to sanity
    -check that discover_set_csvs() actually found all of them (warns, doesn't
    abort, if the counts disagree).
  - A rally's frame range is [min(frame_num), max(frame_num) + END_PAD_FRAMES]
    across that rally's stroke rows
  - pose_features.py's "near"/"far" convention is mapped to HitNet's
    "bottom"/"top" convention as near->bottom, far->top.
  - build_pose_features() only returns rows for frames where at least one
    player was detected -- frames with zero detections are filled with 0.0
    in the dense per-frame pose CSVs this script writes, since HitNet needs
    exactly one row per frame with no gaps.
  - .out files contain the four doubles (OUTER) court corners
    A -------- D
    |          |
    |          |
    G -------- X
    in the order:
    
    1. Top-left (A)
    2. Top-right (D)
    3. Bottom-left (G)
    4. Bottom-right (X)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from court_side import resolve_bottom_players_for_set, player_to_int_label
from pose_features import build_pose_features, USED_KEYPOINTS
from shuttle_features import track_shuttle, write_prediction_csv


# ── Configuration -- adjust to your actual layout ───────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PREP_DIR = BASE_DIR / "data_prep"
SHUTTLESET_DIR = DATA_PREP_DIR / "shuttleset"
WEIGHTS_DIR = DATA_PREP_DIR / "weights"
YOLO_WEIGHTS_PATH = WEIGHTS_DIR / "yolov8s-pose.pt"
# track_shuttle() (in shuttle_features.py) expects a DIRECTORY containing
# both TrackNet_best.pt and InpaintNet_best.pt -- it builds both paths
# internally via weights_dir / "TrackNet_best.pt" and
# weights_dir / "InpaintNet_best.pt". So this must be WEIGHTS_DIR itself,
# not a path to one specific weight file.
TRACKNET_WEIGHTS_DIR = WEIGHTS_DIR

TRAINING_DIR = BASE_DIR / "training"
INPUT_DATA_DIR = TRAINING_DIR / "data"
VIDEO_DIR = INPUT_DATA_DIR / "videos"
COURT_DIR = INPUT_DATA_DIR / "court"

END_PAD_FRAMES = 30        # frames appended after the last recorded hit
NEAR_IS_BOTTOM = True      # near->bottom, far->top (flip if your footage differs
                            # -- see USED_KEYPOINTS ordering; index math below
                            # assumes standard COCO-17 keypoint order)

REQUIRED_SET_COLUMNS = {"rally", "frame_num", "player", "roundscore_A", "roundscore_B"}


# ── 1. Match lookup + video download ────────────────────────────────────────

def load_match_row(match_id: str) -> pd.Series:
    """Look up a match's single row in match.csv.

    match.csv has exactly one row per match id. The `set` column gives the
    number of sets played in that match, which also tells you how many
    {id}_set{n}.csv files to expect on disk -- used by main() to validate
    that discover_set_csvs() actually found all of them.

    Args:
        match_id: The --id value, matched against match.csv's `id` column
            (compared as strings, so "29" matches whether the CSV stores it
            as an int or a string).

    Returns:
        The matching row as a pandas Series (columns: id, video, tournament,
        round, year, month, day, set, duration, winner, loser, downcourt, url).

    Raises:
        FileNotFoundError: If match.csv doesn't exist at SHUTTLESET_DIR.
        ValueError: If zero or more than one row matches match_id (match.csv
            is expected to have exactly one row per id -- more than one is
            treated as a data problem, not resolved silently), or if `url`s
            is missing/blank.
    """
    match_csv_path = SHUTTLESET_DIR / "match.csv"
    if not match_csv_path.is_file():
        raise FileNotFoundError(f"match.csv not found at {match_csv_path}")

    match_df = pd.read_csv(match_csv_path, dtype={"id": str})
    rows = match_df[match_df["id"] == str(match_id)]

    if rows.empty:
        raise ValueError(f"No row found for id={match_id!r} in {match_csv_path}")
    if len(rows) > 1:
        raise ValueError(
            f"Expected exactly one row for id={match_id!r} in match.csv "
            f"(one row per match), found {len(rows)}."
        )

    row = rows.iloc[0]
    if pd.isna(row.get("url")) or not str(row.get("url")).strip():
        raise ValueError(f"id={match_id!r}'s row in match.csv has no url.")

    return row


def download_match_video(
    url: str,
    match_id: str,
    overwrite: bool = False,
    max_height: int | None = None,
) -> Path:
    """Download a match's YouTube video as mp4, via yt-dlp.

    Args:
        url:        YouTube video URL.
        match_id:   Used to name the output file ({match_id}.mp4).
        overwrite:  Re-download even if the file already exists.
        max_height: Optional cap on vertical resolution (e.g. 1080). None
            (default) means best available.

    Returns:
        Path to the downloaded video.

    Raises:
        ImportError: If yt-dlp isn't installed.
        yt_dlp.utils.DownloadError: If the download fails.
    """
    try:
        import yt_dlp
    except ImportError:
        raise ImportError(
            "yt-dlp is required to download match videos. Install with:\n"
            "    pip install -U yt-dlp"
        )

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEO_DIR / f"{match_id}.mp4"

    if out_path.exists() and not overwrite:
        print(f"[SKIP] {out_path} already exists (use overwrite=True to re-download).")
        return out_path

    height_filter = f"[height<={max_height}]" if max_height else ""

    has_ffmpeg = shutil.which("ffmpeg") is not None
    if has_ffmpeg:
        format_selector = (
            f"bestvideo[ext=mp4][vcodec^=avc1]{height_filter}/"
            f"bestvideo{height_filter}"
        )
    else:
        print("[WARN] ffmpeg not found on PATH -- can't merge separate "
              "video/audio streams, so this will fall back to YouTube's "
              "single-file progressive mp4, which is capped at a lower "
              "resolution (often 360p). Install ffmpeg to get full "
              "resolution downloads.")
        format_selector = f"best[ext=mp4][vcodec^=avc1]{height_filter}/best[ext=mp4]{height_filter}/best{height_filter}"

    ydl_opts = {
        "format": format_selector,
        "outtmpl": str(out_path.with_suffix("")) + ".%(ext)s",
    } 

    has_aria2 = shutil.which("aria2c") is not None
    if has_aria2:
        ydl_opts.update({
            "external_downloader": "aria2c",
            "external_downloader_args": [
                "-x", "16",
                "-s", "16",
                "-k", "1M",
            ],
        })
    else:
        print("[INFO] aria2c not found; using yt-dlp downloader.")
    
    print(f"[INFO] Downloading match {match_id} from {url}")
    print(f"[INFO] Format selector: {format_selector}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)

    print(f"[INFO] Downloaded -> {out_path}")
    return out_path


# ── 2. Discover per-set ShuttleSet CSVs ─────────────────────────────────────

def discover_set_csvs(match_id: str) -> list[tuple[int, Path]]:
    """Find every {id}_set{n}.csv for a match, in set order.

    Args:
        match_id: The --id value.

    Returns:
        List of (set_num, path) tuples, sorted by set_num ascending.
        Empty list if none are found (caller should treat this as an error
        condition, not silently do nothing).
    """
    found = []
    for path in sorted(SHUTTLESET_DIR.glob(f"{match_id}_set*.csv")):
        stem = path.stem  # e.g. "29_set1"
        suffix = stem.split("_set")[-1]
        try:
            set_num = int(suffix)
        except ValueError:
            print(f"[WARN] Skipping unexpected filename (can't parse set "
                  f"number): {path.name}")
            continue
        found.append((set_num, path))
    return sorted(found, key=lambda t: t[0])


# ── 3. Per-rally frame-range + video slicing ────────────────────────────────

def compute_rally_frame_range(
    rally_df: pd.DataFrame,
    end_pad_frames: int = END_PAD_FRAMES,
) -> tuple[int, int]:
    """Determine a rally's [start_frame, end_frame] from its stroke rows.

    Args:
        rally_df: Rows from one set's CSV belonging to a single rally
            (already filtered by the `rally` column).
        end_pad_frames: Frames appended after the last recorded hit, to
            better capture the shuttle's landing/outcome beyond the final
            stroke itself (see module docstring).

    Returns:
        (start_frame, end_frame) in the *match/set video's own* frame
        numbering -- not yet rebased to be clip-relative.

    Raises:
        ValueError: If rally_df is empty or has no valid frame_num values.
    """
    frames = rally_df["frame_num"].dropna()
    if frames.empty:
        raise ValueError("rally_df has no valid frame_num values")
    start_frame = int(frames.min())
    end_frame = int(frames.max()) + end_pad_frames
    return start_frame, end_frame


def slice_video_by_frames(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    out_path: Path,
) -> int:
    """Extract a frame-accurate sub-clip [start_frame, end_frame] from a video.

    Args:
        video_path:  Source (full match/set) video.
        start_frame: First frame to include (inclusive), in the source
            video's own frame numbering.
        end_frame:   Last frame to include (inclusive), clamped to the
            source video's actual frame count if it runs past the end.
        out_path:    Output clip path. Parent directory is created if needed.

    Returns:
        Number of frames actually written (may be less than
        end_frame - start_frame + 1 if the source video was shorter than
        expected).

    Raises:
        FileNotFoundError: If video_path can't be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    end_frame = min(end_frame, total - 1)
    if start_frame > end_frame:
        cap.release()
        raise ValueError(f"start_frame {start_frame} > end_frame {end_frame} "
                          f"(video only has {total} frames)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    written = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    
    while frame_idx <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
    
        writer.write(frame)
        written += 1
        frame_idx += 1

    cap.release()
    writer.release()
    return written


# ── 4. Hit CSV (single "hit" column, binary, one row per frame) ────────────

def build_hit_column(
    rally_df: pd.DataFrame,
    bottom_player: str,
    start_frame: int,
    total_frames: int,
) -> list[int]:
    """Build the dense per-frame binary hit column for one rally clip.

    Args:
        rally_df:     Rows from one set's CSV belonging to a single rally.
        start_frame:  The rally's start_frame in the *source* video's frame
            numbering (from compute_rally_frame_range()) -- used to rebase
            each stroke's frame_num to be 0-indexed relative to the clip.
        total_frames: Length of the sliced rally clip, in frames. The
            returned list always has exactly this many entries.

    Returns:
        List of int, length total_frames, 1 at every frame a stroke by
        either player was recorded, 0 elsewhere.
    """
    hit = [0] * total_frames
    for _, row in rally_df.dropna(subset=["frame_num"]).iterrows():
        raw_frame = row["frame_num"]
        rebased = int(raw_frame) - start_frame

        if 0 <= rebased < total_frames:
            hitter = row["player"] # A or B
            hit[rebased] = player_to_int_label(hitter, bottom_player)
            
        else:
            print(
                f"[WARN] Stroke frame_num={int(row['frame_num'])} "
                f"rebases to {rebased}, outside clip."
            )
    return hit


# ── 5. Pose CSVs (dense, 34 columns each, bottom/top) ───────────────────────

# Standard COCO-17 keypoint order -- must match pose_features.py's USED_KEYPOINTS
# indexing convention (0..16) for the reshape(17, 2) HitNet's training code
# performs on these values to be meaningful.
N_KEYPOINTS = 17


def pose_df_to_dense_bottom_top(
    pose_df: pd.DataFrame,
    total_frames: int,
    near_is_bottom: bool = NEAR_IS_BOTTOM,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert build_pose_features()'s sparse near/far output into HitNet's dense bottom/top format.

    Two conversions happen here:
      1. Sparse -> dense: build_pose_features() only emits a row for frames
         where at least one player was detected. HitNet needs exactly one
         row per frame, so missing frames are filled with 0.0.
      2. near/far -> bottom/top: this codebase's convention (from
         pose_features.py) is renamed to HitNet's convention.

    Args:
        pose_df:        Output of build_pose_features() for one rally clip.
        total_frames:   Length of the rally clip in frames. Both returned
            DataFrames always have exactly this many rows.
        near_is_bottom: If True, "near" columns become "bottom", "far"
            becomes "top". If False, the reverse.

    Returns:
        (bottom_df, top_df), each a DataFrame with exactly total_frames
        rows and 34 columns (x0, y0, x1, y1, ..., x16, y16), no confidence
        columns, no frame-number column -- matching HitNet's
        _player_bottom.csv / _player_top.csv format exactly.
    """
    bottom_prefix = "near" if near_is_bottom else "far"
    top_prefix = "far" if near_is_bottom else "near"

    def _extract(prefix: str) -> pd.DataFrame:
        cols = []
        for kp in range(N_KEYPOINTS):
            cols.append(f"x{kp}")
            cols.append(f"y{kp}")

        dense = pd.DataFrame(0.0, index=range(total_frames), columns=cols)

        if pose_df.empty:
            return dense

        indexed = pose_df.set_index("frame")
        for kp in range(N_KEYPOINTS):
            px_col = f"{prefix}_kp_{kp}_px"
            py_col = f"{prefix}_kp_{kp}_py"
            if px_col not in indexed.columns:
                continue
            for frame_idx, row in indexed.iterrows():
                if 0 <= frame_idx < total_frames:
                    x_val, y_val = row[px_col], row[py_col]
                    if pd.notna(x_val) and pd.notna(y_val):
                        dense.at[frame_idx, f"x{kp}"] = float(x_val)
                        dense.at[frame_idx, f"y{kp}"] = float(y_val)
        return dense

    return _extract(bottom_prefix), _extract(top_prefix)


# ── 6. Per-rally orchestration ──────────────────────────────────────────────

def _rally_output_paths(stem: str) -> dict[str, Path]:
    """Compute the four output file paths for a given rally stem.

    Args:
        stem: Output filename stem, e.g. "29_set1_3".

    Returns:
        Dict with keys "ball", "pose_bottom", "pose_top", "hit" mapping to
        their expected paths under INPUT_DATA_DIR.
    """
    return {
        "ball": INPUT_DATA_DIR / "ball_trajectory" / f"{stem}_ball_predicted.csv",
        "pose_bottom": INPUT_DATA_DIR / "poses" / f"{stem}_player_bottom.csv",
        "pose_top": INPUT_DATA_DIR / "poses" / f"{stem}_player_top.csv",
        "hit": INPUT_DATA_DIR / "shot" / f"{stem}_hit.csv",
    }


def _rally_already_done(stem: str) -> bool:
    """Check whether all four output files for a rally already exist.

    Used to skip already-completed rallies on a rerun (e.g. after a crash
    partway through a match)
    All four files must exist -- a partial set (e.g. 3 of 4, from a crash
    mid-rally) is treated as NOT done, since there's no safe way to resume
    a single rally partway through; it gets fully redone.

    Args:
        stem: Output filename stem, e.g. "29_set1_3".

    Returns:
        True if every one of the four expected output files exists.
    """
    return all(p.is_file() for p in _rally_output_paths(stem).values())


def process_rally(
    match_video_path: Path,
    rally_df: pd.DataFrame,
    stem: str,
    calibration: dict,
    bottom_player: str,
    force: bool = False,
    track_shuttle_fn=track_shuttle,
    build_pose_features_fn=build_pose_features,
) -> None:
    """Produce the four HitNet input files for a single rally.

    Args:
        match_video_path: The full downloaded match video.
        rally_df:          Rows from one set's CSV for a single rally.
        stem:              Output filename stem, e.g. "29_set1_3".
        calibration:       Already-loaded calibration dict (homography_matrix),
            shared across every rally in the whole match -- see
            load_match_calibration(), which loads this once per match
            rather than once per rally, since the
            camera doesn't move within a match.
        force:             If False (default) and all four output files for
            this stem already exist, the rally is skipped entirely
            Pass True to always reprocess regardless of existing outputs
        track_shuttle_fn:  Injectable for testing -- defaults to the real
            shuttle_features.track_shuttle.
        build_pose_features_fn: Injectable for testing -- defaults to the
            real pose_features.build_pose_features.

    Returns:
        None. Writes:
          INPUT_DATA_DIR/ball_trajectory/{stem}_ball_predicted.csv
          INPUT_DATA_DIR/poses/{stem}_player_bottom.csv
          INPUT_DATA_DIR/poses/{stem}_player_top.csv
          INPUT_DATA_DIR/shot/{stem}_hit.csv
        Or nothing, if skipped (see force above).
    """
    if not force and _rally_already_done(stem):
        print(f"[SKIP] {stem}: all four output files already exist "
              f"(use force=True / --force to reprocess anyway)")
        return

    start_frame, end_frame = compute_rally_frame_range(rally_df)

    clip_dir = INPUT_DATA_DIR / "rally_video"
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip_path = clip_dir / f"{stem}.mp4"

    n_written = slice_video_by_frames(match_video_path, start_frame, end_frame, clip_path)
    print(f"[INFO] {stem}: sliced frames [{start_frame}, {end_frame}] "
          f"-> {n_written} frames -> {clip_path}")

    # --- Shuttle tracking ---
    pred_dict, width, height, fps, frame_count = track_shuttle_fn(
        video_path=clip_path,
        weights_dir=TRACKNET_WEIGHTS_DIR,
        batch_size=16,
        use_inpaint=True,
    )
    output_paths = _rally_output_paths(stem)
    output_paths["ball"].parent.mkdir(parents=True, exist_ok=True)
    write_prediction_csv(pred_dict, fps, output_paths["ball"])
    print(f"[INFO] {stem}: wrote {output_paths['ball']}")

    # --- Pose estimation (calibration passed in, not re-loaded per rally) ---
    pose_df = build_pose_features_fn(
        video_path=clip_path,
        calibration=calibration,
        pose_model=YOLO_WEIGHTS_PATH,
        sample_rate=1,  # dense: every frame, HitNet needs frame-for-frame alignment
    )
    bottom_df, top_df = pose_df_to_dense_bottom_top(pose_df, total_frames=frame_count)

    output_paths["pose_bottom"].parent.mkdir(parents=True, exist_ok=True)
    bottom_df.to_csv(output_paths["pose_bottom"], index=False)
    top_df.to_csv(output_paths["pose_top"], index=False)
    print(f"[INFO] {stem}: wrote pose CSVs ({len(bottom_df)} rows each)")

    # --- Hit labels ---
    hit = build_hit_column(rally_df, bottom_player, start_frame, total_frames=frame_count)
    output_paths["hit"].parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"hit": hit}).to_csv(output_paths["hit"], index=False)
    print(f"[INFO] {stem}: wrote hit CSV ({sum(hit)} hits / {frame_count} frames)")


def _load_manual_calibration(calibration_out_path: Path) -> dict:
    """Load a manually-created court .out file into build_pose_features()'s expected format.

    Args:
        calibration_out_path: Path to a {stem}.out file -- 4 lines, "x;y"
            pixel coordinates (one court corner per line), in this exact
            order: ADGX
                A -------- D
                |          |
                |          |
                G -------- X            

    Returns:
        Dict with a "homography_matrix" key (3x3 list-of-lists), computed
        via cv2.findHomography mapping these four image points to their
        corresponding real-world court corners
        
    Raises:
        FileNotFoundError: If the .out file doesn't exist yet
        ValueError: If the file doesn't contain exactly 4 lines.
    """
    if not calibration_out_path.is_file():
        raise FileNotFoundError(
            f"Court file not found: {calibration_out_path}\n"
            f"This file is created manually, once per match (per your "
            f"workflow) -- create it before running this match."
        )
 
    with open(calibration_out_path) as f:
        pts = [[float(x) for x in line.strip().split(";")] for line in f if line.strip()]
 
    if len(pts) != 4:
        raise ValueError(f"{calibration_out_path} has {len(pts)} lines, expected 4")
 
    from pose_features import COURT_W, COURT_L
 
    image_pts = np.array(pts, dtype=np.float32)
    
    world_pts = np.array(
    [
        [0.0, 0.0],           # A (0,0)
        [COURT_W, 0.0],       # D (1,0)
        [0.0, COURT_L],       # G (0,1)
        [COURT_W, COURT_L],   # X (1,1)
    ],
    dtype=np.float32,
    )
    h, _ = cv2.findHomography(image_pts, world_pts)
    if h is None:
        raise RuntimeError(f"Homography failed for {calibration_out_path}")
 
    return {"homography_matrix": h.tolist()}

 
def write_court_out(points: list[tuple[float, float]], out_path: Path) -> None:
    """Write 4 clicked corner points to a .out file in the standard A;D;G;X format.
 
    Split out from label_court_corners() specifically so this part -- the
    actual file-writing logic -- can be tested without needing a real
    display/GUI to generate points through.
 
    Args:
        points: Exactly 4 (x, y) pixel coordinate tuples, in this exact
            order: ADGX
                A -------- D
                |          |
                |          |
                G -------- X
        out_path: Where to write the .out file.
 
    Returns:
        None. Writes one "x;y" line per point.
 
    Raises:
        ValueError: If points doesn't contain exactly 4 entries.
    """
    if len(points) != 4:
        raise ValueError(f"Expected exactly 4 points, got {len(points)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for px, py in points:
            f.write(f"{px};{py}\n")
 
 
def label_court_corners(match_id: str, overwrite: bool = False) -> Path:
    """Interactively mark the 4 court corners on a match's calibration reference image.
 
    Opens the saved calibration reference frame (see
    save_calibration_reference_frame()) in an OpenCV window and lets you
    click the 4 court corners in this exact order: ADGX
        A -------- D
        |          |
        |          |
        G -------- X
    (the doubles/OUTER court corners), matching the convention used
    throughout this pipeline. Writes the result directly to
    {match_id}.out, replacing the need to hand-edit the file.
 
    REQUIRES A GUI-CAPABLE DISPLAY. else use label_court.ipynb
 
    Controls:
        Left-click  : mark the next corner (up to 4)
        Right-click : undo the last marked corner
        Enter / 'q' : confirm and save once all 4 corners are marked
        Esc         : cancel without saving
 
    Args:
        match_id: The --id value.
        overwrite: If False (default) and {match_id}.out already exists,
            this is skipped entirely rather than re-labeling.
 
    Returns:
        Path to the written {match_id}.out file (or the existing path,
        unchanged, if skipped).
 
    Raises:
        FileNotFoundError: If the calibration reference image doesn't exist
            yet -- run format_data.py for this match first, which creates
            it automatically.
        RuntimeError: If the window is closed, or Esc is pressed, before
            all 4 corners are marked -- nothing is written in that case.
    """
    import cv2
 
    out_path = COURT_DIR / f"{match_id}.out"
    if out_path.is_file() and not overwrite:
        print(f"[SKIP] {out_path} already exists (pass overwrite=True to relabel).")
        return out_path
 
    ref_path = COURT_DIR / f"{match_id}_calibration_reference.png"
    if not ref_path.is_file():
        raise FileNotFoundError(
            f"Calibration reference image not found: {ref_path}\n"
            f"Run format_data.py for this match first -- it's created "
            f"automatically from the first hit of set 1."
        )
 
    img = cv2.imread(str(ref_path))
    if img is None:
        raise RuntimeError(f"Could not open image: {ref_path}")
 
    labels = ["A (top-left)", "D (top-right)", "G (bottom-left)", "X (bottom-right)"]
    points: list[tuple[float, float]] = []
    window_name = f"Mark court corners -- match {match_id}"
 
    def redraw():
        display = img.copy()
        for i, (px, py) in enumerate(points):
            cv2.circle(display, (int(px), int(py)), 6, (0, 0, 255), -1)
            cv2.putText(display, labels[i][0], (int(px) + 10, int(py) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        prompt = (f"Click {labels[len(points)]}" if len(points) < 4
                  else "All 4 marked -- Enter to save, right-click to undo")
        cv2.putText(display, prompt, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow(window_name, display)
 
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            redraw()
 
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()
 
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 27:  # Esc
            cv2.destroyAllWindows()
            raise RuntimeError("Cancelled -- no file written.")
        if key in (13, ord("q")) and len(points) == 4:
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            raise RuntimeError("Window closed before all 4 corners were marked -- no file written.")
 
    cv2.destroyAllWindows()
    write_court_out(points, out_path)
 
    print(f"[INFO] Saved {out_path}")
    for label, (px, py) in zip(labels, points):
        print(f"  {label}: ({px:.1f}, {py:.1f})")
    return out_path
 

def save_calibration_reference_frame(
    video_path: Path,
    set1_csv_path: Path,
    match_id: str,
) -> Path:
    """Extract and save the frame of the first recorded hit in set 1, for manual court calibration.

    Args:
        video_path:    The full downloaded match video (not a rally clip --
            frame_num values from set 1's CSV are relative to this full
            video, so no rebasing is needed here, unlike the per-rally
            clip-relative rebasing done elsewhere in this file).
        set1_csv_path: Path to {match_id}_set1.csv.
        match_id:      The --id value, used to name the output image.

    Returns:
        Path to the saved reference image
        (COURT_DIR/{match_id}_calibration_reference.png).

    Raises:
        ValueError: If set1_csv_path has no valid frame_num values.
        FileNotFoundError: If video_path can't be opened.
        RuntimeError: If the target frame couldn't be reached (video ended
            before it, e.g. frame_num doesn't actually belong to this video).
    """
    df = pd.read_csv(set1_csv_path)
    frames = df["frame_num"].dropna()
    if frames.empty:
        raise ValueError(f"{set1_csv_path} has no valid frame_num values")
    target_frame = int(frames.min())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_img = None
    frame_idx = 0
    while frame_idx <= target_frame:
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(
                f"Video only has {frame_idx} frame(s) -- couldn't reach "
                f"target frame {target_frame} (first hit in set 1). Check "
                f"that {set1_csv_path.name}'s frame_num values actually "
                f"correspond to this video."
            )
        frame_img = frame
        frame_idx += 1
    cap.release()

    COURT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COURT_DIR / f"{match_id}_calibration_reference.png"
    cv2.imwrite(str(out_path), frame_img)
    print(f"[INFO] Saved calibration reference frame (frame {target_frame}) -> {out_path}")
    return out_path


# ── 7. Per-set + top-level orchestration ────────────────────────────────────

def process_set(
    match_video_path: Path,
    set_csv_path: Path,
    match_id: str,
    downcourt: int,
    set_num: int,
    calibration: dict,
    force: bool = False,
) -> None:
    """Process every rally in one set's ShuttleSet CSV.

    Args:
        match_video_path: The full downloaded match video.
        set_csv_path:      Path to {id}_set{n}.csv.
        match_id:           The --id value, used in output filename stems.
        set_num:            The set number, used in output filename stems.
        calibration:        Already-loaded calibration dict, shared across
            the whole match (see load_match_calibration()) -- the camera
            doesn't move between sets, so one court file per match covers
            every rally in every set.
        force:              Passed straight through to process_rally() --
            if False (default), rallies whose four output files already
            exist are skipped rather than reprocessed. See process_rally().

    Returns:
        None. Calls process_rally() once per rally found in the CSV.

    Raises:
        ValueError: If the CSV is missing rally/frame_num columns.
    """
    df = pd.read_csv(set_csv_path)
    missing = REQUIRED_SET_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{set_csv_path} is missing required column(s): "
                          f"{sorted(missing)}")

    rally_ids = sorted(df["rally"].dropna().unique())

    roundscores = []
    
    for rally_num in rally_ids:
        rally_df = df[df["rally"] == rally_num]
    
        first_row = rally_df.iloc[0]
    
        roundscores.append((
            int(first_row["roundscore_A"]),
            int(first_row["roundscore_B"]),
        ))
    
    bottom_player_list = resolve_bottom_players_for_set( # List of "A"/"B" indicating player on bottom for that rally
        downcourt=downcourt,
        set_num=set_num,
        roundscores=roundscores,
    )
    print(f"[INFO] Set {set_num}: {len(rally_ids)} rallies")

    for row, rally_num in enumerate(rally_ids):
        rally_df = df[df["rally"] == rally_num]
        stem = f"{match_id}_set{set_num}_{int(rally_num)}"
        try:
            bottom_player = bottom_player_list[row]
            process_rally(match_video_path, rally_df, stem, calibration, bottom_player, force=force)
        except Exception as e:
            print(f"[ERROR] {stem}: failed with {type(e).__name__}: {e}")


def load_match_calibration(match_id: str) -> dict:
    """Load a match's single court calibration, shared across every set and rally.

    The camera doesn't move within a match, so one .out file at
    COURT_DIR/{match_id}.out covers every set and every rally in that match
    -- there's no per-set or per-rally variant.

    Args:
        match_id: The --id value.

    Returns:
        Calibration dict (see _load_manual_calibration()).

    Raises:
        FileNotFoundError: If COURT_DIR/{match_id}.out doesn't exist yet.
    """
    court_path = COURT_DIR / f"{match_id}.out"
    return _load_manual_calibration(court_path)


def process_match(
    match_id: str,
    overwrite_video: bool = False,
    max_height: int | None = None,
    force: bool = False,
) -> None:
    """Run the full ShuttleSet-to-HitNet pipeline for a single match id.

    Args:
        match_id:        The match id (e.g. "29").
        overwrite_video: Re-download the match video even if already present.
        max_height:      Optional cap on downloaded video resolution.
        force:           Reprocess every rally even if its four output
            files already exist (default: skip rallies that already
            completed).

    Returns:
        None.

    Raises:
        FileNotFoundError, ValueError: Match lookup/calibration setup
            errors (bad id, no match found in match.csv, missing court
            file). Unlike the old sys.exit(1) behaviour, these now
            propagate to the caller rather than killing the whole process
            -- see main(), which catches per-match so a batch of --id
            values can keep going past one bad match.
        RuntimeError: If video download fails, or no {match_id}_set*.csv
            files are found.
    """
    match_row = load_match_row(match_id)
    url = match_row["url"]
    expected_num_sets = int(match_row["set"])
    downcourt = int(match_row["downcourt"])

    try:
        video_path = download_match_video(
            url, match_id, overwrite=overwrite_video, max_height=max_height
        )
    except Exception as e:
        raise RuntimeError(f"Video download failed: {e}") from e

    set_csvs = discover_set_csvs(match_id)
    if not set_csvs:
        raise RuntimeError(f"No {match_id}_set*.csv files found in {SHUTTLESET_DIR}")

    print(f"[INFO] Found {len(set_csvs)} set(s): "
          f"{[n for n, _ in set_csvs]}")

    if len(set_csvs) != expected_num_sets:
        print(f"[WARN] match.csv says match {match_id} has {expected_num_sets} "
              f"set(s), but found {len(set_csvs)} set CSV file(s) on disk "
              f"({[n for n, _ in set_csvs]}). Proceeding with what was "
              f"found -- check for a missing/misnamed set file if this is "
              f"unexpected.")

    set1_csv_path = SHUTTLESET_DIR / f"{match_id}_set1.csv"
    ref_image_path = COURT_DIR / f"{match_id}_calibration_reference.png"
    if ref_image_path.is_file():
        print(f"[SKIP] Calibration reference frame already exists: {ref_image_path}")
    elif not set1_csv_path.is_file():
        print(f"[WARN] {set1_csv_path} not found -- can't extract a "
              f"calibration reference frame without it.")
    else:
        try:
            save_calibration_reference_frame(video_path, set1_csv_path, match_id)
        except Exception as e:
            print(f"[WARN] Could not save calibration reference frame: {e}")

    calibration = load_match_calibration(match_id)

    for set_num, set_csv_path in set_csvs:
        process_set(video_path, set_csv_path, match_id, downcourt, set_num, calibration, force=force)

 
def main() -> None:
    """CLI entry point.
 
    Runs process_match() once per --id given, in the order provided.
    A failure in one match (bad id, download failure, missing calibration,
    etc.) is logged and that match is skipped -- it does NOT abort the
    rest of the batch, so e.g. `--id 29 30 31` where match 30 has no
    court file yet will still fully process 29 and 31.
 
    Returns:
        None. Prints a final summary of which match ids succeeded/failed.
        Exits with sys.exit(1) only if every match in the batch failed.
    """
    parser = argparse.ArgumentParser(
        description="Convert one or more ShuttleSet matches into per-rally HitNet training inputs."
    )
    parser.add_argument("--id", required=True, nargs="+",
                        help="One or more ShuttleSet match ids to process "
                             "sequentially, e.g. --id 29 30 31")
    parser.add_argument("--overwrite_video", action="store_true",
                        help="Re-download the match video even if already present")
    parser.add_argument("--max_height", type=int, default=720,
                        help="Cap downloaded video resolution, e.g. 1080 "
                             "(default: best available)")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess every rally even if its four output "
                             "files already exist (default: skip rallies "
                             "that already completed)")
    parser.add_argument("--label_court", action="store_true",
                        help="Interactively mark court corners on the "
                             "calibration reference image for each --id, "
                             "then exit -- does not run the full rally "
                             "processing pipeline. Requires a GUI-capable "
                             "display (X11 forwarding if over SSH).")
    parser.add_argument("--overwrite_court", action="store_true",
                        help="With --label_court, relabel even if a "
                             ".out file already exists for that match")
    args = parser.parse_args()
 
    if args.label_court:
        for match_id in args.id:
            try:
                label_court_corners(match_id, overwrite=args.overwrite_court)
            except Exception as e:
                print(f"[ERROR] Match {match_id}: {type(e).__name__}: {e}",
                      file=sys.stderr)
        return
 
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []
 
    for i, match_id in enumerate(args.id, start=1):
        print(f"\n{'=' * 70}")
        print(f"[INFO] Match {i}/{len(args.id)}: id={match_id}")
        print(f"{'=' * 70}")
        try:
            process_match(
                match_id,
                overwrite_video=args.overwrite_video,
                max_height=args.max_height,
                force=args.force,
            )
            succeeded.append(match_id)
        except Exception as e:
            print(f"[ERROR] Match {match_id} failed: {type(e).__name__}: {e}",
                  file=sys.stderr)
            failed.append((match_id, str(e)))
 
    print(f"\n{'=' * 70}")
    print(f"[SUMMARY] {len(succeeded)}/{len(args.id)} match(es) completed")
    if succeeded:
        print(f"  Succeeded: {succeeded}")
    if failed:
        print(f"  Failed:")
        for mid, err in failed:
            print(f"    {mid}: {err}")
    print(f"{'=' * 70}")
 
    if failed and not succeeded:
        sys.exit(1)
 
    print("[INFO] Done.")
 
 
if __name__ == "__main__":
    main()
    