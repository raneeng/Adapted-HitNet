"""
format_tagged.py
-----------------
Tagging-app counterpart to format_data.py. Converts matches tagged in the
tagging app (shot CSV + LOCAL match video) into the same four per-rally
HitNet inputs that format_data.py produces from ShuttleSet:

    ball_trajectory/{stem}_ball_predicted.csv
    poses/{stem}_player_bottom.csv
    poses/{stem}_player_top.csv
    shot/{stem}_hit.csv

It reuses format_data.process_rally() unchanged -- this file only adapts the
input side (folder layout, CSV schema, court side). Any change to rally
slicing in format_data.py (e.g. START_PAD_FRAMES) applies here automatically.

Expected layout (mirrors ShuttleSet's match.csv + per-match files):

    data_prep/tagged/
        match.csv            <- one row per match; must have an `id` column,
                                any other columns are free-form match info
        video/{id}.mp4       <- the match video
        shot/{id}.csv        <- the tagging-app export for that video

Output naming:
    Tagged ids are prefixed with "tag-" so they can never collide with
    ShuttleSet's numeric ids (tagged id "7" -> match id "tag-7", court file
    court/tag-7.out, rally stems tag-7_set1_{rally}). Characters other than
    letters/digits/hyphens are replaced with "-", so stems always split
    correctly on "_set" in train_hitnet.py. Each video is treated as a single
    "set" (set1).

Tagging-app CSV columns used:
    Rally, Frame, Player   (Player is "near"/"far" per shot)
    Every row counts as a hit (serves included); other columns are ignored.

Workflow per match (same as ShuttleSet):
    1. python format_tagged.py --id 7
         -> saves court/tag-7_calibration_reference.png, then stops because
            court/tag-7.out doesn't exist yet
    2. Label the court in label_court.ipynb with match id "tag-7"
    3. Rerun step 1 -> processes every rally

Usage (from data_prep/, in .venv):
    python format_tagged.py --id 7 8 9
    python format_tagged.py --all
    python format_tagged.py --all --force
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import pandas as pd

from format_data import (
    COURT_DIR,
    DATA_PREP_DIR,
    NEAR_IS_BOTTOM,
    load_match_calibration,
    process_rally,
    write_court_out,
)

# ── Configuration ───────────────────────────────────────────────────────────

TAGGED_DIR = DATA_PREP_DIR / "tagged"
TAGGED_MATCH_CSV = TAGGED_DIR / "match.csv"
TAGGED_VIDEO_DIR = TAGGED_DIR / "video"
TAGGED_SHOT_DIR = TAGGED_DIR / "shot"
TAGGED_CALIB_DIR = TAGGED_DIR / "calibration"

EXPECTED_COURT_W = 6.1
EXPECTED_COURT_L = 13.4

MATCH_ID_PREFIX = "tag-"
TAG_REQUIRED_COLUMNS = {"Rally", "Frame", "Player"}

# Rallies with fewer tagged hits than this are skipped. Single-hit rallies in
# the sample exports were mostly serve faults or tagging slips (e.g. a lone
# "Net kill" as its own rally) and give HitNet almost nothing to learn from.
MIN_HITS_PER_RALLY = 2


# ── 1. Match lookup ─────────────────────────────────────────────────────────

def output_match_id(tagged_id: str) -> str:
    """Map a tagged match id to the id used in output filenames.

    "7" -> "tag-7", "YEO_AN" -> "tag-YEO-AN".
    """
    clean = re.sub(r"[^A-Za-z0-9-]+", "-", str(tagged_id).strip()).strip("-")
    if not clean:
        raise ValueError(f"Tagged id {tagged_id!r} has no usable characters")
    return MATCH_ID_PREFIX + clean


def load_tagged_matches() -> pd.DataFrame:
    """Read tagged/match.csv (ids as strings) and check ids are unique."""
    if not TAGGED_MATCH_CSV.is_file():
        raise FileNotFoundError(f"match.csv not found at {TAGGED_MATCH_CSV}")
    df = pd.read_csv(TAGGED_MATCH_CSV, dtype={"id": str})
    if "id" not in df.columns:
        raise ValueError(f"{TAGGED_MATCH_CSV} has no `id` column")
    df["id"] = df["id"].str.strip()
    dupes = df["id"][df["id"].duplicated()].unique()
    if len(dupes):
        raise ValueError(f"{TAGGED_MATCH_CSV} has duplicate ids: {list(dupes)}")
    return df


def match_paths(tagged_id: str) -> tuple[Path, Path]:
    """Return (video_path, shot_csv_path) for a tagged id, checking both exist."""
    video_path = TAGGED_VIDEO_DIR / f"{tagged_id}.mp4"
    shot_path = TAGGED_SHOT_DIR / f"{tagged_id}.csv"
    calib_path = TAGGED_CALIB_DIR / f"{tagged_id}.json"

    missing = [str(p) for p in (video_path, shot_path, calib_path) if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing file(s) for id={tagged_id!r}: {missing}")
    return video_path, shot_path, calib_path


def corners_to_adgx(points: list[list[float]]) -> list[tuple[float, float]]:
    """Order 4 court corners as A (top-left), D (top-right), G (bottom-left),
    X (bottom-right) by their IMAGE position, which is the .out convention.
 
    Ordering by image geometry (rather than trusting the JSON's point order)
    works whichever corner the calibration tool started from:
    the two points with the smallest y are the top (far) baseline, and within
    each pair the smaller x is the left corner.
    """
    if len(points) != 4:
        raise ValueError(f"Expected 4 corner points, got {len(points)}")
    pts = sorted(((float(x), float(y)) for x, y in points), key=lambda p: p[1])
    top, bottom = pts[:2], pts[2:]
    if max(p[1] for p in top) >= min(p[1] for p in bottom):
        raise ValueError(f"Can't separate top/bottom baselines from {points}")
    a, d = sorted(top)
    g, x = sorted(bottom)
    return [a, d, g, x]


def court_out_from_calibration(json_path: Path, out_path: Path) -> Path:
    with open(json_path) as f:
        calib = json.load(f)
    if "src_points" not in calib:
        raise ValueError(f"{json_path.name} has no `src_points`")
 
    w_m = calib.get("court_width_m")
    l_m = calib.get("court_length_m")
    if w_m is not None and l_m is not None and (
            abs(float(w_m) - EXPECTED_COURT_W) > 0.05 or abs(float(l_m) - EXPECTED_COURT_L) > 0.05):
        raise ValueError(
            f"{json_path.name} is calibrated on a {w_m} x {l_m} m court; this pipeline "
            f"expects the doubles outer lines ({EXPECTED_COURT_W} x {EXPECTED_COURT_L} m)."
        )
 
    adgx = corners_to_adgx(calib["src_points"])
    write_court_out(adgx, out_path)
    labels = ["A (top-left)", "D (top-right)", "G (bottom-left)", "X (bottom-right)"]
    print(f"[INFO] Wrote {out_path} from {json_path.name}: " +
          ", ".join(f"{lab[0]}=({px:.0f},{py:.0f})" for lab, (px, py) in zip(labels, adgx)))
    return out_path    

# ── 2. Normalise the tagging CSV into format_data's rally_df shape ──────────

def load_tagged_csv(csv_path: Path) -> pd.DataFrame:
    """Read a tagging-app CSV and normalise it for process_rally().

    Returns:
        DataFrame with exactly the columns process_rally() reads:
        rally, frame_num, player ("near"/"far").
    """
    raw = pd.read_csv(csv_path)
    missing = TAG_REQUIRED_COLUMNS - set(raw.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing column(s): {sorted(missing)}")

    rows = pd.DataFrame({
        "rally": raw["Rally"],
        "frame_num": raw["Frame"],
        "player": raw["Player"].astype(str).str.strip().str.lower(),
    }).dropna(subset=["rally", "frame_num"])

    bad = rows[~rows["player"].isin(["near", "far"])]
    if not bad.empty:
        raise ValueError(
            f"{csv_path.name}: unexpected Player values "
            f"{sorted(bad['player'].unique())} (expected near/far)"
        )
    return rows


# ── 3. Calibration reference frame (seek directly; videos can be long) ──────

def save_reference_frame(video_path: Path, frame_num: int, match_id: str) -> Path:
    """Save one frame (the first tagged hit) for manual court labelling."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_num} from {video_path}")

    COURT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COURT_DIR / f"{match_id}_calibration_reference.png"
    cv2.imwrite(str(out_path), img)
    print(f"[INFO] Saved calibration reference (frame {frame_num}) -> {out_path}")
    return out_path


def video_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


# ── 4. Per-match orchestration ──────────────────────────────────────────────

def process_tagged_match(tagged_id: str, match_info: pd.Series | None = None,
                         force: bool = False, overwrite_court: bool = False) -> None:
    """Produce HitNet inputs for every rally of one tagged match."""
    video_path, shot_path, calib_path = match_paths(tagged_id)
    rows = load_tagged_csv(shot_path)
    match_id = output_match_id(tagged_id)

    info = ""
    if match_info is not None:
        extras = {k: v for k, v in match_info.items() if k != "id" and pd.notna(v)}
        if extras:
            info = " | " + ", ".join(f"{k}={v}" for k, v in extras.items())
    print(f"[INFO] id={tagged_id} -> {match_id}: {video_path.name} "
          f"({video_fps(video_path):.2f} fps), {rows['rally'].nunique()} rallies{info}")

    ref_path = COURT_DIR / f"{match_id}_calibration_reference.png"
    if not ref_path.is_file():
        save_reference_frame(video_path, int(rows["frame_num"].min()), match_id)

    court_path = COURT_DIR / f"{match_id}.out"
    if calib_path.is_file() and (overwrite_court or not court_path.is_file()):
        court_out_from_calibration(calib_path, court_path)

    if not court_path.is_file():
        raise FileNotFoundError(
            f"{court_path} doesn't exist yet. Label the court in "
            f"label_court.ipynb with match id {match_id!r}, then rerun."
        )
    calibration = load_match_calibration(match_id)

    # The tagging app already says which side hit each shot, so the
    # "bottom player" is a fixed side rather than resolved per rally.
    bottom_side = "near" if NEAR_IS_BOTTOM else "far"

    n_done = n_skipped = n_failed = 0
    for rally_num, rally_df in rows.groupby("rally", sort=True):
        stem = f"{match_id}_set1_{int(rally_num)}"
        if len(rally_df) < MIN_HITS_PER_RALLY:
            print(f"[SKIP] {stem}: only {len(rally_df)} tagged hit(s)")
            n_skipped += 1
            continue
        try:
            process_rally(video_path, rally_df, stem, calibration,
                          bottom_side, force=force)
            n_done += 1
        except Exception as e:
            print(f"[ERROR] {stem}: {type(e).__name__}: {e}")
            n_failed += 1

    print(f"[INFO] {match_id}: {n_done} processed, {n_skipped} skipped, {n_failed} failed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert tagging-app matches (data_prep/tagged/) into HitNet inputs."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", nargs="+",
                       help="One or more ids from tagged/match.csv, e.g. --id 7 8 9")
    group.add_argument("--all", action="store_true",
                       help="Process every id in tagged/match.csv")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess rallies whose outputs already exist")
    args = parser.parse_args()

    matches = load_tagged_matches().set_index("id", drop=False)
    ids = list(matches.index) if args.all else [str(i).strip() for i in args.id]

    unknown = [i for i in ids if i not in matches.index]
    if unknown:
        print(f"[WARN] Not in {TAGGED_MATCH_CSV.name}, skipping: {unknown}", file=sys.stderr)
        ids = [i for i in ids if i in matches.index]

    succeeded, failed = [], []
    for n, tagged_id in enumerate(ids, 1):
        print(f"\n{'=' * 70}\n[INFO] Match {n}/{len(ids)}: id={tagged_id}\n{'=' * 70}")
        try:
            process_tagged_match(tagged_id, matches.loc[tagged_id], force=args.force)
            succeeded.append(tagged_id)
        except Exception as e:
            print(f"[ERROR] id={tagged_id}: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append((tagged_id, str(e)))

    print(f"\n{'=' * 70}\n[SUMMARY] {len(succeeded)}/{len(ids)} match(es) completed")
    if succeeded:
        print(f"  Succeeded: {succeeded}")
    if failed:
        print("  Failed:")
        for mid, err in failed:
            print(f"    {mid}: {err}")
    print("=" * 70)
    if failed and not succeeded:
        sys.exit(1)


if __name__ == "__main__":
    main()