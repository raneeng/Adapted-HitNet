"""
format_shots.py
----------------
Companion to format_data.py: builds the shot-type training table.

From ShuttleSet it takes ONLY:
    1. the hit frame        -> rebased to the rally clip (same indexing as
                               _hit.csv and the ball/pose CSVs)
    2. the hitter           -> bottom/top, same convention as _hit.csv
    3. the shot type label  -> mapped to the analyst list (shot_mapping.py)

Every feature is then computed by shot_features.py from the ball, pose and
court files format_data.py already wrote to INPUT_DATA_DIR (data_pad15/) --
the same inputs BGL has at inference, so no annotator-only information
(ShuttleSet positions, heights, areas, ...) reaches the model.

For every rally whose four HitNet files exist, writes:

    INPUT_DATA_DIR/shot_labels/{stem}_shots.csv     one row per stroke
    INPUT_DATA_DIR/shot_labels/all_shots.csv        all rallies of this run

Label columns
-------------
    stem, match_id, set, rally, shot_index      (shot_index 1 = serve)
    frame, prev_frame, next_frame               clip-relative
    hitter ("bottom"/"top"), hitter_label (1/2)
    is_serve, serve_length ("short"/"long")
    shuttleset_type, backhand, analyst_label, base_type (training target)
    keep, drop_reason
Feature columns: see shot_features.py, plus rally context
    prev_launch_speed, prev_launch_vy_up, prev_contact_above_head,
    prev_hitter_depth       -- the previous stroke's values (blank for the first)

Train the rally-shot classifier on `keep & ~is_serve`; serves only need
short vs long.

Usage:
    cd data_prep
    python format_shots.py --id 29 30 31
    python format_shots.py --all          # every match with a court file
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from court_side import resolve_bottom_players_for_set, player_to_int_label
from format_data import (
    COURT_DIR,
    INPUT_DATA_DIR,
    SHUTTLESET_DIR,
    _rally_already_done,
    compute_rally_frame_range,
    discover_set_csvs,
    load_match_row,
)
from shot_features import load_rally, shot_features
from shot_mapping import SERVE_TYPES, base_type, is_backhand, map_shot, normalise_type

SHOT_LABEL_DIR = INPUT_DATA_DIR / "shot_labels"

# Rally context: each shot also gets these features of the PREVIOUS stroke in
# the same rally (the opponent's shot it is answering), as prev_<name>.
# Blank for the first stroke. In BGL the same values come from the previous
# detected shot, so they're available at inference too.
CONTEXT_FEATURES = ["launch_speed", "launch_vy_up", "contact_above_head", "hitter_depth"]


def build_rally_rows(rally_df: pd.DataFrame, stem: str, match_id: str, set_num: int,
                     rally_num: int, bottom_player: str) -> pd.DataFrame:
    """Labels + features, one row per stroke of one rally."""
    # Same start frame format_data.py used to slice the clip -> same indexing.
    start_frame, _ = compute_rally_frame_range(rally_df)
    rally = load_rally(stem, INPUT_DATA_DIR)

    strokes = rally_df.sort_values("ball_round").reset_index(drop=True)
    frames = [int(fn) - start_frame if pd.notna(fn) else None for fn in strokes["frame_num"]]

    rows = []
    for i, r in strokes.iterrows():
        f = frames[i]
        prev_f = frames[i - 1] if i > 0 else None
        next_f = frames[i + 1] if i + 1 < len(frames) else None

        stype = normalise_type(r["type"])
        backhand = is_backhand(r.get("backhand"))
        hitter_label = player_to_int_label(r["player"], bottom_player)
        # The first stroke is the serve -- unless ShuttleSet didn't annotate the
        # serve and the rally starts mid-play; then it's an ordinary rally shot.
        is_serve = i == 0 and stype in SERVE_TYPES

        row = {
            "stem": stem, "match_id": match_id, "set": set_num, "rally": rally_num,
            "shot_index": i + 1,
            "frame": f, "prev_frame": prev_f, "next_frame": next_f,
            "hitter": {1: "bottom", 2: "top"}.get(hitter_label, ""),
            "hitter_label": hitter_label,
            "is_serve": is_serve,
            "serve_length": SERVE_TYPES.get(stype, "") if is_serve else "",
            "shuttleset_type": stype, "backhand": backhand,
            "analyst_label": "", "base_type": "",
            "keep": True, "drop_reason": "",
        }

        try:
            analyst = map_shot(stype, backhand)
        except KeyError:
            analyst = None
            row.update(keep=False, drop_reason=f"unmapped type {stype!r}")
        if analyst is None and row["keep"]:
            reason = ("unknown type in ShuttleSet" if stype == "未知球種"
                      else f"no analyst label ({'BH' if backhand else 'FH'} {stype})")
            row.update(keep=False, drop_reason=reason)
        row["analyst_label"] = analyst or ""
        row["base_type"] = base_type(analyst) or ""

        if f is None or not 0 <= f < rally.n:
            row.update(keep=False, drop_reason=row["drop_reason"] or "frame missing or outside clip")
        else:
            row.update(shot_features(rally, f, hitter_label, next_f, prev_f))
        rows.append(row)

    out = pd.DataFrame(rows)
    # Previous stroke's features (rows are in stroke order within the rally).
    for col in CONTEXT_FEATURES:
        out[f"prev_{col}"] = out[col].shift(1) if col in out else np.nan
    return out


def check_against_hit_csv(rows: pd.DataFrame, stem: str) -> int:
    """Count strokes whose frame/hitter disagrees with the existing _hit.csv."""
    hit = pd.read_csv(INPUT_DATA_DIR / "shot" / f"{stem}_hit.csv")["hit"].to_numpy()
    bad = 0
    for f, label in zip(rows["frame"], rows["hitter_label"]):
        if pd.isna(f) or not 0 <= int(f) < len(hit) or hit[int(f)] != label:
            bad += 1
    return bad


def process_match(match_id: str) -> pd.DataFrame:
    """Write shot label files for every completed rally of one match."""
    downcourt = int(load_match_row(match_id)["downcourt"])
    set_csvs = discover_set_csvs(match_id)
    if not set_csvs:
        raise RuntimeError(f"No {match_id}_set*.csv files found in {SHUTTLESET_DIR}")

    SHOT_LABEL_DIR.mkdir(parents=True, exist_ok=True)
    match_rows, n_skipped, n_mismatch = [], 0, 0

    for set_num, set_csv_path in set_csvs:
        df = pd.read_csv(set_csv_path)
        rally_ids = sorted(df["rally"].dropna().unique())

        # Same bottom-player resolution as format_data.process_set()
        roundscores = []
        for rally_num in rally_ids:
            first = df[df["rally"] == rally_num].iloc[0]
            roundscores.append((int(first["roundscore_A"]), int(first["roundscore_B"])))
        bottom_players = resolve_bottom_players_for_set(
            downcourt=downcourt, set_num=set_num, roundscores=roundscores,
        )

        for i, rally_num in enumerate(rally_ids):
            stem = f"{match_id}_set{set_num}_{int(rally_num)}"
            if not _rally_already_done(stem):
                n_skipped += 1
                continue
            try:
                rows = build_rally_rows(df[df["rally"] == rally_num], stem, match_id,
                                        set_num, int(rally_num), bottom_players[i])
                n_mismatch += check_against_hit_csv(rows, stem)
                rows.to_csv(SHOT_LABEL_DIR / f"{stem}_shots.csv", index=False,
                            encoding="utf-8-sig")   # BOM so Excel shows the Chinese labels
                match_rows.append(rows)
            except Exception as e:
                print(f"[ERROR] {stem}: {type(e).__name__}: {e}")

    if not match_rows:
        raise RuntimeError("no rallies written (run format_data.py for this match first)")

    out = pd.concat(match_rows, ignore_index=True)
    print(f"[INFO] Match {match_id}: {len(match_rows)} rallies, {len(out)} strokes "
          f"({int(out['keep'].sum())} kept, {int(out['is_serve'].sum())} serves)")
    if n_skipped:
        print(f"[WARN] Match {match_id}: skipped {n_skipped} rally(ies) without all four HitNet files")
    if n_mismatch:
        print(f"[WARN] Match {match_id}: {n_mismatch} stroke(s) disagree with _hit.csv "
              f"(frame or hitter) -- check before training")
    return out


def all_match_ids() -> list[str]:
    """Every ShuttleSet match id that has a court .out file.

    Tagged-app matches (tag-*) are skipped: their labels come from the app's
    export, not ShuttleSet's per-set CSVs.
    """
    return sorted((p.stem for p in COURT_DIR.glob("*.out") if not p.stem.startswith("tag-")),
                  key=lambda m: (not m.isdigit(), int(m) if m.isdigit() else 0, m))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the shot-type training table.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", nargs="+", help="ShuttleSet match ids, e.g. --id 29 30")
    group.add_argument("--all", action="store_true",
                       help="Process every match that has a court .out file")
    args = parser.parse_args()

    match_ids = all_match_ids() if args.all else args.id
    frames, failed = [], []
    for match_id in match_ids:
        try:
            frames.append(process_match(match_id))
        except Exception as e:
            print(f"[ERROR] Match {match_id}: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(match_id)

    if not frames:
        sys.exit("[ERROR] No matches processed.")

    all_shots = pd.concat(frames, ignore_index=True)
    index_path = SHOT_LABEL_DIR / "all_shots.csv"
    all_shots.to_csv(index_path, index=False, encoding="utf-8-sig")

    kept = all_shots[all_shots["keep"]]
    print(f"\n[SUMMARY] {len(frames)}/{len(match_ids)} match(es), {len(all_shots)} strokes -> {index_path}")
    print("[SUMMARY] Rally shots by base type (kept, serves excluded):")
    print(kept[~kept["is_serve"]]["base_type"].value_counts().to_string())
    print("[SUMMARY] Serves:")
    print(kept[kept["is_serve"]]["serve_length"].value_counts().to_string())
    dropped = all_shots.loc[~all_shots["keep"], "drop_reason"].value_counts()
    if not dropped.empty:
        print("[SUMMARY] Dropped:")
        print(dropped.to_string())
    label_cols = {"set", "rally", "shot_index", "frame", "prev_frame", "next_frame",
                  "hitter_label", "flight_s", "prev_gap_s"}   # blank by design at rally ends
    nan_rate = kept.select_dtypes("number").drop(columns=label_cols, errors="ignore").isna().mean()
    nan_rate = nan_rate[nan_rate > 0.2].sort_values(ascending=False)
    if not nan_rate.empty:
        print("[SUMMARY] Features missing for >20% of kept shots (tracking/pose gaps):")
        print(nan_rate.round(2).to_string())
    if failed:
        print(f"[SUMMARY] Failed matches: {failed}")


if __name__ == "__main__":
    main()