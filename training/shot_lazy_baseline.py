"""
shot_lazy_baseline.py
----------------------
Quick model shortlist for the shot-type classifier with LazyClassifier.

Trains a set of standard classifiers with default settings on the
shot-feature table from format_shots.py and ranks them on the VALIDATION
matches. The TEST matches are left untouched -- use them only once, for the
final tuned model.

Same match-level split as HitNet (train_hitnet.py), so no match appears in
more than one split.

Usage (from training/):
    python shot_lazy_baseline.py
Writes:
    lazy_results.csv      one row per model, sorted by balanced accuracy
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from lazypredict.Supervised import LazyClassifier

SHOTS = Path("data_pad15/shot_labels/all_shots.csv")

# Same split as train_hitnet.py (ShuttleSet ids only -- tagged matches aren't
# in the shot table yet).
# Duplicates removed: ShuttleSet22 ids 45-51 and 96 are re-uploads of
# ShuttleSet 23, 38, 39, 41, 42, 43, 44 and 24 (same video), so 49/96 (val)
# and 47 (test) are gone.
VAL_MATCH_IDS = {
    "2", "7", "11", "18", "21", "29", "30", "31", "36", "40",
    "43", "53", "57", "60", "65", "67", "71", "76", "84",
    "89", "93",
}
TEST_MATCH_IDS = {"5", "16", "24", "38", "58", "69", "80", "91"}
DUPLICATE_MATCH_IDS = {"45", "46", "47", "48", "49", "50", "51", "96"}

# Model inputs: only features computed from TrackNet / YOLO / court.
# Left out on purpose: labels, ids, frames, hitter_label (just file naming),
# raw court x/y (not side-invariant), body_height_px (mostly encodes near/far),
# landing_source (text).
FEATURES = [
    "hitter_near",
    "hitter_lat", "hitter_depth", "opp_lat", "opp_depth", "land_lat", "land_depth",
    "shot_distance", "lat_change", "hitter_move_speed",
    "flight_s", "prev_gap_s", "is_last",
    "contact_above_head", "shuttle_detect_rate", "apex_above_start", "apex_above_end",
    "mean_speed", "img_dx", "img_dy_fwd", "launch_vx", "launch_vy_up", "launch_speed",
    "pose_detect_rate", "wrist_above_head", "elbow_angle", "shoulder_angle",
    "wrist_lat_offset", "knee_angle_min", "knee_angle_max", "stance_width",
    "torso_lean", "jump",
    # rally context: the previous stroke (the shot being answered)
    "prev_launch_speed", "prev_launch_vy_up", "prev_contact_above_head", "prev_hitter_depth",
]
TARGET = "base_type"


def shortlist():
    """A practical set of classifiers.

    LazyClassifier's default 'all' includes models that are very slow or run
    out of memory on ~30k+ rows (e.g. LabelPropagation/LabelSpreading build an
    n x n matrix, SVC scales badly), so pick the ones worth comparing.
    """
    from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier,
                                  HistGradientBoostingClassifier, AdaBoostClassifier)
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    models = [ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier,
              AdaBoostClassifier, LogisticRegression, RidgeClassifier,
              KNeighborsClassifier, DecisionTreeClassifier, GaussianNB,
              LinearDiscriminantAnalysis]
    try:
        from lightgbm import LGBMClassifier
        models.append(LGBMClassifier)
    except ImportError:
        print("[WARN] lightgbm not installed -- skipping LGBMClassifier")
    try:
        from xgboost import XGBClassifier
        models.append(XGBClassifier)
    except ImportError:
        print("[WARN] xgboost not installed -- skipping XGBClassifier")
    return models


def main() -> None:
    df = pd.read_csv(SHOTS, dtype={"match_id": str})
    df = df[df["keep"] & ~df["is_serve"]].copy()          # rally shots with a label
    dupes = df["match_id"].isin(DUPLICATE_MATCH_IDS)
    if dupes.any():                                        # safety net if a stale table slips in
        print(f"[WARN] dropping {dupes.sum()} shots from duplicate matches")
        df = df[~dupes]
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing feature columns (rerun format_shots.py?): {missing}")

    train = df[~df["match_id"].isin(VAL_MATCH_IDS | TEST_MATCH_IDS)]
    val = df[df["match_id"].isin(VAL_MATCH_IDS)]
    print(f"[INFO] train: {len(train)} shots ({train['match_id'].nunique()} matches) | "
          f"val: {len(val)} shots ({val['match_id'].nunique()} matches) | "
          f"test held out: {df['match_id'].isin(TEST_MATCH_IDS).sum()} shots")
    print("[INFO] train class counts:\n" + train[TARGET].value_counts().to_string())

    # True/False -> 1.0/0.0 here only; the CSV keeps True/False.
    X_tr = train[FEATURES].astype(float)
    X_va = val[FEATURES].astype(float)

    # Integer class labels (XGBoost requires 0..n-1).
    enc = LabelEncoder().fit(df[TARGET])
    y_tr, y_va = enc.transform(train[TARGET]), enc.transform(val[TARGET])
    print("[INFO] classes:", dict(enumerate(enc.classes_)))

    t0 = time.time()
    clf = LazyClassifier(verbose=0, ignore_warnings=True, predictions=False,
                         classifiers=shortlist())
    models, _ = clf.fit(X_tr, X_va, y_tr, y_va)
    models = models.sort_values("Balanced Accuracy", ascending=False)

    print(f"\n[INFO] done in {time.time() - t0:.0f}s -- ranked on VALIDATION matches:")
    print(models.to_string())
    models.to_csv("lazy_results.csv")
    print("[INFO] saved lazy_results.csv")


if __name__ == "__main__":
    main()