"""
eval_events.py
---------------
Event-level evaluation of a trained HitNet model.

train_hitnet.py scores every 12-frame WINDOW independently, so a hit that the
model places one frame early or late counts as an error at the window edges,
and one real hit produces many correlated windows. This script instead asks the
question that matters downstream:

    "For each real hit, did the model report a hit by the right player,
     within +/- T frames?"

Pipeline per rally:
  1. Build + scale windows exactly as train_hitnet.py does, predict.
  2. Window -> frame: each frame's class probabilities are the average over
     every window whose label region (frames LEFT_WINDOW..NUM_CONSEC-1 of the
     window) contains that frame.
  3. Frame -> events: runs of the same non-zero argmax class become ONE hit
     event, placed at the run's peak-probability frame. Same-class runs
     separated by <= --merge_gap no-hit frames are merged first.
  4. Ground truth events = frames where {stem}_hit.csv is non-zero.
  5. One-to-one greedy matching (closest first) within each tolerance.

Reports, for each tolerance:
  - Hitter-aware precision / recall / F1 per class (bottom, top) + macro-F1
  - Hitter-agnostic detection P / R / F1, and hitter accuracy of matched hits
Plus: timing-offset distribution, and a per-match breakdown (worst first) to
spot matches whose footage the model struggles with.

"Unscorable" ground-truth hits: a hit in the first LEFT_WINDOW frames of a clip
is never inside any window's label region, so no window can ever predict it.
format_data.py starts each rally clip exactly AT the first hit (the serve), so
every serve lands there. These are reported separately and excluded from the
main metrics; see the summary line for how many there are.

Usage (run from hitnet/training, in .venv-train):
    python eval_events.py                                   # hitnet_model.h5 on TEST
    python eval_events.py --model hitnet_model_19oct.h5
    python eval_events.py --split val --tolerances 0 1 2 3 5
    python eval_events.py --out test_events_per_match.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASS_NAMES = {1: "bottom", 2: "top"}


# ====== Window -> frame -> event conversion ===================================

def window_to_frame_probs(
    win_probs: np.ndarray,
    num_consec: int,
    left_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average each frame's class probabilities over the windows that label it.

    Window w spans frames w .. w+num_consec-1 and its label covers frames
    w+left_window .. w+num_consec-1.

    Args:
        win_probs: (n_windows, 3) softmax outputs, in window order.

    Returns:
        (frame_probs, covered): frame_probs is (n_frames, 3); frames no window
        labels (the first left_window frames) are set to pure no-hit.
        covered is a bool mask of frames at least one window labels.
    """
    n_win = len(win_probs)
    n_frames = n_win + num_consec - 1
    acc = np.zeros((n_frames, 3))
    cnt = np.zeros(n_frames)
    for k in range(left_window, num_consec):
        acc[k:k + n_win] += win_probs
        cnt[k:k + n_win] += 1
    covered = cnt > 0
    probs = np.zeros((n_frames, 3))
    probs[:, 0] = 1.0
    probs[covered] = acc[covered] / cnt[covered, None]
    return probs, covered


def frames_to_events(
    probs: np.ndarray,
    merge_gap: int = 2,
    min_run: int = 1,
) -> list[tuple[int, int]]:
    """Collapse per-frame predictions into discrete hit events.

    Args:
        probs: (n_frames, 3) per-frame class probabilities.
        merge_gap: Same-class runs separated by <= this many frames are merged
            (a single hit flickering on/off). Real consecutive hits are
            normally 10+ frames apart, so small values are safe.
        min_run: Drop runs shorter than this many frames.

    Returns:
        List of (frame, class) events, class in {1, 2}, frame = the run's
        peak-probability frame for that class.
    """
    cls = probs.argmax(axis=1)
    runs = []
    i, n = 0, len(cls)
    while i < n:
        c = cls[i]
        if c == 0:
            i += 1
            continue
        j = i
        while j + 1 < n and cls[j + 1] == c:
            j += 1
        runs.append([i, j, int(c)])
        i = j + 1

    merged: list[list[int]] = []
    for r in runs:
        if merged and merged[-1][2] == r[2] and r[0] - merged[-1][1] - 1 <= merge_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)

    events = []
    for s, e, c in merged:
        if e - s + 1 < min_run:
            continue
        peak = s + int(np.argmax(probs[s:e + 1, c]))
        events.append((peak, c))
    return events


def labels_to_events(hit: np.ndarray) -> list[tuple[int, int]]:
    """Ground-truth events: every frame with a non-zero hit label."""
    idx = np.flatnonzero(hit != 0)
    return [(int(f), int(hit[f])) for f in idx]


# ====== Matching ===============================================================

def match_events(
    gt: list[tuple[int, int]],
    pred: list[tuple[int, int]],
    tol: int,
    require_class: bool,
) -> list[tuple[int, int, int]]:
    """Greedy one-to-one matching, closest pairs first.

    Returns:
        List of (gt_index, pred_index, offset) with offset = pred_frame - gt_frame.
    """
    pairs = []
    for i, (gf, gc) in enumerate(gt):
        for j, (pf, pc) in enumerate(pred):
            d = abs(pf - gf)
            if d <= tol and (not require_class or gc == pc):
                pairs.append((d, i, j))
    pairs.sort()
    used_g, used_p, matched = set(), set(), []
    for _, i, j in pairs:
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        matched.append((i, j, pred[j][0] - gt[i][0]))
    return matched


def prf(tp: int, n_pred: int, n_gt: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# ====== Scoring ================================================================

class Tally:
    """Accumulates event counts for one tolerance (overall or one match)."""

    def __init__(self):
        self.tp_cls = defaultdict(int)       # hitter-aware TPs per class
        self.n_gt = defaultdict(int)
        self.n_pred = defaultdict(int)
        self.tp_any = 0                      # hitter-agnostic TPs
        self.tp_any_correct_hitter = 0
        self.offsets: list[int] = []

    def add(self, gt, pred, tol):
        for _, c in gt:
            self.n_gt[c] += 1
        for _, c in pred:
            self.n_pred[c] += 1
        for i, _, _ in match_events(gt, pred, tol, require_class=True):
            self.tp_cls[gt[i][1]] += 1
        agnostic = match_events(gt, pred, tol, require_class=False)
        self.tp_any += len(agnostic)
        self.tp_any_correct_hitter += sum(gt[i][1] == pred[j][1] for i, j, _ in agnostic)
        self.offsets.extend(o for _, _, o in agnostic)

    def class_prf(self, c):
        return prf(self.tp_cls[c], self.n_pred[c], self.n_gt[c])

    def macro_f1(self):
        return float(np.mean([self.class_prf(c)[2] for c in (1, 2)]))

    def detection_prf(self):
        return prf(self.tp_any, sum(self.n_pred.values()), sum(self.n_gt.values()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Event-level HitNet evaluation.")
    parser.add_argument("--model", default="hitnet_model.h5", help="Path to the .h5 model")
    parser.add_argument("--split", choices=["test", "val", "train"], default="test")
    parser.add_argument("--tolerances", type=int, nargs="+", default=[0, 1, 2, 3, 5],
                        help="Frame tolerances to report (default: 0 1 2 3 5)")
    parser.add_argument("--match_tol", type=int, default=3,
                        help="Tolerance used for the per-match breakdown (default: 3)")
    parser.add_argument("--merge_gap", type=int, default=2)
    parser.add_argument("--min_run", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None,
                        help="Optional CSV path for the per-match breakdown")
    parser.add_argument("--matches", nargs="+", default=None,
                        help="Only evaluate these match ids, e.g. --matches 21 29")
    args = parser.parse_args()

    import pandas as pd
    import tensorflow as tf
    import train_hitnet as th   # run from hitnet/training (BASE_DIR uses cwd)

    train_s, val_s, test_s = th.split_stems_by_match(th.discover_rally_stems())
    stems = {"train": train_s, "val": val_s, "test": test_s}[args.split]

    if args.matches:
        keep = {str(m) for m in args.matches}
        stems = [s for s in stems if th.stem_to_match_id(s) in keep]

    print(f"[INFO] {args.split}: {len(stems)} rallies, "
          f"{len({th.stem_to_match_id(s) for s in stems})} matches")

    # ---- Build all windows, predict in one go -------------------------------
    rallies, xs = [], []
    for stem in stems:
        result = th.build_rally_windows(stem)
        if result is None:
            continue
        x_t, _ = result
        n_frames = len(x_t) + th.NUM_CONSEC - 1
        hit = th.read_hit_labels(th.rally_file_paths(stem)["hit"])[:n_frames]
        rallies.append((stem, len(x_t), hit))
        xs.append(th.scale_data(x_t).astype(np.float32))

    model = tf.keras.models.load_model(
        args.model, compile=False,
        custom_objects={"MacroF1Score": th._build_macro_f1_metric()},
    )
    print(f"[INFO] Loaded {args.model}; predicting {sum(len(x) for x in xs)} windows...")
    all_probs = model.predict(np.vstack(xs), batch_size=th.VAL_BATCH_SIZE, verbose=0)

    # ---- Score ---------------------------------------------------------------
    overall = {t: Tally() for t in args.tolerances}
    per_match: dict[str, Tally] = defaultdict(Tally)
    n_unscorable = defaultdict(int)
    n_gt_total = defaultdict(int)

    offset = 0
    for stem, n_win, hit in rallies:
        win_probs = all_probs[offset:offset + n_win]
        offset += n_win
        probs, covered = window_to_frame_probs(win_probs, th.NUM_CONSEC, th.LEFT_WINDOW)
        pred = frames_to_events(probs, args.merge_gap, args.min_run)

        gt_all = labels_to_events(hit)
        gt = [(f, c) for f, c in gt_all if covered[f]]
        for f, c in gt_all:
            n_gt_total[c] += 1
            if not covered[f]:
                n_unscorable[c] += 1

        for t in args.tolerances:
            overall[t].add(gt, pred, t)
        per_match[th.stem_to_match_id(stem)].add(gt, pred, args.match_tol)

    # ---- Report --------------------------------------------------------------
    print(f"\n[INFO] Ground-truth hits: {sum(n_gt_total.values())} "
          f"(bottom {n_gt_total[1]}, top {n_gt_total[2]})")
    unsc = sum(n_unscorable.values())
    print(f"[INFO] Unscorable (in first {th.LEFT_WINDOW} frames of a clip, e.g. serves): "
          f"{unsc} ({100 * unsc / max(1, sum(n_gt_total.values())):.1f}%) -- "
          f"excluded below; no window can predict them.")

    print(f"\n{'tol':>4} | {'bottom P/R/F1':^20} | {'top P/R/F1':^20} | {'macroF1':>7} "
          f"| {'detect P/R/F1':^20} | {'hitter acc':>10}")
    print("-" * 98)
    for t in args.tolerances:
        tl = overall[t]
        b, tp = tl.class_prf(1), tl.class_prf(2)
        d = tl.detection_prf()
        hacc = tl.tp_any_correct_hitter / tl.tp_any if tl.tp_any else 0.0
        fmt = lambda v: f"{v[0]:.3f}/{v[1]:.3f}/{v[2]:.3f}"
        print(f"{t:>4} | {fmt(b):^20} | {fmt(tp):^20} | {tl.macro_f1():7.3f} "
              f"| {fmt(d):^20} | {hacc:10.3f}")

    t_max = max(args.tolerances)
    offs = np.array(overall[t_max].offsets)
    if len(offs):
        print(f"\n[INFO] Timing offset of matched hits at tol={t_max} "
              f"(pred - true, frames): mean {offs.mean():+.2f}, "
              f"mean |offset| {np.abs(offs).mean():.2f}")
        values, counts = np.unique(offs, return_counts=True)
        print("       " + "  ".join(f"{v:+d}:{c}" for v, c in zip(values, counts)))

    rows = []
    for mid, tl in per_match.items():
        b, tp = tl.class_prf(1), tl.class_prf(2)
        d = tl.detection_prf()
        rows.append({
            "match": mid, "n_hits": sum(tl.n_gt.values()),
            "bottom_recall": b[1], "top_recall": tp[1],
            "bottom_precision": b[0], "top_precision": tp[0],
            "macro_f1": tl.macro_f1(), "detect_f1": d[2],
        })
    df = pd.DataFrame(rows).sort_values("macro_f1")
    print(f"\n[INFO] Per-match breakdown at tol={args.match_tol} (worst first):")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\n[INFO] Wrote {args.out}")


if __name__ == "__main__":
    main()