"""
shot_lgbm.py
-------------
Tune and evaluate the LightGBM shot-type classifier (the LazyClassifier winner).

1. Small grid over the settings that matter most for LightGBM, each trained
   with early stopping on the VALIDATION matches.
2. For the best config: per-class report, confusion matrix and feature
   importance on validation.
3. Saves the model bundle (model + class names + feature list).
4. With --test: evaluates the saved model ONCE on the held-out TEST matches.
   Only do this after you've finished tuning -- that's the number to report.

Usage (from training/, in .venv-shot):
    python shot_lgbm.py           # tune + validation report, saves shot_lgbm.joblib
    python shot_lgbm.py --test    # final test evaluation of shot_lgbm.joblib
"""

from __future__ import annotations

import argparse
import itertools

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from shot_lazy_baseline import (DUPLICATE_MATCH_IDS, FEATURES, SHOTS, TARGET,
                                TEST_MATCH_IDS, VAL_MATCH_IDS)

MODEL_PATH = "shot_lgbm.joblib"

GRID = {
    "num_leaves": [31, 63, 127],
    "min_child_samples": [20, 50],
    "class_weight": [None, "balanced"],
}
FIXED = dict(learning_rate=0.05, n_estimators=3000, subsample=0.8, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1)


def load():
    df = pd.read_csv(SHOTS, dtype={"match_id": str})
    df = df[df["keep"] & ~df["is_serve"] & ~df["match_id"].isin(DUPLICATE_MATCH_IDS)].copy()
    X = df[FEATURES].astype(float)
    return df, X


def report(name, y_true, y_pred, classes):
    print(f"\n===== {name}: {len(y_true)} shots =====")
    print(f"accuracy {np.mean(y_true == y_pred):.3f} | balanced accuracy "
          f"{balanced_accuracy_score(y_true, y_pred):.3f} | macro F1 "
          f"{f1_score(y_true, y_pred, average='macro'):.3f}")
    print(classification_report(y_true, y_pred, labels=range(len(classes)),
                                target_names=classes, digits=3, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())
    # biggest confusions, to see which classes get mixed up
    off = [(cm[i, j], classes[i], classes[j]) for i in range(len(classes))
           for j in range(len(classes)) if i != j]
    print("\nTop confusions (true -> predicted):")
    for n, a, b in sorted(off, reverse=True)[:8]:
        print(f"  {a:>9} -> {b:<9} {n:5d}  ({n / cm[classes.index(a)].sum():.1%} of {a})")


def tune():
    df, X = load()
    classes = sorted(df[TARGET].unique())
    y = df[TARGET].map({c: i for i, c in enumerate(classes)}).to_numpy()
    tr = ~df["match_id"].isin(VAL_MATCH_IDS | TEST_MATCH_IDS).to_numpy()
    va = df["match_id"].isin(VAL_MATCH_IDS).to_numpy()
    print(f"[INFO] train {tr.sum()} | val {va.sum()} shots | classes: {classes}")

    results = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID.keys(), values), **FIXED)
        m = lgb.LGBMClassifier(**params)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        pred = m.predict(X[va])
        bal = balanced_accuracy_score(y[va], pred)
        mf1 = f1_score(y[va], pred, average="macro")
        results.append((bal, mf1, m.best_iteration_, params))
        print(f"  {dict(zip(GRID.keys(), values))} -> bal acc {bal:.4f}, macro F1 {mf1:.4f}, "
              f"trees {m.best_iteration_}")

    bal, mf1, n_trees, best = max(results, key=lambda r: r[0])
    chosen = {k: best[k] for k in GRID}
    print(f"\n[INFO] best (by balanced accuracy): {chosen} -> {bal:.4f}, trees {n_trees}")

    # refit best on train with the early-stopped tree count, report on val
    final = lgb.LGBMClassifier(**{**best, "n_estimators": n_trees})
    final.fit(X[tr], y[tr])
    report("VALIDATION (used for tuning)", y[va], final.predict(X[va]), classes)

    imp = pd.Series(final.booster_.feature_importance("gain"), index=FEATURES)
    print("\nFeature importance (gain, top 15):")
    print((imp / imp.sum()).sort_values(ascending=False).head(15).round(3).to_string())

    joblib.dump({"model": final, "classes": classes, "features": FEATURES,
                 "params": {**best, "n_estimators": n_trees}}, MODEL_PATH)
    print(f"\n[INFO] saved {MODEL_PATH}")


def test():
    bundle = joblib.load(MODEL_PATH)
    df, X = load()
    te = df["match_id"].isin(TEST_MATCH_IDS).to_numpy()
    classes = bundle["classes"]
    y = df[TARGET].map({c: i for i, c in enumerate(classes)}).to_numpy()
    report("TEST (held out)", y[te], bundle["model"].predict(X[te][bundle["features"]]), classes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="evaluate the saved model on TEST")
    test() if ap.parse_args().test else tune()