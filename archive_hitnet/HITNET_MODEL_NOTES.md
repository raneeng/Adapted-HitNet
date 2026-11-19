# HitNet model iterations

Running notes on each HitNet hit-detection model: what it was trained on, what
was wrong with that data (where known), and how it scores on a common test set.

> **Status (19 Nov 2026):** v0, v1, v2 and v2.1 are documented below.
> **v2 is the deployed model.** v2.1 (retrained on the cleaned data) scores
> slightly lower on detection but attributes the hitter more accurately; it is
> the leak-free baseline for future changes.

---

## Summary

All models evaluated with `eval_events.py` on the **same, cleaned test set**
(see [Evaluation setup](#evaluation-setup)), event-level, tolerance **±3 frames**:

| Model | Trained on | Serves | Macro F1 | Detect P / R | Bottom F1 | Top F1 | Hitter acc. | Worst match (macro F1) |
|---|---|:---:|---:|---|---:|---:|---:|---|
| v0 `hitnet_model_v0.h5` | smaller dataset (original ShuttleSet) | No | 0.895 | 0.912 / 0.898 | 0.925 | 0.865 | 98.8% | 16 (0.858) |
| v1 `hitnet_model_v1.h5` | `data/` (92 ShuttleSet and ShuttleSet22 matches) | No | 0.909 | 0.948 / 0.886 | 0.925 | 0.894 | 99.3% | 38 (0.876) |
| v2 `hitnet_model_v2.h5` | `data_pad15/` (Same 92 matches, with 15-frame serve pre-roll) | **Yes** | **0.949** | **0.946 / 0.966** | **0.963** | **0.935** | 99.1% | **16 (0.910)** |
| v2.1 `hitnet_model_v2-1.h5` | cleaned `data_pad15/` (duplicates removed, poses fixed) | **Yes** | 0.940 | 0.935 / 0.948 | 0.955 | 0.924 | **99.8%** | 38 (0.899) |

Macro F1 at other tolerances:

| Model | ±0 | ±1 | ±2 | ±3 | ±5 | Mean \|timing offset\| (frames) |
|---|---:|---:|---:|---:|---:|---:|
| v0 | 0.410 | 0.691 | 0.856 | 0.895 | 0.908 | 0.86 |
| v1 | 0.430 | 0.708 | 0.872 | 0.909 | 0.919 | 0.83 |
| v2 | 0.447 | 0.752 | 0.915 | 0.949 | 0.959 | 0.81 |
| v2.1 | 0.436 | 0.738 | 0.901 | 0.940 | 0.951 | 0.83 |

**Caveats when comparing:**
- **v0 and v1 were never trained on serves**, but this test set includes them
  (~10% of hits). That is the main reason their recall is lower (0.898 / 0.886
  vs 0.966).
- **v1 and v2 were trained on data containing duplicate matches**
  (see [Known data issues](#known-data-issues)); _e.g. test match **38** has a
  duplicate (46) in their training set, so their score on match 38 is
  optimistic._ v2.1 is the only model with no leakage, which is why match 38
  is its worst match. Excluding 38, v2 still leads (0.950 vs 0.945).
- ±2–3 frames is the realistic tolerance: ShuttleSet's frame annotations are
  only accurate to about ±1–2 frames.

---

## Evaluation setup

- Script: `eval_events.py --model <file>` (event-level: each true hit must be
  matched by a prediction for the right player within ±T frames; runs of hit
  frames collapsed to one event at their peak).
- Data: `training/data_pad15/` (clips with 15-frame serve pre-roll), so serves
  are scorable (0 unscorable hits).
- Test set (cleaned): 9 matches, 783 rallies, 7,682 hits (bottom 3,840 /
  top 3,842): ShuttleSet `5, 16, 24, 38, 58, 69, 80, 91` + tagged `tag-3`.
  Match 47 was removed from test because it duplicates match 39 (train).

---

## Iterations

### v0 -- `hitnet_model_v0.h5` (original)
 
- Trained 3 Aug 2026.
- Data: **2,504 rallies from 30 ShuttleSet matches** -- ids 1–7, 21–26,
  28–44. Videos downloaded at **1080p** at the time.
- Split by match, **train / validation only (no test split):**
  - Train (24): 1, 2, 3, 4, 5, 6, 22, 23, 24, 25, 26, 28, 32, 33, 34, 35,
    37, 38, 39, 40, 41, 42, 43, 44
  - Validation (6): 7, 21, 29, 30, 31, 36
- No serve pre-roll: clips start on the serve frame, so serves can't be detected.
- Pipeline: Bayesian search over GRU layers / units /
  L2 / dropout, augmentation (reflection, frame drop/corruption, small
  rotation/shear), training-set rebalancing, temperature scaling.
  <!-- TODO: winning hyperparameters, if recorded -->
- Original **window-level** result:
  | Class | Precision | Recall | F1 | Support |
  |---|---:|---:|---:|---:|
  | No hit (0) | 0.972 | 0.932 | 0.951 | 105,495 |
  | Bottom hit (1) | 0.780 | 0.911 | 0.840 | 12,793 |
  | Top hit (2) | 0.736 | 0.858 | 0.792 | 13,139 |

  Accuracy 92.2%, macro F1 0.861, weighted F1 0.925; 11.0% of true hit
  windows (2,861 / 25,932) predicted as no hit. Not comparable with the
  event-level table above.

### v1 -- `hitnet_model_v1.h5`

- Trained 19 Oct 2026.
- Data: `training/data/` -- 7,809 rallies from 92 ShuttleSet / ShuttleSet22
  matches (ids 1–96 minus 9, 10, 12, 27), split by match 60 / 23 / 9.
- Clips start on the serve frame -> **serves not detectable** (~10.5% of hits).
- Model: bidirectional GRU, 3 layers × 256 units, dropout 0, L2 1.05e-5,
  learning rate 5.1e-4, batch 256 (30-trial Bayesian search on val macro-F1).
- Data issues found later: duplicate matches (below).

### v2 -- `hitnet_model_v2.h5`

- Data: `training/data_pad15/` -- clips start **15 frames before the serve**,
  so every window labelling the serve can see it (`START_PAD_FRAMES = 15`).
  Added tagged matches: `tag-1` (val), `tag-3` (test), others in train.
- Model: bidirectional GRU, 4 layers × 256 units, dropout 0, L2 4.7e-7,
  learning rate 5.6e-4 (settled by two searches, Oct 2026; `FIXED_HP`).
- Main gain: serves detectable -> end-to-end recall 0.966 (vs 0.886 for v1).
- Data issues found later: duplicate matches **and** missing far-player poses
  in 13 matches (both below).

### v2.1 -- `hitnet_model_v2-1.h5`

- Trained 19 Nov 2026 on cleaned `data_pad15/`: duplicate matches removed, the
  13 broken-pose matches regenerated with corrected court files.
- Split: 58 train / 22 val / 9 test matches (4,874 / 1,722 / 783 rallies).
- Model: same hyperparameters as v2 (`FIXED_HP`): 4 × 256, dropout 0,
  L2 4.7e-7, LR 5.6e-4. Stopped at epoch 28 (best val epoch 18).
- Window-level macro F1: val 0.848, test 0.853.
- A first retrain with a new 30-trial Bayesian search (best: 4 × 256,
  dropout 0, L2 1.25e-6, LR 6.9e-4; kept as `hitnet_model_v2-1-search.h5`)
  scored about the same: ±3 macro F1 0.938, hitter acc. 99.5%. So the
  hyperparameters aren't the cause of the difference from v2.
- Per-match vs v2 (±3 macro F1):

  | Match | v2 | v2.1 |
  |---|---:|---:|
  | 38 (duplicate 46 was in v2 train) | 0.943 | 0.899 |
  | 16 | 0.911 | 0.908 |
  | 24 | 0.926 | 0.921 |
  | 80 | 0.931 | 0.934 |
  | 69 | 0.960 | 0.945 |
  | 58 | 0.962 | 0.958 |
  | 91 | 0.963 | 0.957 |
  | 5 | 0.967 | 0.961 |
  | tag-3 | 0.977 | 0.963 |
  | **Excl. 38, weighted by hits** | **0.950** | **0.945** |

- **Findings:**
  - The cleaned data costs ~0.006 macro F1 on the leak-free matches, mostly
    lower recall (0.948 vs 0.966). Likely causes: fewer training rallies once
    the duplicate re-uploads were removed, and v2's val set contained 96 (a
    duplicate of test match 24), which steered its epoch selection.
  - Hitter attribution improves: 99.8% vs 99.1%, about 4× fewer
    wrong-player hits (likely from the fixed far-player poses).
  - None of the 13 previously broken-pose matches is in the test set, so the
    pose fix can't show up in detection scores.
- **Decision:** v2 stays deployed (fewer errors overall: it misses ~140 fewer
  hits). v2.1 is kept as the leak-free baseline for future changes.

---

## Known data issues

### 1. Duplicate matches (affects v1, v2; fixed in v2.1)

ShuttleSet22 ids **45–51 and 96** are re-uploads of earlier ShuttleSet matches
(same YouTube URL in `match.csv`):

| Removed | Same video as | Split conflict (old split) |
|---|---|---|
| 45 | 23 | both train |
| 46 | 38 | 46 train, **38 test** |
| 47 | 39 | **47 test**, 39 train |
| 48 | 41 | both train |
| 49 | 42 | **49 val**, 42 train |
| 50 | 43 | 50 train, **43 val** |
| 51 | 44 | both train |
| 96 | 24 | **96 val**, **24 test** |

Their data was moved out of `data_pad15/`;

### 2. Missing far-player poses in 13 matches (affects v2; fixed in v2.1)

Matches **1, 2, 3, 4, 7, 21, 22, 25, 26, 29, 31, 36, 40** had one player's pose
CSV almost entirely zero in `data_pad15/`.

- Cause: their court `.out` files were labelled on **1080p** video in Aug 2026,
  but `data_pad15/` uses **720p** downloads, so the court outline was ~1.5×
  too large and `pose_features.py` treated the far player as off-court.
- Fix: corners scaled by 2/3 (match 7 relabelled -- one corner was also wrong),
  matches regenerated with `format_data.py --force`. Far-player detection went
  from ~0% to 100% on spot-checked rallies.
- Prevention to add: reject a `.out` whose corners fall outside the video frame.

### 3. Pose file names are swapped (historical -- keep them, match the order in BGL)
 
`format_data.py` writes the **near** player (bottom of the image, hitter
label **1**) into `_player_top.csv`, and the **far** player (label 2) into
`_player_bottom.csv`.
 
- **Don't rename the files.** HitNet only sees the order of the two poses in
  its 78-value input, and every model (v0–v2.1) was trained with this order.
  Renaming would mean updating `format_data.py`, the training loader,
  `eval_events.py` and `shot_features.py` together, for no gain in accuracy.
- **BGL must put the near player's pose (P1) in the same position** of
  HitNet's input as `_player_top.csv` fills in training, or hitter
  attribution flips. To check: run BGL on `tag-3` and compare hitters with
  the tags. Offline, v2 gets 99% of hitters right.
- `shot_features.py` handles the mapping via `POSE_FILE_FOR_LABEL`
  (`{1: "top", 2: "bottom"}`).
- Comment to keep in `format_data.py` (where the files are written) and in the
  training loader (where they're read):
  `# NOTE: _player_top.csv holds the NEAR (bottom-of-image, label 1) player. Name is historical; do not swap.`
---

## Training environment notes

- Train with `.venv-train` (TensorFlow + CUDA 12). Running from `.venv` falls
  back to CPU silently (~12 min/epoch instead of ~1 min).
- Changing the search space or the data invalidates `tuner/` -- delete it
  before running the Bayesian search again (`FIXED_HP = None`).