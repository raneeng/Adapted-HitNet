# Adapted-Hitnet

Two models for automated badminton match analysis, both trained on ShuttleSet / ShuttleSet22
plus matches tagged in the BGL tagging app:

| Model | Task | Input | Code |
|---|---|---|---|
| **HitNet** | Hit detection: for every frame, no hit / bottom-player hit / top-player hit | windows of shuttle, pose and court data | `train_hitnet.py` |
| **Shot-type classifier** | For each detected hit, which of 8 rally-shot types it is | per-shot features from the same shuttle, pose and court data | `shot_lgbm.py` |

The HitNet section comes first, then the [shot-type classifier](#shot-type-classifier).
Model history and known data issues: [`HITNET_MODEL_NOTES.md`](hitnet_archive/HITNET_MODEL_NOTES.md).

---

## HitNet Training Pipeline

Trains a badminton hit-detection model (no-hit [class 0] / bottom-player-hit [class 1] / top-player-hit [class 2]).

Two stages: **1) data preparation** builds per-rally training inputs, then **2) training**
(`train_hitnet.py`) consumes those inputs to train and evaluate the model. Training data can come
from two sources, both producing the same per-rally files:

| Source | Script | Labels from |
|--------|--------|-------------|
| ShuttleSet / ShuttleSet22 matches | `format_data.py` | ShuttleSet's stroke-level annotations |
| Matches tagged in the BGL tagging app | `format_tagged.py` | The app's exported shot CSV (analyst-reviewed) |

The deployed model, `hitnet_archive/hitnet_model_v2.h5`, also detects **serves**, which earlier models
could not (see [Model Performance](#model-performance)).

## Installation

The project uses **three separate environments**. PyTorch (TrackNet, YOLO) uses CUDA 13
libraries while TensorFlow 2.21 needs CUDA 12, so they can't share one environment; the
shot-type classifier gets its own to keep `lazypredict`'s pinned packages away from the others.

| Environment | Used for | Requirements |
|-------------|----------|--------------|
| `.venv` | `format_data.py`, `format_tagged.py`, `format_shots.py`, court labelling notebook | `requirements-prep.txt` |
| `.venv-train` | `train_hitnet.py`, `eval_events.py`, `hitnet_infer.py` | `requirements-train.txt` |
| `.venv-shot` | `shot_lazy_baseline.py`, `shot_lgbm.py` | `requirements-shot.txt` |

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-prep.txt

python -m venv .venv-train
.venv-train/bin/pip install -r requirements-train.txt

python -m venv .venv-shot
.venv-shot/bin/pip install -r requirements-shot.txt
```

**TensorFlow GPU fix (training env only).** TensorFlow doesn't find its pip-installed
`libcusolver` on its own and silently falls back to CPU. Append this to
`.venv-train/bin/activate` once:

```bash
# CUDA 12 libs from tensorflow[and-cuda]
_NV=$(python -c "import nvidia; print(list(nvidia.__path__)[0])" 2>/dev/null)
[ -n "$_NV" ] && export LD_LIBRARY_PATH=$(find "$_NV" -name "*.so*" -printf "%h\n" | sort -u | paste -sd:)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
unset _NV
```

Check with:

```bash
source .venv-train/bin/activate
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

For pm2, set the same `LD_LIBRARY_PATH` in the training ecosystem file's `env`, since pm2
does not run the activate script.

> **Always train HitNet from `.venv-train`.** Started from `.venv`, TensorFlow can't find CUDA
> and silently trains on CPU (~12 min per epoch instead of ~1 min).


## Directory structure

```
hitnet/
  data_prep/
    format_data.py           <- ShuttleSet -> per-rally HitNet inputs
    format_tagged.py         <- tagging-app export -> the same per-rally HitNet inputs
    format_shots.py          <- per-rally HitNet inputs + ShuttleSet labels -> shot-type table
    shot_features.py         <- per-shot features (shuttle, pose, court); shared with BGL
    shot_mapping.py          <- ShuttleSet stroke type -> analyst shot type mapping
    court_side.py            <- A/B -> bottom/top per rally (side switches)
    pose_features.py         <- YOLOv8-Pose player extraction
    shuttle_features.py      <- TrackNetV3 + InpaintNet shuttle tracking
    label_court.ipynb        <- notebook court-corner labelling
    shuttleset/              <- ShuttleSet match.csv + per-set CSVs go here
    tagged/                  <- tagging-app shot CSVs + their LOCAL match videos go here
    weights/                 <- TrackNet_best.pt, InpaintNet_best.pt, yolov8s-pose.pt
  training/
    train_hitnet.py          <- HitNet training
    eval_events.py           <- HitNet event-level evaluation
    hitnet_infer.py          <- HitNet inference on new rallies
    shot_lazy_baseline.py    <- shot-type model shortlist (LazyClassifier) + shared split/features
    shot_lgbm.py             <- shot-type LightGBM tuning, validation and test evaluation
    data/                    <- clips starting on the serve frame (used by v1)
    data_pad15/              <- clips with 15-frame serve pre-roll (used by current models)
      videos/                <- full match videos
      rally_video/           <- sliced per-rally clips
      ball_trajectory/       <- per-rally shuttle position CSVs
      poses/                 <- per-rally pose CSVs (bottom/top -- see note below)
      shot/                  <- per-rally hit label CSVs
      shot_labels/           <- per-rally shot-type tables + all_shots.csv
      court/                 <- one .out calibration file per match
    tuner/                   <- Keras Tuner search results (delete to force a fresh search)
    hitnet_model.h5          <- default output of train_hitnet.py (latest run)
    shot_lgbm.joblib         <- current shot-type classifier
  hitnet_archive/
    HITNET_MODEL_NOTES.md    <- model iterations and known data issues
    hitnet_model_v2.h5       <- deployed HitNet model (serves detectable)
    hitnet_model_v2-1.h5     <- v2 retrained on cleaned data (leak-free baseline)
    hitnet_model_v2-1-search.h5 <- same data, new hyperparameter search (not used)
    hitnet_model_v1.h5       <- full dataset, serves not detectable
    hitnet_model_v0.h5       <- original model
  requirements-prep.txt / requirements-train.txt / requirements-shot.txt
```

`train_hitnet.py` reads from the folder set in `INPUT_DATA_DIR` (currently `data_pad15/`).


## Where to run this

**An NVIDIA GPU with CUDA is required in practice.** Data prep runs three GPU models per rally
(TrackNet, InpaintNet, YOLOv8-Pose), and training runs a multi-layer bidirectional GRU over
hundreds of thousands of windows. Both technically run on CPU, but far too slowly to be usable.
(The shot-type classifier is light and trains in seconds on CPU.)

- `train_hitnet.py` enables TensorFlow memory growth, so it only takes the GPU memory it needs and
  can share the GPU with data prep or the webapps. Expect slower jobs when they overlap.
- Both stages take hours for a full dataset. Run them as a background process (`pm2`, `tmux`,
  `nohup`) so they survive a closed terminal.


### Getting the training data
The per-rally data isn't stored in the repo. Download `data_pad15.zip` (and `data.zip`, only
needed for v1) from the [`data-2026-11-19` release](../../releases/tag/data-2026-11-19) and
unzip each into its folder: `training/data_pad15/` and `training/data/`.
`data_prep/shuttleset/shuttleset.zip` holds the ShuttleSet CSVs; unzip it in place.

---

## Stage 1 — Data preparation

Both scripts produce exactly the same outputs, so ShuttleSet and tagged matches can be mixed
freely in training and evaluation.

Each rally that completes produces exactly 4 files: `{stem}_ball_predicted.csv`,
`{stem}_player_bottom.csv`, `{stem}_player_top.csv`, `{stem}_hit.csv` — a rally is
only considered done once all 4 exist, which is what `--force`/skip-logic checks.

**Serve pre-roll.** Rally clips start **15 frames before the serve** (`data_pad15/`), so the serve
hit falls inside a window's prediction region and can be detected. Clips in the original `data/`
folder start exactly on the serve frame, so their serves are undetectable.

> **Pose file names are swapped (historical).** `_player_top.csv` holds the **near** player
> (bottom of the image, hitter label 1) and `_player_bottom.csv` the **far** player (label 2).
> Don't rename them: every model was trained with this order, and HitNet only sees which
> position each pose fills in its input. **BGL must put the near player's pose (P1) in the same
> position as `_player_top.csv` fills in training**, or hitter attribution flips.
> `shot_features.py` handles the mapping via `POSE_FILE_FOR_LABEL`. Details:
> [`HITNET_MODEL_NOTES.md`](hitnet_archive/HITNET_MODEL_NOTES.md), known data issue 3.

### Option A — ShuttleSet matches (`format_data.py`)

Downloads each match's video, slices it into per-rally clips, runs shuttle tracking
(TrackNet + InpaintNet) and pose estimation (YOLOv8-Pose), and derives ground-truth
hit labels directly from ShuttleSet's own stroke-level annotations (not inferred —
see [Hit labels](#hit-labels) below).

#### 1. Court calibration (one-time, per match)

Each match needs a `.out` file at `training/<data dir>/court/{match_id}.out`
containing the **doubles (outer) court corners**, one per line as `x;y` pixel
coordinates, in this exact order:

```
A -------- D      1. Top-left     (A)
|          |      2. Top-right    (D)
|          |      3. Bottom-left  (G)
G -------- X      4. Bottom-right (X)
```

To create this: run `format_data.py` once for the match first — it downloads the
video and automatically saves a calibration reference frame
(`{match_id}_calibration_reference.png`, taken at the first hit of set 1). Then either:

- **Interactive labeling** (needs a GUI-capable display / X11 forwarding):
  ```bash
  python format_data.py --id <match_id> --label_court
  ```
- **Notebook-based labeling** (`label_court.ipynb`):
  Input `<match_id>` and run the cell. A widget displaying `{match_id}_calibration_reference.png`
  will appear. Left-click to select the 4 corners in ADGX order; right-click to undo the last selection.
  ```python
  %matplotlib widget
  from label_court_notebook import label_court_corners_notebook
  picker = label_court_corners_notebook("<match_id>")
  # click the 4 corners in the figure, in A/D/G/X order
  ```
  When done selecting the 4 corners, run the next cell:
  ```python
  picker.save()
  ```
- **Manual labeling**: open `{match_id}_calibration_reference.png` in a photo editing app
  (*e.g. Microsoft Paint*), create a `{match_id}.out` file and type the court corner coordinates
  in the required format.

One `.out` file covers the *entire match* — this assumes the main camera doesn't move within a
match, so it's shared across every set and rally, not re-labeled per set or per rally.

> **Label on the same resolution as the video.** Corners are pixel coordinates, so a `.out`
> labelled on a 1080p frame is wrong for a 720p video: the far player then falls "off court" and
> their pose CSV comes out all zeros. This happened to 13 matches (fixed by scaling their corners
> by 2/3; see `hitnet_archive/HITNET_MODEL_NOTES.md`).

#### 2. Run data prep

```bash
cd data_prep
python format_data.py --id <match_id> [<match_id_2> ...]
```

Supports multiple match ids in one call, processed sequentially — a failure on one
match (missing calibration, failed download) is logged and skipped, not fatal to
the rest of the batch.

| Flag | Purpose |
|---|---|
| `--overwrite_video` | Re-download a match's video even if already present (default: skip already-downloaded matches) |
| `--max_height N` | Cap downloaded video resolution (default: 720) |
| `--force` | Reprocess every rally even if its output files already exist (default: skip already-completed rallies) |

#### Hit labels

`_hit.csv` has a single `hit` column, one row per frame, values `0` (no hit),
`1` (bottom/near player hit), `2` (top/far player hit), which correctly accounts for
badminton's side-switching rules (players swap ends each set, plus a mid-set
switch in a deciding third set once either player's score first reaches 11).
This is *ground truth*, not inferred from pose proximity the way some reference
implementations of this idea do. Labels are derived directly from ShuttleSet's own
`player` column via `court_side.py`. ShuttleSet convention: player `A` is the match winner
(entire match, not set), player `B` is the match loser; `downcourt = 0` in `match.csv` means
A starts set 1 on the bottom half.

### Option B — Tagging-app matches (`format_tagged.py`)

The tagging-app counterpart to `format_data.py`. It converts matches tagged in the BGL tagging
app — the app's exported **shot CSV** plus the **local match video**, both placed in the
`tagged/` folder — into the same four per-rally HitNet inputs.

```bash
cd data_prep
python format_tagged.py --help    # see current arguments
```

- **Match ids** are prefixed with `tag-` (e.g. `tag-1`, `tag-YEO-AN`), so tagged matches never
  collide with ShuttleSet ids and can be assigned to train / validation / test like any other match.
- **Labels come from the analysts' reviewed tags**, so they are ground truth for the footage the
  model is actually deployed on. This makes tagged matches the most honest test data.
- **Rally boundaries** come from the app's approved rallies, so no ShuttleSet stroke CSVs are needed.

### Extending the dataset beyond ShuttleSet

The tagging app is now the main route for new labelled data: analysts review the model's
predictions in the app, correct them, and export the shot CSV, which `format_tagged.py` turns
into training data. This gives human-verified labels on our own footage without labelling from
scratch.

> **Don't train on unreviewed predictions.** Given the known top-player weakness (see below),
> using `hitnet_infer.py` output as labels without analyst review risks reinforcing that same
> weakness in any model retrained on it.

---

## Stage 2 — Training (`train_hitnet.py`)

Consumes Stage 1's output directly — no separate conversion step.

```bash
cd training
source ../.venv-train/bin/activate
python train_hitnet.py
```

### Before running

Edit the settings at the top of the file:

| Setting | Purpose |
|---------|---------|
| `INPUT_DATA_DIR` | Which data folder to train on (currently `data_pad15/`). |
| `VAL_MATCH_IDS`, `TEST_MATCH_IDS` | Matches held out for validation and testing, as **strings** (e.g. `"31"`, `"tag-1"`). Every other match trains. A match can't be in both. |
| `FIXED_HP` | Hyperparameters to train with directly. Set to `None` to run the Bayesian search instead. |

- **Changing the search space** in `build_model()`, or the data, makes an existing `tuner/`
  folder stale — delete it before rerunning the search.
- **GPU/XLA setup**: if you hit `libdevice not found` errors, set
  `XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/nvidia/cuda_nvcc` (find the right
  path with `find / -name libdevice.10.bc`) before running, or add the missing
  component with `pip install nvidia-cuda-nvcc-cuXX` (matching your CUDA major
  version — check via `tf.sysconfig.get_build_info()['cuda_version']`).

### What it does, in order

1. Discovers every rally in `INPUT_DATA_DIR`, and splits into train / validation / test **by
   match id** (never by individual rally — avoids leaking correlated data between sets).
2. Builds windowed features (12-frame sliding windows, 78 features per frame:
   shuttle x/y + both players' 34 pose values + 8 flattened court corner values),
   scaled to `[1, 2]` with `0` reserved as the "undetected" sentinel.
3. Rebalances the *training* set per rally to roughly equal hit / no-hit counts
   (validation and test keep their natural distribution).
4. If `FIXED_HP` is `None`: runs a fast Bayesian hyperparameter search (Keras Tuner, 30 trials,
   short epochs, validated on a 100k-window subsample) over learning rate, GRU layers, units,
   L2 and dropout, with live data augmentation (reflection, frame dropout/corruption, small
   rotation/shear "camera jiggle").
5. Trains the chosen hyperparameters in full, with early stopping and learning-rate reduction on
   validation macro-F1.
6. Evaluates on **validation** (used for selection, so optimistic) and **test** (never used for
   any decision — the number to report), printing per-class precision/recall/F1 and a confusion
   matrix rather than just accuracy.
7. Saves the model to `hitnet_model.h5`.

Event-level metrics (did each real hit get detected, by the right player, within ±T frames?)
are computed separately with `eval_events.py`:

```bash
python eval_events.py --model ../hitnet_archive/hitnet_model_v2.h5
```

### Known limitation

Top-player (class 2) recall is consistently weaker than bottom-player recall. The leading
hypothesis is weaker YOLO pose confidence for the farther, smaller player in frame, plus shuttle
movements appearing smaller further from the camera.

## Inference on a new rally (no ground truth available)

Use `hitnet_infer.py`. It needs only the ball trajectory, both players' pose CSVs and the match's
court file, and writes predictions to `{stem}_hit_predicted.csv`.

```bash
python hitnet_infer.py \
    --model hitnet_archive/hitnet_model_v2.h5 \
    --stem <match_id>_set<N>_<rally> \
    --data_dir training/data_pad15 \
    --temperature <calibrated temperature>
```

> **Serves:** the input must include the pre-roll frames before the serve, or the serve falls
> outside every window's prediction region.

> **Output format:** a single real hit usually appears as a *short run of consecutive frames*
> predicting the same class, because per-frame predictions are averaged across overlapping
> windows. If you need one frame per hit (e.g. counting shots, timing analysis), collapse each
> run into one representative frame (e.g. its peak-confidence frame) — don't treat every predicted
> frame as a separate hit.


## Debugging tools

Visualization tools for spot-checking specific rallies, each overlaying its subject onto
the rally video so problems are visible directly rather than inferred from numbers alone.

| Tool | Checks | Usage |
|---|---|---|
| `render_tracknet_check.py` | Shuttle tracking quality — draws the predicted shuttle position with a fading motion trail per frame, and flags frames where the shuttle wasn't detected at all | `python render_tracknet_check.py --video <clip.mp4> --ball <_ball_predicted.csv>` |
| `pose_quality_check.py` | Pose tracking quality — `--stats` aggregates per-keypoint detection rates across the dataset (bottom vs. top player), `--overlay` draws both players' skeletons on one rally's video | `python pose_quality_check.py --stats --poses_dir data/poses` or `--overlay --video ... --pose_bottom ... --pose_top ...` |
| `render_shot_labels.py` | Predicted/ground-truth hit labels — overlays shot markers and player attribution on the video, to confirm predicted hits line up with real contact moments | see the script's own `--help` |


## Model Performance

All results are **event-level** (`eval_events.py`): for each real hit, did the model report a
hit by the right player within ±T frames? **±3 frames is the realistic tolerance**, since
ShuttleSet's own frame annotations are only accurate to about ±1–2 frames.

All models below are scored on the **same cleaned test set**: 9 matches (ShuttleSet `5, 16, 24,
38, 58, 69, 80, 91` + `tag-3`), 783 rallies, 7,682 hits, serves included.

| Model | Serves | Macro F1 | Detection P / R | Bottom F1 | Top F1 | Hitter acc. |
|---|:---:|---:|---|---:|---:|---:|
| v0 `hitnet_model_v0.h5` (Aug 2026) | No | 0.895 | 0.912 / 0.898 | 0.925 | 0.865 | 98.8% |
| v1 `hitnet_model_v1.h5` (Oct 2026) | No | 0.909 | 0.948 / 0.886 | 0.925 | 0.894 | 99.3% |
| **v2 `hitnet_model_v2.h5`** (Nov 2026, **deployed**) | **Yes** | **0.949** | **0.946 / 0.966** | **0.963** | **0.935** | 99.1% |
| v2.1 `hitnet_model_v2-1.h5` (Nov 2026) | **Yes** | 0.940 | 0.935 / 0.948 | 0.955 | 0.924 | **99.8%** |

Versioning: a major version adds a capability (v2 = serve detection); a minor version retrains
the same setup (v2.1 = v2 on cleaned data).

- **v2** is a 4-layer bidirectional GRU, 256 units, no dropout, L2 4.7e-7, initial learning
  rate 5.6e-4, trained on `data_pad15/` (15-frame serve pre-roll). The main gain over v1 is
  end-to-end recall (0.966 vs 0.886), because serves (~10% of hits) are now detectable.
- **v2.1** has the same settings, retrained on the cleaned data (duplicate matches removed,
  far-player poses fixed in 13 matches). It is the only model with no leakage into the test set.
  Excluding match 38 (whose duplicate was in v2's training set), v2 still scores slightly higher
  (0.950 vs 0.945), mostly from better recall, while v2.1 attributes the hitter more accurately
  (99.8% vs 99.1%). **v2 stays deployed**; v2.1 is the leak-free baseline for future changes.
- **The v0–v2 numbers are somewhat optimistic.** v0 trained on test matches 5, 24 and 38; v1 and
  v2 were trained on data containing duplicate matches (test match 38's duplicate was in their
  training set).

Full details per model, the data issues and their fixes: [`HITNET_MODEL_NOTES.md`](hitnet_archive/HITNET_MODEL_NOTES.md).

---

## Shot-type classifier

Classifies each hit into one of **8 rally-shot types** (the analysts' list, with forehand /
backhand removed): `net`, `net lift`, `net kill`, `block`, `drive`, `clear`, `smash`,
`dropshot`. **Serves** are handled separately (the first shot of every rally is a serve; only
short vs long has to be decided). Racket face (forehand / backhand) is planned as a separate
model later.

### Pipeline

```
data_pad15/ (ball, poses, court, clips)  +  ShuttleSet per-set CSVs
        │
        ▼
format_shots.py  ──uses──>  shot_features.py, shot_mapping.py, court_side.py
        │
        ▼
data_pad15/shot_labels/{stem}_shots.csv, all_shots.csv     (one row per stroke)
        │
        ▼
shot_lazy_baseline.py   (shortlist of models)  ->  shot_lgbm.py   (tune, validate, test)
        │
        ▼
shot_lgbm.joblib
```

### 1. Build the shot table (`format_shots.py`, in `.venv`)

```bash
cd data_prep
python format_shots.py --all          # every ShuttleSet match with a court file
python format_shots.py --id 29 30     # specific matches
```

From ShuttleSet it takes **only** the hit frame (rebased to the rally clip, same indexing as
`_hit.csv`), the hitter (A/B → bottom/top via `court_side.py`) and the shot type label. Every
feature is computed by `shot_features.py` from the TrackNet / YOLO / court files — the same
inputs BGL has at inference — so no annotator-only information (ShuttleSet positions, heights,
areas) reaches the model. Rallies whose four HitNet files don't exist are skipped; tagged
matches (`tag-*`) are skipped by `--all`.

Each row also has `keep` / `drop_reason`. Strokes are dropped for: ShuttleSet type `未知球種`
(unknown), backhand smashes (no analyst label), or a missing frame.

### 2. Label mapping (`shot_mapping.py`)

ShuttleSet types are mapped to the analysts' labels; the model trains on the label without
racket face (`base_type`).

| Analyst label | ShuttleSet types |
|---|---|
| FH / BH net | 放小球 net shot, 勾球 cross-court net shot |
| Block | 擋小球 return net, 防守回挑 defensive return lob |
| FH smash | 殺球 smash, 點扣 wrist smash (backhand versions dropped) |
| FH / BH net lift | 挑球 lob, **推球 push** |
| FH / BH clear | 長球 clear |
| FH / BH drive | 平球 drive, 小平球 driven flight, 防守回抽 defensive return drive |
| FH / BH dropshot | 切球 drop, 過度切球 passive drop, 後場抽平球 back-court drive |
| FH / BH net kill | 撲球 rush |
| FH / BH low serve | 發短球 short service |
| FH high serve / BH flick serve | 發長球 long service |
| *(dropped)* | 未知球種 unknown |


### 3. Features (`shot_features.py`)

Computed per shot from the shuttle trajectory, both players' poses and the court homography.
Positions are expressed **from the hitter's point of view** (lateral > 0 = hitter's right,
depth = metres behind the net), so a shot gets the same features from either end of the court.
Which half the hitter is on (`hitter_near`) is measured from their YOLO feet, not assumed.

| Group | Features |
|---|---|
| Positions (m) | `hitter_lat`, `hitter_depth`, `opp_lat`, `opp_depth`, `land_lat`, `land_depth`, `shot_distance`, `lat_change`, `hitter_move_speed`, `hitter_near` |
| Timing | `flight_s` (to next shot), `prev_gap_s` (since previous shot), `is_last` |
| Shuttle (image, court-normalised) | `contact_above_head`, `apex_above_start`, `apex_above_end`, `mean_speed`, `img_dx`, `img_dy_fwd`, `launch_vx`, `launch_vy_up`, `launch_speed`, `shuttle_detect_rate` |
| Hitter pose at contact | `wrist_above_head`, `elbow_angle`, `shoulder_angle`, `wrist_lat_offset`, `knee_angle_min`, `knee_angle_max`, `stance_width`, `torso_lean`, `jump`, `pose_detect_rate` |
| Rally context (previous stroke) | `prev_launch_speed`, `prev_launch_vy_up`, `prev_contact_above_head`, `prev_hitter_depth` |

Landing position is the receiver's feet at their next contact (the shuttle's 3D landing point
isn't observable); for the last shot of a rally it's the shuttle's last tracked point.

### 4. Train and evaluate (in `.venv-shot`)

```bash
cd training
python shot_lazy_baseline.py   # quick shortlist of ~12 classifiers on validation -> lazy_results.csv
python shot_lgbm.py            # tune LightGBM, validation report, saves shot_lgbm.joblib
python shot_lgbm.py --test     # final evaluation on TEST -- run once, after tuning
```

The split matches HitNet's (by match, duplicates removed): validation = 21 ShuttleSet matches,
test = `5, 16, 24, 38, 58, 69, 80, 91`. Serves are excluded (`keep & ~is_serve`).

### Results

LightGBM was the best of the shortlist (with XGBoost and HistGradientBoosting essentially tied).
Tuned settings: `num_leaves` 63, `min_child_samples` 50, `class_weight="balanced"`, learning
rate 0.05, 173 trees.

| Split | Shots | Accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|
| Train | 44,606 | | | |
| Validation (21 matches) | 15,062 | 0.894 | 0.852 | 0.844 |
| **Test (8 matches)** | **6,479** | **0.893** | **0.845** | **0.841** |

Per-class F1 on test:

| net lift | clear | net | smash | dropshot | block | net kill | drive |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.951 | 0.939 | 0.921 | 0.899 | 0.875 | 0.816 | 0.756 | 0.570 |

Main confusions (test): **net ↔ block** (6.6% / 8.0%), **dropshot ↔ smash** (9.7% / 5.7%),
**drive → block** (22% of drives). The most important features are flight time, time since the
previous shot, hitter and landing depth, and the shuttle's launch angle.

### Serves

Not part of the classifier. Short vs long can be decided from `land_depth` (and `flight_s`)
of the first shot; combined with racket face this gives the analysts' serve labels
(low serve, high serve, flick serve).

---

## Dataset

This project is trained and evaluated using the **ShuttleSet** and **ShuttleSet22** datasets, large-scale badminton match datasets with fine-grained, stroke-level annotations, plus matches tagged in the BGL tagging app.

ShuttleSet was annotated by badminton domain experts using the efficient **S2-labeling** tool and the **BLSR** annotation format. The dataset provides stroke-by-stroke ground truth annotations for **44 international badminton matches** played between **2018 and 2021**, featuring **27 elite players** (16 men's singles and 11 women's singles). In total, ShuttleSet contains:

- **104 sets**
- **3,685 rallies**
- **36,492 annotated strokes**

ShuttleSet22 is an extension from ShuttleSet. ShuttleSet22 contains **33,612 strokes (3,992 rallies)** with 35 top-ranking men's singles and women's singles players.

> **Duplicates excluded.** ShuttleSet22 ids **45–51 and 96** are re-uploads of ShuttleSet matches
> 23, 38, 39, 41, 42, 43, 44 and 24 (same YouTube video). They are excluded from all training and
> evaluation; `match.csv` is left unchanged.

---

## Acknowledgements

This project builds upon and makes use of several excellent open-source resources from the badminton computer vision community.

### MonoTrack

Parts of the shuttle tracking pipeline are adapted from **MonoTrack**, an open-source implementation for shuttle trajectory reconstruction from monocular badminton videos.

**Reference**

> Liu, P., & Wang, J.-H. **MonoTrack: Shuttle Trajectory Reconstruction from Monocular Badminton Video.** CVPR Workshop, 2022.  
> Paper: https://arxiv.org/abs/2204.01899  
> Code: https://github.com/jhwang7628/monotrack

### ShuttleSet

This work is trained and evaluated using the **ShuttleSet** dataset, whose high-quality expert annotations make large-scale badminton stroke recognition and rally analysis possible.

**Reference**

> Jiang, Y., Tian, Y., Wang, Y., *et al.* **ShuttleSet: A Large-scale Badminton Dataset for Fine-grained Stroke Recognition.** arXiv:2306.04948, 2023.  
> https://arxiv.org/pdf/2306.04948

I gratefully acknowledge the authors and contributors of both **MonoTrack** and **ShuttleSet** for making their code, datasets, and research publicly available, enabling further advances in badminton computer vision research.