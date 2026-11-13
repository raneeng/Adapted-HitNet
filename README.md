# Adapted-Hitnet

## HitNet Training Pipeline

Trains a badminton hit-detection model (no-hit [class 0] / bottom-player-hit [class 1] / top-player-hit [class 2]).

Two stages: **1) data preparation** builds per-rally training inputs, then **2) training**
(`train_hitnet.py`) consumes those inputs to train and evaluate the model. Training data can come
from two sources, both producing the same per-rally files:

| Source | Script | Labels from |
|--------|--------|-------------|
| ShuttleSet / ShuttleSet22 matches | `format_data.py` | ShuttleSet's stroke-level annotations |
| Matches tagged in the BGL tagging app | `format_tagged.py` | The app's exported shot CSV (analyst-reviewed) |

The current model, `hitnet_model_with_serve.h5`, also detects **serves**, which earlier models
could not (see [Model Performance](#model-performance)).

## Installation

Data prep and training need **separate environments**: PyTorch (TrackNet, YOLO) uses CUDA 13
libraries, while TensorFlow 2.21 needs CUDA 12, and the two conflict in one environment.

| Environment | Used for | Requirements |
|-------------|----------|--------------|
| `.venv` | `format_data.py`, `format_tagged.py`, court labelling notebook | `requirements-prep.txt` |
| `.venv-train` | `train_hitnet.py`, `eval_events.py`, `hitnet_infer.py` | `requirements-train.txt` |

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-prep.txt

python -m venv .venv-train
.venv-train/bin/pip install -r requirements-train.txt
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


## Directory structure

```
hitnet/
  data_prep/
    format_data.py           <- ShuttleSet -> per-rally HitNet inputs
    format_tagged.py         <- tagging-app export -> the same per-rally HitNet inputs
    shuttleset/              <- ShuttleSet match.csv + per-set CSVs go here
    tagged/                  <- tagging-app shot CSVs + their LOCAL match videos go here
    weights/                 <- TrackNet_best.pt, InpaintNet_best.pt, yolov8s-pose.pt
  training/
    train_hitnet.py
    court_side.py
    eval_events.py
    hitnet_infer.py
    data/                    <- original clips (start on the serve frame)
    data_pad15/              <- clips with 15-frame serve pre-roll (used by the current model)
      videos/                <- full match videos
      rally_video/           <- sliced per-rally clips
      ball_trajectory/       <- per-rally shuttle position CSVs
      poses/                 <- per-rally pose CSVs (bottom/top)
      shot/                  <- per-rally hit label CSVs
      court/                 <- one .out calibration file per match
    tuner/                   <- Keras Tuner search results (delete to force a fresh search)
    hitnet_model.h5          <- default output of train_hitnet.py
    hitnet_model_with_serve.h5  <- current model (serves detectable)
    hitnet_model_19oct.h5    <- previous model (serves not detectable)
```

`train_hitnet.py` reads from the folder set in `INPUT_DATA_DIR` (currently `data_pad15/`).

## Where to run this

**An NVIDIA GPU with CUDA is required in practice.** Data prep runs three GPU models per rally
(TrackNet, InpaintNet, YOLOv8-Pose), and training runs a multi-layer bidirectional GRU over
hundreds of thousands of windows. Both technically run on CPU, but far too slowly to be usable.

- `train_hitnet.py` enables TensorFlow memory growth, so it only takes the GPU memory it needs and
  can share the GPU with data prep or the webapps. Expect slower jobs when they overlap.
- Both stages take hours for a full dataset. Run them as a background process (`pm2`, `tmux`,
  `nohup`) so they survive a closed terminal.

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
(entire match, not set), player `B` is the match loser.

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
python train_hitnet.py
```

### Before running

Edit the settings at the top of the file:

| Setting | Purpose |
|---------|---------|
| `INPUT_DATA_DIR` | Which data folder to train on (currently `data_pad15/`). |
| `VAL_MATCH_IDS`, `TEST_MATCH_IDS` | Matches held out for validation and testing, as **strings** (e.g. `"31"`, `"tag-1"`). Every other match trains. A match can't be in both. |
| `FIXED_HP` | Hyperparameters to train with directly. Set to `None` to run the Bayesian search instead. |

- **Changing the search space** in `build_model()` makes an existing `tuner/` folder
  incompatible — delete it before rerunning.
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
are computed separately with `eval_events.py`.

### Known limitation

Top-player (class 2) recall is consistently weaker than bottom-player recall. The leading
hypothesis is weaker YOLO pose confidence for the farther, smaller player in frame, plus shuttle
movements appearing smaller further from the camera.

## Inference on a new rally (no ground truth available)

Use `hitnet_infer.py`. It needs only the ball trajectory, both players' pose CSVs and the match's
court file, and writes predictions to `{stem}_hit_predicted.csv`.

```bash
python hitnet_infer.py \
    --model hitnet_model_with_serve.h5 \
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

Three visualization tools for spot-checking specific rallies, each overlaying its subject onto
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

### Current model: `hitnet_model_with_serve.h5`

- **Detects serves.** Trained on clips with a 15-frame serve pre-roll, so every hit is
  predictable — previously ~10.5% of hits (the serves) could never be detected.
- **Architecture:** 4-layer bidirectional GRU, 256 units, no dropout, L2 4.7e-7, initial
  learning rate 5.6e-4 (settled by two Bayesian searches, Oct 2026).
- **Split by match:** validation = 23 ShuttleSet matches + `tag-1`; test = 9 ShuttleSet
  matches + `tag-3`; all other matches train.

**Validation results** (24 matches, 1,896 rallies, 18,914 hits, serves included):

| Tolerance | Bottom P / R | Top P / R | Macro F1 | Detection recall | Hitter accuracy |
|----------:|:------------:|:---------:|---------:|-----------------:|----------------:|
| ±1 | 0.767 / 0.800 | 0.720 / 0.748 | 0.758 | 0.787 | 98.1% |
| ±2 | 0.886 / 0.923 | 0.852 / 0.885 | 0.886 | 0.919 | 98.1% |
| **±3** | **0.914 / 0.952** | **0.889 / 0.924** | **0.920** | **0.953** | **98.1%** |

- **Timing:** 81% of detected hits are within ±1 frame of the label, 95% within ±2.
- **Weakest matches:** 21 and 29 (macro F1 0.77–0.79 at ±3), held down by low precision for
  both players. The other 22 matches score 0.86–0.98.

> **Test-set results pending.** Validation matches were used for model selection, so these
> numbers are likely slightly optimistic. Run `eval_events.py` on the test matches for the
> number to report.

### Model history

Event-level, ±3 frames.

| | Aug 2026 | Oct 2026 (`hitnet_model_19oct.h5`) | **With serve (current)** |
|---|---:|---:|---:|
| Evaluated on | Test (9 matches) | Test (9 matches) | Validation (24 matches) |
| Serves detectable | No | No | **Yes** |
| Macro F1 (serves excluded from scoring) | 0.940 | 0.960 | — |
| Macro F1 (serves included) | — | — | 0.920 |
| End-to-end hit recall (serves included) | — | ~87% | **95.3%** |
| Hitter accuracy | 99.6% | 99.9% | 98.1% |
| Worst match (macro F1) | 0.905 | 0.941 | 0.774 |

- **The main gain is end-to-end recall:** the Oct model scored well on the hits it *could* see,
  but missed every serve, so only ~87% of all hits were found. The current model finds 95.3%.
- **The columns aren't directly comparable yet.** The current model is scored on validation with
  serves included, the older ones on test with serves excluded. Its lower hitter accuracy and
  worst-match score should be rechecked on the test set before drawing conclusions.

---

## Dataset

This project is trained and evaluated using the **ShuttleSet** and **ShuttleSet22** datasets, large-scale badminton match datasets with fine-grained, stroke-level annotations, plus matches tagged in the BGL tagging app.

ShuttleSet was annotated by badminton domain experts using the efficient **S2-labeling** tool and the **BLSR** annotation format. The dataset provides stroke-by-stroke ground truth annotations for **44 international badminton matches** played between **2018 and 2021**, featuring **27 elite players** (16 men's singles and 11 women's singles). In total, ShuttleSet contains:

- **104 sets**
- **3,685 rallies**
- **36,492 annotated strokes**

ShuttleSet22 is an extension from ShuttleSet. ShuttleSet22 contains **33,612 strokes (3,992 rallies)** with 35 top-ranking men's singles and women's singles players.

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

We gratefully acknowledge the authors and contributors of both **MonoTrack** and **ShuttleSet** for making their code, datasets, and research publicly available, enabling further advances in badminton computer vision research.