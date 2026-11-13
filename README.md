# Adapted-Hitnet

## HitNet Training Pipeline

Trains a badminton hit-detection model (no-hit [class 0] / bottom-player-hit [class 1] / top-player-hit [class 2])
on ShuttleSet match data. 

Two stages: **1) data preparation** (`format_data.py`) builds
per-rally training inputs from raw ShuttleSet data, then **2) training** (`train_hitnet.py`) consumes those inputs 
to train and evaluate the model.

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
    format_data.py
    format_tagged.py
    shuttleset/              <- ShuttleSet match.csv + per-set CSVs go here
    weights/                 <- TrackNet_best.pt, InpaintNet_best.pt, yolov8s-pose.pt
  training/
    train_hitnet.py
    court_side.py
    data/
      videos/                <- downloaded full match videos
      rally_video/            <- sliced per-rally clips
      ball_trajectory/        <- per-rally shuttle position CSVs
      poses/                   <- per-rally pose CSVs (bottom/top)
      shot/                    <- per-rally hit label CSVs
      court/                   <- one .out calibration file per match
    tuner/                    <- Keras Tuner search results (safe to delete to force a fresh search)
    hitnet_model.h5           <- final trained model
```

## Where to run this
 
Development for this project was done on **JupyterHub**, not local hardware due to 
memory and hardware requirements. Run long jobs as a background process (`pm2`, `tmux`, `nohup`).

---

## Stage 1 — Data preparation (`format_data.py`)

Downloads each match's video, slices it into per-rally clips, runs shuttle tracking
(TrackNet + InpaintNet) and pose estimation (YOLOv8-Pose), and derives ground-truth
hit labels directly from ShuttleSet's own stroke-level annotations (not inferred —
see `court_side.py` below).

### 1. Court calibration (one-time, per match)

Each match needs a `.out` file at `data_prep/../training/data/court/{match_id}.out`
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
  
  Input `<match_id>` and run cell. A widget displaying `{match_id}_calibration_reference.png` will appear, left-click 
  to select the 4 corners in ADGX order. Right-click to undo the last selection.
  ```python
  %matplotlib widget
  from label_court_notebook import label_court_corners_notebook
  picker = label_court_corners_notebook("<match_id>")
  # click the 4 corners in the figure, in A/D/G/X order
  ```
  When done selecting the 4 corners, run next cell:

  ``` python
  picker.save()
  ```

- **Manual labeling**:
Open `{match_id}_calibration_reference.png` in in an photo editing app (*e.g. Microsoft Paint*),
manually create a `{match_id}.out` file and type the court corner coordinates in the required format.


One `.out` file covers the *entire match* — the is on the assumption that the main camera doesn't move within a match,
so it's shared across every set and rally, not re-labeled per set or per rally.


### 2. Run data prep

```bash
cd data_prep
python format_data.py --id <match_id> [<match_id_2> ...]
```

Supports multiple match ids in one call, processed sequentially — a failure on one
match (missing calibration, failed download) is logged and skipped, not fatal to
the rest of the batch.

Useful flags:

| Flag | Purpose |
|---|---|
| `--overwrite_video` | Re-download a match's video even if already present (default: skip already-downloaded matches)|
| `--max_height N` | Cap downloaded video resolution (default: 720) |
| `--force` | Reprocess every rally even if its output files already exist (default: skip already-completed rallies) |

Each rally that completes produces exactly 4 files: `{stem}_ball_predicted.csv`,
`{stem}_player_bottom.csv`, `{stem}_player_top.csv`, `{stem}_hit.csv` — a rally is
only considered done once all 4 exist, which is what `--force`/skip-logic checks.

### Hit labels

`_hit.csv` has a single `hit` column, one row per frame, values `0` (no hit),
`1` (bottom/near player hit), `2` (top/far player hit), which correctly accounts for
badminton's side-switching rules (players swap ends each set, plus a mid-set
switch in a deciding third set once either player's score first reaches 11).
This is *ground truth*, not inferred from pose proximity the way some reference
implementations of this idea do. These shot labels are derived directly from
ShuttleSet's own `player` column via `court_side.py`. Shuttleset convention: player `A` represents match winner 
(entire match, not set), while player `B` represents match loser.

### Current progress with ShuttleSet dataset formatting

All 4 training inputs: 
`{stem}_ball_predicted.csv`, `{stem}_player_bottom.csv`, `{stem}_player_top.csv`, `{stem}_hit.csv`
have been generated for *all sets* of the following 30 match ids (see match.csv for match information):

- 1 to 7 (inclusive)
- 21 to 26 (inclusive)
- 28 to 44 (inclusive)

>**updated as of 3 Aug 2026**

>*Note: match 27's match video is unavailable on youtube*

### Future data preparation beyond ShuttleSet data
>*An extension dataset ShuttleSet22 published at IJCAI-24 Demo & IJCAI-23 IT4PSS Workshop is also released (https://github.com/wywyWang/CoachAI-Projects/tree/main/CoachAI-Challenge-IJCAI2023/ShuttleSet22)*

TrackNet shuttle tracking, and YOLOv8-Pose estimation are directly reusable for new,
non-ShuttleSet footage as-is
 
Two parts do *not* carry over automatically, and need a different approach for new
matches:
 
- **Rally boundaries.** `format_data.py` currently gets rally start/end frames
  directly from ShuttleSet's own per-set stroke CSVs (`{id}_set{n}.csv`) and 
the hitnet model is trained and predicts on single-rally data (sliced automatically by `format_data.py`).
- **Hit labels.** `court_side.py`'s ground-truth
  derivation depends entirely on ShuttleSet's own stroke-level `player`
  annotations (who hit each shot, already labeled by ShuttleSet's creators). Two options, in order of effort:
  1. **Manual labeling** — a human reviews the rally and records hit frame +
     hitter (bottom/top) directly in the same `_hit.csv` format. Most reliable,
     most labor-intensive.
  2. **Model-assisted pseudo-labeling** — run the already-trained model via
     `hitnet_infer.py` on new footage and treat its output as a starting point,
     with human spot-checking rather than treating it as ground truth outright.
     Given the known top-player recall weakness (see below), predictions used
     this way risk quietly reinforcing that same weakness in any model
     retrained on them, unless reviewed rather than accepted as-is.

---

## Stage 2 — Training (`train_hitnet.py`)

Consumes `data_prep`'s output directly — no separate conversion step.

```bash
cd training
python train_hitnet.py
```

### Before running

- **Set `VAL_MATCH_IDS`** at the top of the file — an explicit set of match ids
  held out for validation (as **strings**, e.g. `{"31"}`, not ints — this is
  compared against `stem_to_match_id()`'s output).
- **GPU/XLA setup**: if you hit `libdevice not found` errors, set
  `XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/nvidia/cuda_nvcc` (find the right
  path with `find / -name libdevice.10.bc`) before running, or add the missing
  component with `pip install nvidia-cuda-nvcc-cuXX` (matching your CUDA major
  version — check via `tf.sysconfig.get_build_info()['cuda_version']`).

### What it does, in order

1. Discovers every rally clip in `training/data/rally_video/`, splits into
   train/validation by match id (never by individual rally — avoids leaking
   correlated data between the two sets).
2. Builds windowed features (12-frame sliding windows, 78 features per frame:
   shuttle x/y + both players' 34 pose values + 8 flattened court corner values),
   scaled to `[1, 2]` with `0` reserved as the "undetected" sentinel.
3. Rebalances the *training* set to roughly equal no-hit/bottom-hit/top-hit
   counts (validation stays at its natural distribution).
4. Runs a Bayesian hyperparameter search (Keras Tuner) over GRU layer count,
   units, L2 regularization, and dropout — with data augmentation (reflection,
   frame dropout/corruption, small rotation/shear "camera jiggle") applied
   live during training.
5. Retrains the winning hyperparameters on the full training set.
6. Evaluates on the real (non-rebalanced) validation set — reports precision/
   recall/f1 **per class**, not just overall accuracy (overall accuracy on an
   imbalanced set can look deceptively good even from a model that's bad at
   actually detecting hits).
7. Runs temperature-scaling calibration on the same real validation set.
8. Saves the final model to `hitnet_model.h5`.

### Known limitation

Top-player (class 2) recall has historically been notably weaker than
bottom-player recall in this pipeline — the leading hypothesis is weaker YOLO
pose-tracking confidence for the farther/smaller player in frame, as well as
shuttle movements being scaled down due to shuttle being further from camera.

## Inference on a new rally (no ground truth available)

For rallies you don't have ShuttleSet labels for (e.g. new matches), use
`hitnet_infer.py` instead of `format_data.py`'s label-derivation step — it needs
only the ball trajectory, both players' pose CSVs, and the match's court file. It outputs hit predictions 
in `{stem}_hit_predicted.csv`

> **Output format note**: a single real hit typically registers across *multiple
> consecutive frames* in this output, not one clean frame — the model's
> per-frame predictions get smoothed across overlapping windows before writing,
> so a genuine hit event usually appears as a short run of frames all predicting
> the same class rather than a single spike. If your downstream use case needs
> one discrete frame per hit (e.g. counting shots, timing analysis), you'll need
> to collapse each consecutive same-class run into a single representative frame
> (e.g. the run's peak-confidence frame) before using this output directly —
> don't treat every predicted frame as a separate hit.
 

```bash
python hitnet_infer.py \
    --model hitnet_model.h5 \
    --stem <match_id>_set<N>_<rally> \
    --data_dir training/data \
    --temperature <value from step 7 above>
```


## Debugging tools
 
Three visualization tools for spot-checking specific rallies when results look
wrong, each overlaying its subject onto the actual rally video so problems are
visible directly rather than inferred from numbers alone.
 
| Tool | Checks | Usage |
|---|---|---|
| `render_tracknet_check.py` | Shuttle tracking quality — draws the predicted shuttle position with a fading motion trail per frame, and flags frames where the shuttle wasn't detected at all | `python render_tracknet_check.py --video <clip.mp4> --ball <_ball_predicted.csv>` |
| `pose_quality_check.py` | Pose tracking quality — two modes: `--stats` aggregates per-keypoint detection rates across your whole dataset (bottom vs. top player), `--overlay` draws both players' skeletons on one rally's video for visual inspection | `python pose_quality_check.py --stats --poses_dir data/poses` or `--overlay --video ... --pose_bottom ... --pose_top ...` |
| `render_shot_labels.py` | Predicted/ground-truth hit labels — overlays shot markers and player attribution directly on the video, for visually confirming whether detected/predicted hits actually line up with real contact moments | see the script's own `--help` for current usage |


## Model Performance

> **Last updated:** 19 Oct 2026 — model file: `hitnet_model_19oct.h5`

The current hit detection model was trained on **7,809 rally clips** from **92 ShuttleSet matches**
(ids 1–96, excluding 9, 10, 12 and 27, whose match videos are unavailable).

### Dataset Split

Rallies are split by **match ID**, so no match contributes to more than one split.

| Split | Matches | Rallies | Used for |
|-------|--------:|--------:|----------|
| Training | 60 | 5,132 | Fitting the weights |
| Validation | 23 | 1,851 | Hyperparameter search, early stopping, LR scheduling |
| Test | 9 | 826 | Final evaluation only — never used for any training decision |
| **Total** | **92** | **7,809** | |

#### Match IDs

Training matches:

```
1, 3, 4, 6, 8, 13, 14, 15, 17, 19, 20, 22, 23, 25, 26, 28,
32, 33, 34, 35, 37, 39, 41, 42, 44, 45, 46, 48, 50, 51, 52,
54, 55, 56, 59, 61, 62, 63, 64, 66, 68, 70, 72, 73, 74, 75,
77, 78, 79, 81, 82, 83, 85, 86, 87, 88, 90, 92, 94, 95
```

Validation matches:

```
2, 7, 11, 18, 21, 29, 30, 31, 36, 40, 43, 49,
53, 57, 60, 65, 67, 71, 76, 84, 89, 93, 96
```

Test matches:

```
5, 16, 24, 38, 47, 58, 69, 80, 91
```

### Model

Bidirectional GRU, selected by a 30-trial Bayesian search on validation macro-F1:

| Hyperparameter | Value |
|----------------|------:|
| GRU layers | 3 |
| GRU units | 256 |
| Dropout | 0.0 |
| L2 strength | 1.05e-5 |
| Learning rate (initial) | 5.1e-4 |
| Batch size | 256 |


### Evaluation Results
The latest HitNet model was evaluated on a held-out test set of **131,427** frames with three classes:

- **0** – No hit
- **1** – Bottom player hit
- **2** – Top player hit

### Event-level results

These measure: **for each real hit, did the model report a hit by the right player within ±T frames?**
 Window predictions are averaged per frame, each run of hit frames is collapsed into one event at its 
 peak, and events are matched one-to-one to ground truth (`eval_events.py`).

| Tolerance, T | Bottom P / R / F1 | Top P / R / F1 | Macro F1 | Detection F1 (any player) | Hitter accuracy |
|----------:|:-----------------:|:--------------:|---------:|--------------------------:|----------------:|
| ±0 | 0.449 / 0.458 / 0.454 | 0.395 / 0.399 / 0.397 | 0.425 | 0.426 | 100.0% |
| ±1 | 0.757 / 0.772 / 0.764 | 0.708 / 0.715 / 0.712 | 0.738 | 0.739 | 99.9% |
| ±2 | 0.938 / 0.956 / 0.947 | 0.886 / 0.895 / 0.891 | 0.919 | 0.920 | 99.9% |
| **±3** | **0.968 / 0.986 / 0.976** | **0.938 / 0.948 / 0.943** | **0.960** | **0.961** | **99.9%** |
| ±5 | 0.972 / 0.991 / 0.981 | 0.950 / 0.960 / 0.955 | 0.968 | 0.969 | 99.8% |

- **±2–3 frames is the realistic tolerance**: ShuttleSet's own frame annotations are only
  accurate to about ±1–2 frames.
- **Timing:** 76% of detected hits are within ±1 frame of the label and 95% within ±2.
- **Hitter attribution:** 99.9% of detected hits assigned to the right player.
- **No outlier matches:** per-match macro-F1 at ±3 ranges from 0.941 to 0.971.

> **Serves are not yet detectable.** `format_data.py` starts each rally clip on the serve
> frame, so serves fall before any window's prediction region. They make up **10.5%** of
> test hits (838 / 8,019) and are excluded from the table above. Including them,
> end-to-end recall at ±3 frames is about **87%**. Fix planned: add pre-roll frames to
> each rally clip (`START_PAD_FRAMES`) and retrain.

### Comparison with the previous (3 Aug 2026) model

Both models evaluated on the same 9 TEST matches, event-level at ±3 frames:

| Metric | Aug 2026 | **Oct 2026** |
|--------|---------:|-------------:|
| Macro F1 | 0.940 | **0.960** |
| Detection F1 | 0.943 | **0.961** |
| Bottom precision / recall | 0.938 / 0.982 | **0.968 / 0.986** |
| Top precision / recall | 0.904 / 0.938 | **0.938 / 0.948** |
| Hitter accuracy | 99.6% | **99.9%** |
| Worst performing match (macro F1) | 0.905 | **0.941** |

### Window-level classification report

Per-window metrics (12-frame windows, natural class distribution), as printed by
`train_hitnet.py`. These are much stricter than the event-level metrics: a hit predicted one
frame early or late counts as wrong at the window edges, which mainly lowers hit precision.

Test set: **205,871** windows, three classes: **0** no hit, **1** bottom-player hit, **2** top-player hit.

| Class | Precision | Recall | F1-score | Support |
|------:|----------:|-------:|---------:|--------:|
| No hit (0) | 0.970 | 0.932 | 0.950 | 162,905 |
| Bottom hit (1) | 0.788 | 0.912 | 0.845 | 21,536 |
| Top hit (2) | 0.755 | 0.861 | 0.805 | 21,430 |

| Metric | Test | Validation |
|--------|-----:|-----------:|
| Accuracy | **92.2%** | 92.9% |
| Macro F1 | **0.867** | 0.873 |
| Weighted F1 | **0.924** | 0.931 |

#### Confusion matrix (test)

| True \\ Pred | No hit | Bottom hit | Top hit |
|-------------|--------:|-----------:|--------:|
| No hit | 151,764 | 5,176 | 5,965 |
| Bottom hit | 1,879 | 19,633 | 24 |
| Top hit | 2,866 | 107 | 18,457 |

Bottom/top confusion is negligible (131 of 42,966 hit windows). The remaining weakness is the
**top (far) player**: lower recall and precision than the bottom player in every test match,
most likely because the far player is smaller in frame and their pose detections are less
reliable.

> The Aug 2026 report (macro F1 0.861) was measured on its *validation* matches, which were
> also used to select that model, so it is not directly comparable to the test numbers above.

---

## Dataset

This project is trained and evaluated using the **ShuttleSet** and **ShuttleSet22** dataset, a large-scale badminton match dataset with fine-grained, stroke-level annotations.

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
