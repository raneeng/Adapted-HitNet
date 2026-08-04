# Adapted-Hitnet

## HitNet Training Pipeline

Trains a badminton hit-detection model (no-hit [class 0] / bottom-player-hit [class 1] / top-player-hit [class 2])
on ShuttleSet match data. 

Two stages: **1) data preparation** (`format_data.py`) builds
per-rally training inputs from raw ShuttleSet data, then **2) training** (`train_hitnet.py`) consumes those inputs to train and evaluate the model.

## Directory structure

```
hitnet/
  data_prep/
    format_data.py
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
(`{match_id}_calibration_reference.png`, taken at the first hit of set 1) even
before the `.out` file exists. Use that image to identify the four corners, then
either:

- **Interactive labeling** (needs a GUI-capable display / X11 forwarding):
  ```bash
  python format_data.py --id <match_id> --label_court
  ```
- **Notebook-based labeling** (`label_court.ipynb`):
  
  Input `<match_id>` and run cell. A widget displaying `{match_id}_calibration_reference.png` will appear, left-click to select the 4 corners in ADGX order. Right-click to undo the last selection.
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

One `.out` file covers the *entire match* — the camera doesn't move within a match,
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

### What the hit labels actually mean

`_hit.csv` has a single `hit` column, one row per frame, values `0` (no hit),
`1` (bottom/near player hit), `2` (top/far player hit), which correctly accounts for
badminton's side-switching rules (players swap ends each set, plus a mid-set
switch in a deciding third set once either player's score first reaches 11).
This is *ground truth*, not inferred from pose proximity the way some reference
implementations of this idea do. These shot labels are derived directly from
ShuttleSet's own `player` column via `court_side.py`. Shuttleset convention: player `A` represents match winner (entire match, not set), while player `B` represents match loser.

### Current progress with ShuttleSet dataset formatting

All 4 training inputs: 
`{stem}_ball_predicted.csv`, `{stem}_player_bottom.csv`, `{stem}_player_top.csv`, `{stem}_hit.csv`
have been generated for *all sets* of the following match ids (see match.csv for match information):

- 1 to 7 (inclusive)
- 21 to 26 (inclusive)
- 28 to 44 (inclusive)

>**updated as of 3 Aug 2026**

*Note: match 27's match video is unavailable on youtube*

### Future data preparation beyond ShuttleSet data
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

### Known limitation, worth checking before trusting results

Top-player (class 2) recall has historically been notably weaker than
bottom-player recall in this pipeline — the leading hypothesis is weaker YOLO
pose-tracking confidence for the farther/smaller player in frame, as well as
shuttle movements being scaled down due to shuttle being further from camera.

## Inference on a new rally (no ground truth available)

For rallies you don't have ShuttleSet labels for (e.g. new matches), use
`hitnet_infer.py` instead of `format_data.py`'s label-derivation step — it needs
only the ball trajectory, both players' pose CSVs, and the match's court file. It outputs hit predictions in `{stem}_hit_predicted.csv`

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
 
`render_tracknet_check.py` and `pose_quality_check.py` are the most directly
useful for the known top-player recall issue above — if you suspect a specific
rally's poor prediction traces back to bad input data rather than the model
itself, these are the fastest way to check visually before assuming either way.
