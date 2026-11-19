"""
shot_features.py
-----------------
Per-shot features for the shot-type classifier, computed ONLY from inputs the
BGL workflow also has at inference time:

    - shuttle trajectory     {stem}_ball_predicted.csv   (TrackNet)
    - bottom/top poses       {stem}_player_bottom/top.csv (YOLOv8-Pose)
    - court corners          court/{match_id}.out         (manual calibration)
    - hit frames + hitter    ShuttleSet labels in training, HitNet output in BGL

So the same function produces training features (format_shots.py) and
inference features (BGL), and nothing annotator-only can leak in.

Conventions
-----------
- Frames are clip-relative, same indexing as _hit.csv and the ball/pose CSVs.
- hitter: 1 / 2 exactly as in _hit.csv (ShuttleSet A/B + downcourt via
  court_side.py). Each label is paired with its pose CSV via
  POSE_FILE_FOR_LABEL (the files are swapped: label 1 -> _player_top.csv).
  Which half of the court the hitter is on is measured from YOLO feet per
  shot (hitter_near), not assumed from the label.
- Undetected values (0 in the CSVs) become NaN. Features that can't be
  computed are NaN -- LightGBM handles NaN natively.
- Court metres: A (top-left) = (0, 0), x across the width to 6.1,
  y along the length to 13.4, net at y = 6.7.
- "Hitter frame": positions and directions are expressed from the hitter's
  point of view, so a shot gets the same features whichever end it was
  played from:
      lateral  > 0  = towards the hitter's RIGHT
      depth    > 0  = distance behind the net on that player's own side
  The near player faces away from the camera (image right = their right);
  the far player faces the camera (image right = their left), so the far
  player's horizontal values are mirrored.

Racket hand is unknown, so arm features use whichever wrist is higher at
contact (the hitting arm for overhead shots; less reliable for low shots).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

COURT_W = 6.1    # doubles width (m)
COURT_L = 13.4   # doubles length (m)
NET_Y = COURT_L / 2

# COCO-17 keypoint indices
NOSE = 0
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANK, R_ANK = 15, 16

# Which pose CSV holds the player for each hitter label (1 = bottom, 2 = top,
# from ShuttleSet A/B + downcourt via court_side.py). format_data.py writes the
# poses with bottom/top swapped relative to these labels, so label 1 -> the
# "top" file. If format_data.py's near/far mapping is ever fixed (and the pose
# files regenerated), set this back to {1: "bottom", 2: "top"}.
POSE_FILE_FOR_LABEL = {1: "top", 2: "bottom"}

DEFAULT_FPS = 30.0
CONTACT_HALF_WINDOW = 2     # pose at contact = mean over [f-2, f+2]
SPEED_FRAMES = 4            # frames after contact used for launch velocity
MAX_FLIGHT_FRAMES = 90      # cap on the flight window for the last shot
JUMP_BASELINE = (15, 5)     # feet baseline = median over [f-15, f-5]
MOVE_FRAMES = 10            # hitter movement measured over [f-10, f]


# ── Loading ────────────────────────────────────────────────────────────────

@dataclass
class RallyData:
    ball: np.ndarray          # (n, 2) image px, NaN = undetected
    pose: dict                # {1: bottom (n, 17, 2), 2: top (n, 17, 2)}, NaN = undetected
    homography: np.ndarray    # (3, 3) image px -> court metres
    court_px: np.ndarray      # (4, 2) corners A, D, G, X in image px
    fps: float
    n: int


def read_court_corners(court_path: Path) -> np.ndarray:
    """A, D, G, X corners from a .out file, shape (4, 2)."""
    with open(court_path) as f:
        pts = [[float(v) for v in line.strip().split(";")] for line in f if line.strip()]
    if len(pts) != 4:
        raise ValueError(f"{court_path} has {len(pts)} lines, expected 4 (A, D, G, X)")
    return np.array(pts, dtype=np.float64)


def court_homography(corners_px: np.ndarray) -> np.ndarray:
    """Image px -> court metres, same world points as format_data.py."""
    world = np.array([[0, 0], [COURT_W, 0], [0, COURT_L], [COURT_W, COURT_L]], dtype=np.float32)
    h, _ = cv2.findHomography(corners_px.astype(np.float32), world)
    if h is None:
        raise RuntimeError("Court homography failed -- check the .out corners")
    return h


def _read_pose(path: Path) -> np.ndarray:
    arr = pd.read_csv(path).to_numpy(dtype=np.float64).reshape(-1, 17, 2)
    arr[(arr[..., 0] == 0) & (arr[..., 1] == 0)] = np.nan
    return arr


def load_rally(stem: str, data_dir: Path) -> RallyData:
    """Load one rally's ball, pose and court files from a data_pad15-style folder."""
    data_dir = Path(data_dir)
    match_id = stem.split("_set")[0]

    ball_df = pd.read_csv(data_dir / "ball_trajectory" / f"{stem}_ball_predicted.csv")
    ball = ball_df[["X", "Y"]].to_numpy(dtype=np.float64)
    undetected = (ball[:, 0] == 0) & (ball[:, 1] == 0)
    if "Visibility" in ball_df:
        undetected |= ball_df["Visibility"].to_numpy() == 0
    ball[undetected] = np.nan

    fps = DEFAULT_FPS
    if "Time" in ball_df and len(ball_df) > 2:
        dt = np.nanmedian(np.diff(ball_df["Time"].to_numpy(dtype=np.float64)))
        if dt > 0 and 10 <= 1 / dt <= 240:
            fps = 1 / dt

    file_a = _read_pose(data_dir / "poses" / f"{stem}_player_bottom.csv")
    file_b = _read_pose(data_dir / "poses" / f"{stem}_player_top.csv")

    corners = read_court_corners(data_dir / "court" / f"{match_id}.out")
    homography = court_homography(corners)
    n = min(len(ball), len(file_a), len(file_b))
    # The pose CSVs are swapped relative to the hitter labels: the player
    # labelled 1 (bottom, from ShuttleSet A/B + downcourt) is in
    # _player_top.csv, and label 2 is in _player_bottom.csv. Confirmed visually
    # across sets 1-3 (render_shot_check.py). See POSE_FILE_FOR_LABEL.
    files = {"bottom": file_a[:n], "top": file_b[:n]}
    return RallyData(
        ball=ball[:n], pose={lab: files[name] for lab, name in POSE_FILE_FOR_LABEL.items()},
        homography=homography, court_px=corners, fps=fps, n=n,
    )


# Feet must land within these margins of the court (m) to count as a player --
# rejects the umpire (beside the net, ~1 m outside the sideline in matches
# 88/90/93), line judges and coaches picked up by the pose model. Players
# rarely go far past the sidelines but do go well behind the baselines.
SIDE_MARGIN = 0.8
END_MARGIN = 2.0


def assign_players_by_court(pose_a: np.ndarray, pose_b: np.ndarray,
                            homography: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decide bottom/top per frame from where each player's feet are on court.

    Does NOT trust which CSV a pose came from: the existing pose files have
    bottom/top swapped (the "bottom" file holds the far player), so each
    frame's detections are re-sorted by court position instead.

        feet beyond the net (court y > 6.7) -> bottom (near) player
        feet before the net                 -> top (far) player
        both on the same side               -> the one deeper in that half wins
        feet off court (beyond the margins)  -> discarded

    Returns:
        (bottom, top), each (n, 17, 2) with NaN where no player was assigned.
    """
    n = len(pose_a)
    bottom = np.full_like(pose_a, np.nan)
    top = np.full_like(pose_a, np.nan)

    def feet_court(pose):
        feet = _nanmean(pose[:, [L_ANK, R_ANK]], axis=1)          # (n, 2)
        out = np.full((n, 2), np.nan)
        ok = ~np.isnan(feet).any(axis=1)
        if ok.any():
            out[ok] = cv2.perspectiveTransform(
                feet[ok].reshape(-1, 1, 2).astype(np.float64), homography).reshape(-1, 2)
        on_court = ((out[:, 0] > -SIDE_MARGIN) & (out[:, 0] < COURT_W + SIDE_MARGIN) &
                    (out[:, 1] > -END_MARGIN) & (out[:, 1] < COURT_L + END_MARGIN))
        return np.where(on_court, out[:, 1], np.nan)               # court y, NaN if unusable

    ya, yb = feet_court(pose_a), feet_court(pose_b)
    for i in range(n):
        cands = [(y, p) for y, p in ((ya[i], pose_a[i]), (yb[i], pose_b[i])) if not np.isnan(y)]
        if len(cands) == 2:
            (y1, p1), (y2, p2) = sorted(cands, key=lambda c: c[0])  # smaller y = further away
            top[i], bottom[i] = p1, p2
        elif len(cands) == 1:
            y, p = cands[0]
            (bottom if y > NET_Y else top)[i] = p
    return bottom, top


# ── Geometry helpers ───────────────────────────────────────────────────────

def _nanmean(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanmedian(a, axis=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def _to_court(pt: np.ndarray, h: np.ndarray) -> np.ndarray:
    """(2,) image px -> (2,) court metres; NaN in, NaN out."""
    if np.isnan(pt).any():
        return np.array([np.nan, np.nan])
    return cv2.perspectiveTransform(pt.reshape(1, 1, 2).astype(np.float64), h).reshape(2)


def _angle(a, b, c) -> float:
    """Angle ABC in degrees (NaN if any point is missing)."""
    ba, bc = a - b, c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if np.isnan(norm) or norm == 0:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / norm, -1, 1))))


def _window(arr: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """arr[lo:hi] clamped to valid indices."""
    return arr[max(lo, 0):max(min(hi, len(arr)), 0)]


def _feet(pose_frame: np.ndarray) -> np.ndarray:
    """Midpoint of the ankles, (2,) image px."""
    return _nanmean(pose_frame[[L_ANK, R_ANK]])


def _hitter_is_near(r: "RallyData", f: int, hitter: int) -> bool:
    """Is the hitter on the camera-near half (court y > net) at this shot?

    Measured from YOLO feet via the homography: the hitter's own feet, else the
    opponent's (opposite half), else the label convention as a last resort.
    """
    hw = CONTACT_HALF_WINDOW
    for player, flip in ((hitter, False), (3 - hitter, True)):
        p = _nanmean(_window(r.pose[player], f - hw, f + hw + 1))
        y = _to_court(_feet(p), r.homography)[1]
        if not np.isnan(y):
            return (y > NET_Y) != flip
    return hitter == 1


def _side_sign(near: bool) -> int:
    """+1 for the near player, -1 for the far player (mirrors horizontal values)."""
    return 1 if near else -1


def _hitter_view(court_xy: np.ndarray, near: bool) -> tuple[float, float]:
    """Court metres -> (lateral, depth behind the net on the hitter's side)."""
    x, y = court_xy
    lateral = (x - COURT_W / 2) * _side_sign(near)
    depth = (y - NET_Y) if near else (NET_Y - y)
    return float(lateral), float(depth)


def _court_px_scale(corners: np.ndarray) -> tuple[float, float]:
    """Court length and width in image px (for normalising image-space motion)."""
    a, d, g, x = corners
    length = np.linalg.norm((a + d) / 2 - (g + x) / 2)
    width = (np.linalg.norm(d - a) + np.linalg.norm(x - g)) / 2
    return float(length), float(width)


# ── Feature groups ─────────────────────────────────────────────────────────

def _position_features(r: RallyData, f: int, nf: int | None, hitter: int, near: bool) -> dict:
    opp = 3 - hitter
    hw = CONTACT_HALF_WINDOW
    out = {}

    hit_pose = _nanmean(_window(r.pose[hitter], f - hw, f + hw + 1))
    opp_pose = _nanmean(_window(r.pose[opp], f - hw, f + hw + 1))
    hitter_court = _to_court(_feet(hit_pose), r.homography)
    opp_court = _to_court(_feet(opp_pose), r.homography)

    out["hitter_lat"], out["hitter_depth"] = _hitter_view(hitter_court, near)
    # opponent, also in the hitter's frame: depth = how far behind the net on THEIR side
    lat, depth = _hitter_view(opp_court, near)
    out["opp_lat"], out["opp_depth"] = lat, -depth

    # Landing: receiver's feet at the next contact; for the last shot, the
    # shuttle's last tracked point (it has come down by then).
    landing = np.array([np.nan, np.nan])
    source = ""
    if nf is not None:
        rec_pose = _nanmean(_window(r.pose[opp], nf - hw, nf + hw + 1))
        landing = _to_court(_feet(rec_pose), r.homography)
        source = "receiver"
    else:
        flight = _window(r.ball, f + 1, f + MAX_FLIGHT_FRAMES)
        valid = flight[~np.isnan(flight).any(axis=1)]
        if len(valid):
            landing = _to_court(valid[-1], r.homography)
            source = "shuttle_end"
    lat, depth = _hitter_view(landing, near)
    out["land_lat"], out["land_depth"] = lat, -depth   # > 0 = into the opponent's court
    out["landing_source"] = source

    out["shot_distance"] = float(np.linalg.norm(landing - hitter_court))
    out["lat_change"] = out["land_lat"] - out["hitter_lat"]     # ~0 straight, large = cross-court
    out["hitter_court_x"], out["hitter_court_y"] = hitter_court
    out["land_court_x"], out["land_court_y"] = landing

    # Hitter movement into the shot (m/s): lunges and jumps arrive fast
    before = _window(r.pose[hitter], f - MOVE_FRAMES, f + 1)
    if len(before) >= 2:
        p0 = _to_court(_feet(before[0]), r.homography)
        p1 = _to_court(_feet(before[-1]), r.homography)
        out["hitter_move_speed"] = float(np.linalg.norm(p1 - p0) * r.fps / (len(before) - 1))
    else:
        out["hitter_move_speed"] = np.nan
    return out


def _timing_features(r: RallyData, f: int, nf: int | None, pf: int | None) -> dict:
    return {
        "flight_s": (nf - f) / r.fps if nf is not None else np.nan,
        "prev_gap_s": (f - pf) / r.fps if pf is not None else np.nan,
        "is_last": nf is None,
    }


def _shuttle_features(r: RallyData, f: int, nf: int | None, near: bool, body_h: float,
                      nose_y: float) -> dict:
    s = _side_sign(near)
    court_len_px, court_w_px = _court_px_scale(r.court_px)
    out = {}

    # Shuttle at contact: nearest detection within the contact window
    contact = np.array([np.nan, np.nan])
    for off in (0, -1, 1, -2, 2):
        if 0 <= f + off < r.n and not np.isnan(r.ball[f + off]).any():
            contact = r.ball[f + off]
            break
    # > 0 = contact above the head (overhead shots), in body heights
    out["contact_above_head"] = (nose_y - contact[1]) / body_h if body_h > 0 else np.nan

    end = nf if nf is not None else f + MAX_FLIGHT_FRAMES
    flight = _window(r.ball, f, end + 1)
    valid = ~np.isnan(flight).any(axis=1)
    out["shuttle_detect_rate"] = float(valid.mean()) if len(flight) else np.nan

    if valid.sum() >= 2:
        pts = flight[valid]
        apex_y = pts[:, 1].min()                         # image y grows downwards
        out["apex_above_start"] = (pts[0, 1] - apex_y) / court_len_px
        out["apex_above_end"] = (pts[-1, 1] - apex_y) / court_len_px
        path = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
        out["mean_speed"] = path / court_len_px / ((len(flight) - 1) / r.fps)
        out["img_dx"] = s * (pts[-1, 0] - pts[0, 0]) / court_w_px   # mirrored for the top player
        out["img_dy_fwd"] = -s * (pts[-1, 1] - pts[0, 1]) / court_len_px  # > 0 = towards opponent
    else:
        for k in ("apex_above_start", "apex_above_end", "mean_speed", "img_dx", "img_dy_fwd"):
            out[k] = np.nan

    # Launch velocity: first frames after contact (smash = fast, steep, forward)
    launch = _window(r.ball, f, f + SPEED_FRAMES + 1)
    idx = np.flatnonzero(~np.isnan(launch).any(axis=1))
    if len(idx) >= 2:
        dt = (idx[-1] - idx[0]) / r.fps
        v = (launch[idx[-1]] - launch[idx[0]]) / dt
        out["launch_vx"] = s * v[0] / court_w_px
        out["launch_vy_up"] = -v[1] / court_len_px     # > 0 = rising in the image
        out["launch_speed"] = float(np.hypot(v[0] / court_w_px, v[1] / court_len_px))
    else:
        out["launch_vx"] = out["launch_vy_up"] = out["launch_speed"] = np.nan
    return out


def _pose_features(r: RallyData, f: int, hitter: int, near: bool) -> tuple[dict, float, float]:
    """Hitter pose at contact. Also returns (body height px, nose y) for normalising."""
    s = _side_sign(near)
    hw = CONTACT_HALF_WINDOW
    p = _nanmean(_window(r.pose[hitter], f - hw, f + hw + 1))   # (17, 2)
    out = {"pose_detect_rate": float((~np.isnan(p).any(axis=1)).mean())}

    feet = _feet(p)
    nose_y = p[NOSE, 1]
    body_h = feet[1] - nose_y
    if not body_h > 0:
        body_h = np.nan

    # Hitting arm = the higher wrist
    wrists = {"L": (L_SHO, L_ELB, L_WRI), "R": (R_SHO, R_ELB, R_WRI)}
    wy = {k: p[w, 1] for k, (_, _, w) in wrists.items()}
    if all(np.isnan(v) for v in wy.values()):
        side = None
    else:
        side = min(wy, key=lambda k: np.inf if np.isnan(wy[k]) else wy[k])

    hip_mid = _nanmean(p[[L_HIP, R_HIP]])
    sho_mid = _nanmean(p[[L_SHO, R_SHO]])

    if side is not None:
        sho, elb, wri = wrists[side]
        hip = L_HIP if side == "L" else R_HIP
        out["wrist_above_head"] = (nose_y - p[wri, 1]) / body_h
        out["elbow_angle"] = _angle(p[sho], p[elb], p[wri])
        out["shoulder_angle"] = _angle(p[hip], p[sho], p[elb])       # arm raise
        out["wrist_lat_offset"] = s * (p[wri, 0] - hip_mid[0]) / body_h
    else:
        out["wrist_above_head"] = out["elbow_angle"] = np.nan
        out["shoulder_angle"] = out["wrist_lat_offset"] = np.nan

    knees = [_angle(p[L_HIP], p[L_KNEE], p[L_ANK]), _angle(p[R_HIP], p[R_KNEE], p[R_ANK])]
    out["knee_angle_min"] = float(np.nanmin(knees)) if not np.isnan(knees).all() else np.nan
    out["knee_angle_max"] = float(np.nanmax(knees)) if not np.isnan(knees).all() else np.nan
    out["stance_width"] = np.linalg.norm(p[L_ANK] - p[R_ANK]) / body_h

    torso = sho_mid - hip_mid
    out["torso_lean"] = (float(np.degrees(np.arctan2(s * torso[0], -torso[1])))
                         if not np.isnan(torso).any() else np.nan)

    # Jump: feet raised relative to the player's stance shortly before contact
    lo, hi = JUMP_BASELINE
    base = _window(r.pose[hitter], f - lo, f - hi + 1)
    base_y = _nanmedian(_nanmean(base[:, [L_ANK, R_ANK], 1], axis=1)) if len(base) else np.nan
    out["jump"] = (base_y - feet[1]) / body_h
    return out, body_h, nose_y


# ── Public API ─────────────────────────────────────────────────────────────

def shot_features(
    rally: RallyData,
    frame: int,
    hitter: int,
    next_frame: int | None = None,
    prev_frame: int | None = None,
) -> dict:
    """All features for one shot.

    Args:
        rally: Output of load_rally().
        frame: Clip-relative hit frame.
        hitter: 1 = bottom, 2 = top.
        next_frame: Clip-relative frame of the next shot (None for the last).
        prev_frame: Clip-relative frame of the previous shot (None for the serve).
    """
    if not 0 <= frame < rally.n:
        raise ValueError(f"frame {frame} outside rally (0..{rally.n - 1})")
    nf = None if next_frame is None or pd.isna(next_frame) else int(next_frame)
    pf = None if prev_frame is None or pd.isna(prev_frame) else int(prev_frame)

    near = _hitter_is_near(rally, frame, hitter)
    pose, body_h, nose_y = _pose_features(rally, frame, hitter, near)
    feats = {"hitter_near": near}
    feats.update(_position_features(rally, frame, nf, hitter, near))
    feats.update(_timing_features(rally, frame, nf, pf))
    feats.update(_shuttle_features(rally, frame, nf, near, body_h, nose_y))
    feats.update(pose)
    feats["body_height_px"] = body_h
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in feats.items()}