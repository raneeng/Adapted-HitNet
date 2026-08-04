#!/usr/bin/env python3
"""
movement_tracker.py
===================
Standalone whole-video player movement tracker.

This script is intentionally self-contained. It copies the court calibration
logic it needs instead of importing court_calibration.py, so it can be moved to
another machine with only its Python dependencies and YOLO pose weights.

Note: .out files contain the four doubles court corners
    A -------- D
    |          |
    |          |
    G -------- X
    in the order:
    
    1. Top-left (A)
    2. Top-right (D)
    3. Bottom-left (G)
    4. Bottom-right (X)

Workflow:
  1. Open a video frame and ask the user to click 4 court corners:
     bottom-left, bottom-right, top-right, top-left.
  2. Compute a pixel-to-court homography.
  3. Run YOLOv8 pose across the whole video, sampled every N frames.
  4. Save absolute-frame player tracks to player_tracks.csv.

Output CSV columns:
  rally_id, frame, player, pixel_x, pixel_y, world_x, world_y, confidence

The web app later slices this full-video CSV by rally boundaries and rewrites
rally_id values for per-rally movement plots and training-load outputs.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Court dimensions and calibration helpers copied from court_calibration.py.
# Official court dimensions (Full doubles court)
COURT_W = 6.1 # doubles width (m)
COURT_L = 13.4 # full court length (m)

PT_COLOURS = [(0, 255, 255), (0, 255, 0), (255, 0, 255), (255, 165, 0)]

# Keep only these COCO keypoints
USED_KEYPOINTS = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

MIN_KEYPOINT_CONF = 0.3
MIN_BBOX_CONF = 0.4
COURT_MARGIN = 1.5


def _court_lines():
    """Generates coordinates representing standard badminton court lines.
    
    Returns:
        list[tuple[tuple[float, float], tuple[float, float]]]: list of start/end pairs of 
            coordinate pairs representing court lines
        
    """
    w, l = COURT_W, COURT_L
    net = l / 2
    ssl_near = net - 1.98
    ssl_far = net + 1.98
    lsl_near = 0.76
    lsl_far = l - 0.76
    side_in = 0.46
    cx = w / 2
    return [
        ((0, 0), (w, 0)),
        ((w, 0), (w, l)),
        ((w, l), (0, l)),
        ((0, l), (0, 0)),
        ((0, net), (w, net)),
        ((0, ssl_near), (w, ssl_near)),
        ((0, ssl_far), (w, ssl_far)),
        ((0, lsl_near), (w, lsl_near)),
        ((0, lsl_far), (w, lsl_far)),
        ((cx, ssl_near), (cx, ssl_far)),
        ((side_in, 0), (side_in, l)),
        ((w - side_in, 0), (w - side_in, l)),
    ]


COURT_LINES = _court_lines()


def compute_court_zones(court_w=COURT_W, court_l=COURT_L):
    """ Divides the badminton court into a grid of 16 zones (4x4)
    Args:
        court_w (float): The total width of the court in meters. Defaults to COURT_W.
        court_l (float): The total length of the court in meters. Defaults to COURT_L.

    Returns:
        list[dict]: A list of dictionaries containing spatial bounds and metadata for each zone
    """
    net_y = court_l / 2
    rows = ["A", "B", "C", "D"]
    col_w = court_w / 4
    zones = []

    row_h = net_y / 4
    for ri, row in enumerate(rows):
        y_max = net_y - ri * row_h
        y_min = net_y - (ri + 1) * row_h
        for ci in range(4):
            zones.append({
                "name": f"P1_{row}{ci + 1}",
                "player": "P1",
                "label": f"P1 {row}{ci + 1}",
                "x_min": ci * col_w,
                "x_max": (ci + 1) * col_w,
                "y_min": y_min,
                "y_max": y_max,
            })

    row_h = (court_l - net_y) / 4
    for ri, row in enumerate(rows):
        y_min = net_y + ri * row_h
        y_max = net_y + (ri + 1) * row_h
        for ci in range(4):
            zones.append({
                "name": f"P2_{row}{ci + 1}",
                "player": "P2",
                "label": f"P2 {row}{ci + 1}",
                "x_min": court_w - (ci + 1) * col_w,
                "x_max": court_w - ci * col_w,
                "y_min": y_min,
                "y_max": y_max,
            })

    return zones


COURT_ZONES = compute_court_zones()


def world_to_image(wx, wy, h_inv):
    """Projects real-world court coordinates (in meters) back to 2D image pixel coordinates.

    Args:
        wx (float): Real-world X position coordinate on court layout in meters.
        wy (float): Real-world Y position coordinate on court layout in meters.
        h_inv (np.ndarray): inverse homography matrix 

    Returns:
        tuple[int, int] | None: A tuple of (x, y) pixel coordinates if the projection 
            is valid; None otherwise.

    """
    p = np.array([wx, wy, 1.0])
    q = h_inv @ p
    if abs(q[2]) < 1e-8:
        return None
    return int(q[0] / q[2]), int(q[1] / q[2])


def project_to_court(px, py, h):
    """Projects image pixel coordinates into real-world court coordinates in meters.

    Args:
        px: Image pixel x-coordinate
        py: Image pixel y-coordinate
        h: A NumPy array matrix tracking the computed forward homography transform.

    Returns:
        tuple[float, float] | None: A tuple coordinate pair (world_x, world_y) in meters, else None.
    """
    p = np.array([px, py, 1.0])
    w = h @ p
    if abs(w[2]) < 1e-8:
        return None
    return float(w[0] / w[2]), float(w[1] / w[2])


def point_in_court(x, y, margin=COURT_MARGIN):
    """Checks whether a point falls within court boundaries.

    Incorporates a flexible tolerance buffer threshold to track layout spillover.

    Args:
        x (float): Real-world lateral distance evaluation target coordinate in meters.
        y (float): Real-world longitudinal distance evaluation target coordinate in meters.
        margin (float): Buffer boundary context allowing tracking slightly off-court.
            Defaults to COURT_MARGIN.

    Returns:
        bool: True if position falls safely within region boundaries, else False.
    """
    return -margin <= x <= COURT_W + margin and -margin <= y <= COURT_L + margin


def build_pose_features(
    video_path,
    calibration,
    pose_model,
    sample_rate=3,
    start_frame=0,
    end_frame=None,
):

    court_w = COURT_W
    court_l = COURT_L
    net_y = court_l / 2.0
    
    h = np.array(calibration["homography_matrix"], dtype=np.float64)
 
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    sf = max(0, int(start_frame))
    ef = min(
        total - 1,
        int(end_frame) if end_frame is not None else total - 1,
    )

    if sf > ef:
        raise ValueError("start_frame after end_frame")

    print(f"[Pose] Loading model: {pose_model}")
    model = YOLO(pose_model)

    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)

    rows = []
    processed = 0

    for frame_idx in range(sf, ef + 1):

        ok, frame = cap.read()

        if not ok:
            break

        if (frame_idx - sf) % sample_rate != 0:
            continue

        processed += 1

        results = model(
            frame,
            verbose=False,
            conf=MIN_BBOX_CONF,
        )

        if (
            not results
            or results[0].keypoints is None
            or len(results[0].keypoints) == 0
        ):
            continue

        kp_data = results[0].keypoints.data.cpu().numpy()

        ############################################################
        # One row per frame
        ############################################################

        frame_row = {
            "frame": frame_idx,
        }

        ############################################################
        # Initialise ALL columns to NaN
        ############################################################

        for side in ["near", "far"]:

            for kp_idx in range(17):

                frame_row[f"{side}_kp_{kp_idx}_px"] = np.nan
                frame_row[f"{side}_kp_{kp_idx}_py"] = np.nan
                frame_row[f"{side}_kp_{kp_idx}_conf"] = np.nan

            frame_row[f"{side}_left_ankle_world_x"] = np.nan
            frame_row[f"{side}_left_ankle_world_y"] = np.nan
            frame_row[f"{side}_right_ankle_world_x"] = np.nan
            frame_row[f"{side}_right_ankle_world_y"] = np.nan

        ############################################################
        # Keep highest-confidence player on each side
        ############################################################

        best_conf = {
            "near": -1.0,
            "far": -1.0,
        }

        detected_player = False

        for det_idx in range(kp_data.shape[0]):

            right_ankle = kp_data[
                det_idx,
                USED_KEYPOINTS["right_ankle"],
            ]

            ankle_conf = float(right_ankle[2])

            if ankle_conf < MIN_KEYPOINT_CONF:
                continue

            world = project_to_court(
                float(right_ankle[0]),
                float(right_ankle[1]),
                h,
            )

            if world is None:
                continue

            rx, ry = world

            # check if right ankle in court
            if (
                rx < -COURT_MARGIN
                or rx > court_w + COURT_MARGIN
                or ry < -COURT_MARGIN
                or ry > court_l + COURT_MARGIN
            ):
                continue

            player_prefix = "near" if ry < net_y else "far"

            ########################################################
            # Ignore weaker duplicate detections
            ########################################################

            if ankle_conf <= best_conf[player_prefix]:
                continue

            best_conf[player_prefix] = ankle_conf
            detected_player = True

            ########################################################
            # Save all COCO keypoints
            ########################################################

            for kp_idx in range(17):

                kp = kp_data[det_idx, kp_idx]

                frame_row[f"{player_prefix}_kp_{kp_idx}_px"] = float(kp[0])
                frame_row[f"{player_prefix}_kp_{kp_idx}_py"] = float(kp[1])
                frame_row[f"{player_prefix}_kp_{kp_idx}_conf"] = float(kp[2])

            ########################################################
            # Left ankle world coordinate
            ########################################################

            left = kp_data[
                det_idx,
                USED_KEYPOINTS["left_ankle"],
            ]

            if left[2] >= MIN_KEYPOINT_CONF:

                world = project_to_court(
                    float(left[0]),
                    float(left[1]),
                    h,
                )

                if world is not None:

                    frame_row[
                        f"{player_prefix}_left_ankle_world_x"
                    ] = float(world[0])

                    frame_row[
                        f"{player_prefix}_left_ankle_world_y"
                    ] = float(world[1])

            ########################################################
            # Right ankle world coordinate
            ########################################################

            world = project_to_court(
                float(right_ankle[0]),
                float(right_ankle[1]),
                h,
            )

            if world is not None:

                frame_row[
                    f"{player_prefix}_right_ankle_world_x"
                ] = float(world[0])

                frame_row[
                    f"{player_prefix}_right_ankle_world_y"
                ] = float(world[1])

        ############################################################
        # Save frame only if at least one player detected
        ############################################################

        if detected_player:
            rows.append(frame_row)

    cap.release()

    df = pd.DataFrame(rows)

    print(f"[Pose] Video: {total} frames @ {fps:.2f} fps")
    print(f"[Pose] Sampled every {sample_rate} frame(s)")
    print(f"[Pose] Processed {processed} sampled frames")
    print(f"[Pose] Returned {len(df)} sampled frames")

    return df
