import pandas as pd
import math
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
import os
import sys
import time
import csv
from collections import deque
from pathlib import Path

if sys.platform == "win32":
    os.add_dll_directory("C:/Windows/System32")

import cv2
from PIL import Image
from tqdm import tqdm


TRACKNET_PATH = Path(__file__).parent / "weights" / "TrackNet_best.pt"
INPAINTNET_PATH = Path(__file__).parent / "weights" / "InpaintNet_best.pt"
TN_WIDTH = 512
TN_HEIGHT = 288
COOR_TH = 50 / math.sqrt(TN_HEIGHT**2 + TN_WIDTH**2)


def print_device_info():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        props = torch.cuda.get_device_properties(0)
        vram = props.total_memory / 1024**3
        print(f"GPU   : {props.name} ({vram:.1f} GB VRAM)")
    else:
        print("GPU   : not available; running on CPU")
    return device


def get_model(model_name, seq_len=None, bg_mode=None):
    from tracknetv3_model import InpaintNet, TrackNet

    if model_name == "TrackNet":
        if bg_mode == "subtract":
            return TrackNet(in_dim=seq_len, out_dim=seq_len)
        if bg_mode == "subtract_concat":
            return TrackNet(in_dim=seq_len * 4, out_dim=seq_len)
        if bg_mode == "concat":
            return TrackNet(in_dim=(seq_len + 1) * 3, out_dim=seq_len)
        return TrackNet(in_dim=seq_len * 3, out_dim=seq_len)

    if model_name == "InpaintNet":
        return InpaintNet()

    raise ValueError(f"Invalid model name: {model_name}")


def get_ensemble_weight(seq_len, eval_mode="weight"):
    if eval_mode == "average":
        return torch.ones(seq_len) / seq_len
    if eval_mode != "weight":
        raise ValueError(f"Invalid ensemble mode: {eval_mode}")

    weight = torch.ones(seq_len)
    for idx in range(math.ceil(seq_len / 2)):
        weight[idx] = idx + 1
        weight[seq_len - idx - 1] = idx + 1
    return weight / weight.sum()


def predict_location(heatmap):
    if np.amax(heatmap) == 0:
        return 0, 0, 0, 0

    contours, _ = cv2.findContours(
        heatmap.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    rects = [cv2.boundingRect(contour) for contour in contours]
    return max(rects, key=lambda rect: rect[2] * rect[3])


def load_tracknet(tracknet_file, device):
    checkpoint = torch.load(tracknet_file, map_location=device)
    seq_len = checkpoint["param_dict"]["seq_len"]
    bg_mode = checkpoint["param_dict"]["bg_mode"]

    model = get_model("TrackNet", seq_len, bg_mode).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"[TrackNetV3] Loaded TrackNet from {tracknet_file}")
    print(f"[TrackNetV3] seq_len={seq_len}, bg_mode='{bg_mode}'")
    return model, seq_len, bg_mode


def load_inpaintnet(inpaintnet_file, device):
    checkpoint = torch.load(inpaintnet_file, map_location=device)
    seq_len = checkpoint["param_dict"]["seq_len"]

    model = get_model("InpaintNet").to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    print(f"[TrackNetV3] Loaded InpaintNet from {inpaintnet_file}")
    print(f"[TrackNetV3] inpaint seq_len={seq_len}")
    return model, seq_len


def predict_coordinates(indices, y_pred=None, c_pred=None, img_scaler=(1, 1)):
    pred_dict = {"Frame": [], "X": [], "Y": [], "Visibility": []}

    batch_size, seq_len = indices.shape[0], indices.shape[1]
    indices = indices.detach().cpu().numpy() if torch.is_tensor(indices) else indices

    if y_pred is not None:
        y_pred = y_pred > 0.5
        y_pred = y_pred.detach().cpu().numpy() if torch.is_tensor(y_pred) else y_pred
    if c_pred is not None:
        c_pred = c_pred.detach().cpu().numpy() if torch.is_tensor(c_pred) else c_pred

    prev_frame_idx = -1
    for batch_idx in range(batch_size):
        for seq_idx in range(seq_len):
            frame_idx = indices[batch_idx][seq_idx][1]
            if frame_idx == prev_frame_idx:
                break

            if c_pred is not None:
                coordinate = c_pred[batch_idx][seq_idx]
                x_pred = int(coordinate[0] * TN_WIDTH * img_scaler[0])
                y_value = int(coordinate[1] * TN_HEIGHT * img_scaler[1])
            elif y_pred is not None:
                x, y, width, height = predict_location(
                    (y_pred[batch_idx][seq_idx] * 255).astype("uint8")
                )
                x_pred = int((x + width / 2) * img_scaler[0])
                y_value = int((y + height / 2) * img_scaler[1])
            else:
                raise ValueError("Either y_pred or c_pred must be provided.")

            visibility = 0 if x_pred == 0 and y_value == 0 else 1
            pred_dict["Frame"].append(int(frame_idx))
            pred_dict["X"].append(x_pred)
            pred_dict["Y"].append(y_value)
            pred_dict["Visibility"].append(visibility)
            prev_frame_idx = frame_idx

    return pred_dict


def generate_inpaint_mask(pred_dict, th_h=30):
    y_values = np.array(pred_dict["Y"])
    visibility = np.array(pred_dict["Visibility"])
    inpaint_mask = np.zeros_like(y_values)
    disappear_idx = 0
    appear_idx = 0

    while appear_idx < len(visibility):
        while disappear_idx < len(visibility) - 1 and visibility[disappear_idx] == 1:
            disappear_idx += 1
        appear_idx = disappear_idx
        while appear_idx < len(visibility) - 1 and visibility[appear_idx] == 0:
            appear_idx += 1

        if appear_idx == disappear_idx:
            break
        if disappear_idx == 0 and y_values[appear_idx] > th_h:
            inpaint_mask[:appear_idx] = 1
        elif (
            disappear_idx > 1
            and y_values[disappear_idx - 1] > th_h
            and appear_idx < len(visibility)
            and y_values[appear_idx] > th_h
        ):
            inpaint_mask[disappear_idx:appear_idx] = 1

        disappear_idx = appear_idx

    return inpaint_mask.tolist()


def read_video_metadata(video_path):
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if width == 0 or height == 0:
        raise RuntimeError(f"cv2 could not open video: {video_path}")
    if not fps or fps <= 0:
        fps = 30.0

    return width, height, fps, frame_count


def predict_with_tracknet(
    video_path,
    tracknet,
    seq_len,
    bg_mode,
    batch_size,
    device,
    num_workers,
):
    width, height, fps, frame_count = read_video_metadata(video_path)
    img_scaler = (width / TN_WIDTH, height / TN_HEIGHT)

    pred_dict = {
        "Frame": [],
        "X": [],
        "Y": [],
        "Visibility": [],
        "Inpaint_Mask": [],
        "Img_scaler": img_scaler,
        "Img_shape": (width, height),
    }

    dataset = VideoIterableDataset(
        str(video_path),
        seq_len=seq_len,
        sliding_step=1,
        bg_mode=bg_mode,
    )
    pin_memory = device == "cuda"
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )

    video_len = dataset.video_len
    num_sample = max(1, video_len - seq_len + 1)
    total_batches = max(1, math.ceil(num_sample / batch_size))

    print(f"[TrackNetV3] Video: {width}x{height} @ {fps:.2f}fps, {frame_count} frames")
    print(f"[TrackNetV3] Running TrackNet inference: ~{total_batches} batches")

    sample_count = 0
    buffer_size = seq_len - 1
    batch_i = torch.arange(seq_len)
    frame_i = torch.arange(seq_len - 1, -1, -1)
    y_pred_buffer = torch.zeros(
        (buffer_size, seq_len, TN_HEIGHT, TN_WIDTH),
        dtype=torch.float32,
    )
    weight = get_ensemble_weight(seq_len, "weight")

    started_at = time.time()
    estimate_printed = False
    for step, (indices, x) in enumerate(
        tqdm(data_loader, desc="TrackNet", total=total_batches, unit="batch")
    ):
        x = x.float().to(device, non_blocking=pin_memory)
        batch_len = indices.shape[0]
        with torch.no_grad():
            y_pred = tracknet(x).detach().cpu()

        y_pred_buffer = torch.cat((y_pred_buffer, y_pred), dim=0)
        ensemble_i = []
        ensemble_y = []

        for batch_idx in range(batch_len):
            if sample_count < buffer_size:
                y_ensemble = y_pred_buffer[batch_i + batch_idx, frame_i].sum(0)
                y_ensemble = y_ensemble / (sample_count + 1)
            else:
                y_ensemble = (
                    y_pred_buffer[batch_i + batch_idx, frame_i]
                    * weight[:, None, None]
                ).sum(0)

            ensemble_i.append(indices[batch_idx][0].reshape(1, 1, 2))
            ensemble_y.append(y_ensemble.reshape(1, 1, TN_HEIGHT, TN_WIDTH))
            sample_count += 1

            if sample_count == num_sample:
                zero_pad = torch.zeros(
                    (buffer_size, seq_len, TN_HEIGHT, TN_WIDTH),
                    dtype=torch.float32,
                )
                y_pred_buffer = torch.cat((y_pred_buffer, zero_pad), dim=0)
                for frame_offset in range(1, seq_len):
                    y_ensemble = y_pred_buffer[
                        batch_i + batch_idx + frame_offset, frame_i
                    ].sum(0)
                    y_ensemble = y_ensemble / (seq_len - frame_offset)
                    ensemble_i.append(
                        indices[-1][frame_offset].reshape(1, 1, 2)
                    )
                    ensemble_y.append(
                        y_ensemble.reshape(1, 1, TN_HEIGHT, TN_WIDTH)
                    )

        if ensemble_y:
            batch_pred = predict_coordinates(
                torch.cat(ensemble_i, dim=0).float(),
                y_pred=torch.cat(ensemble_y, dim=0),
                img_scaler=img_scaler,
            )
            for key, values in batch_pred.items():
                pred_dict[key].extend(values)

        y_pred_buffer = y_pred_buffer[-buffer_size:]

        if not estimate_printed and step + 1 >= min(5, total_batches):
            elapsed = time.time() - started_at
            seconds_per_batch = elapsed / (step + 1)
            total_seconds = total_batches * seconds_per_batch
            total_minutes = int(total_seconds // 60)
            tqdm.write(
                f"[TrackNetV3] ~{seconds_per_batch:.2f}s/batch, "
                f"estimated total ~{total_minutes}m"
            )
            estimate_printed = True

    return pred_dict, width, height, fps, frame_count


def run_inpaintnet(pred_dict, inpaintnet, seq_len, batch_size, device, num_workers):
    width, height = pred_dict["Img_shape"]

    pred_dict["Inpaint_Mask"] = generate_inpaint_mask(pred_dict, th_h=height * 0.05)
    dataset = CoordinateDataset(
        pred_dict=pred_dict,
        seq_len=seq_len,
        sliding_step=1,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    weight = get_ensemble_weight(seq_len, "weight")
    num_sample = len(dataset)
    total_batches = max(1, math.ceil(num_sample / batch_size))

    print(f"[TrackNetV3] Running InpaintNet rectification: ~{total_batches} batches")

    inpaint_pred = {"Frame": [], "X": [], "Y": [], "Visibility": []}
    sample_count = 0
    buffer_size = seq_len - 1
    batch_i = torch.arange(seq_len)
    frame_i = torch.arange(seq_len - 1, -1, -1)
    coor_buffer = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)

    for indices, coor_pred, mask in tqdm(
        data_loader, desc="InpaintNet", total=total_batches, unit="batch"
    ):
        coor_pred = coor_pred.float()
        mask = mask.float()
        batch_len = indices.shape[0]

        with torch.no_grad():
            coor_inpaint = inpaintnet(
                coor_pred.to(device),
                mask.to(device),
            ).detach().cpu()
            coor_inpaint = coor_inpaint * mask + coor_pred * (1 - mask)

        too_small = (coor_inpaint[:, :, 0] < COOR_TH) & (
            coor_inpaint[:, :, 1] < COOR_TH
        )
        coor_inpaint[too_small] = 0.0

        coor_buffer = torch.cat((coor_buffer, coor_inpaint), dim=0)
        ensemble_i = []
        ensemble_c = []

        for batch_idx in range(batch_len):
            if sample_count < buffer_size:
                coor_ensemble = coor_buffer[batch_i + batch_idx, frame_i].sum(0)
                coor_ensemble = coor_ensemble / (sample_count + 1)
            else:
                coor_ensemble = (
                    coor_buffer[batch_i + batch_idx, frame_i] * weight[:, None]
                ).sum(0)

            ensemble_i.append(indices[batch_idx][0].view(1, 1, 2))
            ensemble_c.append(coor_ensemble.view(1, 1, 2))
            sample_count += 1

            if sample_count == num_sample:
                zero_pad = torch.zeros((buffer_size, seq_len, 2), dtype=torch.float32)
                coor_buffer = torch.cat((coor_buffer, zero_pad), dim=0)
                for frame_offset in range(1, seq_len):
                    coor_ensemble = coor_buffer[
                        batch_i + batch_idx + frame_offset, frame_i
                    ].sum(0)
                    coor_ensemble = coor_ensemble / (seq_len - frame_offset)
                    ensemble_i.append(indices[-1][frame_offset].view(1, 1, 2))
                    ensemble_c.append(coor_ensemble.view(1, 1, 2))

        if ensemble_c:
            ensemble_c = torch.cat(ensemble_c, dim=0)
            too_small = (ensemble_c[:, :, 0] < COOR_TH) & (
                ensemble_c[:, :, 1] < COOR_TH
            )
            ensemble_c[too_small] = 0.0

            batch_pred = predict_coordinates(
                torch.cat(ensemble_i, dim=0).float(),
                c_pred=ensemble_c,
                img_scaler=pred_dict["Img_scaler"],
            )
            for key, values in batch_pred.items():
                inpaint_pred[key].extend(values)

        coor_buffer = coor_buffer[-buffer_size:]

    return inpaint_pred


def normalize_prediction_length(pred_dict, frame_count):
    lookup = {
        int(frame): (int(vis), int(x), int(y))
        for frame, vis, x, y in zip(
            pred_dict["Frame"],
            pred_dict["Visibility"],
            pred_dict["X"],
            pred_dict["Y"],
        )
    }

    normalized = {"Frame": [], "Visibility": [], "X": [], "Y": []}
    for frame_idx in range(frame_count):
        vis, x, y = lookup.get(frame_idx, (0, 0, 0))
        normalized["Frame"].append(frame_idx)
        normalized["Visibility"].append(vis)
        normalized["X"].append(x)
        normalized["Y"].append(y)
    return normalized


def track_shuttle(video_path, weights_dir, batch_size, use_inpaint, num_workers=0):
    """ Executes the core computer vision inference loop to extract shuttlecock positions frame-by-frame.

    Instantiates multi-threaded data loaders, pumps video frame matrices through the TrackNet spatial layer, 
        maps heatmaps to maximum probability coordinate vectors, and extracts structural metadata.

    Args:
        video_path (pathlib.Path): Absolute location of the target video file container.
        weights_dir (pathlib.Path): Folder containing the model weights.
        batch_size (int): Total number of video frames bundled into a single parallel GPU forward pass.
        use_inpaint (bool): Flag toggling intermediate tracking recovery layers.
        num_workers (int): Total subprocess count allocated for async frame loading pipelines.

       Returns:
        tuple: A five-element data package detailing:
            - pred_dict (dict): Compiled dictionary structure holding tracking results (Frame, Visibility, X, Y).
            - width (int): Original pixel width of the source video asset.
            - height (int): Original pixel height of the source video asset.
            - fps (float): Native frame rate frequency of the video container.
            - frame_count (int): Absolute number of frames evaluated. 
    """
    device = print_device_info()
    tracknet_file = weights_dir / "TrackNet_best.pt"
    inpaintnet_file = weights_dir / "InpaintNet_best.pt"

    if not tracknet_file.is_file():
        raise FileNotFoundError(f"Missing TrackNet checkpoint: {tracknet_file}")

    tracknet, tracknet_seq_len, bg_mode = load_tracknet(tracknet_file, device)
    pred_dict, width, height, fps, frame_count = predict_with_tracknet(
        video_path,
        tracknet,
        tracknet_seq_len,
        bg_mode,
        batch_size,
        device,
        num_workers,
    )

    if use_inpaint:
        if inpaintnet_file.is_file():
            inpaintnet, inpaint_seq_len = load_inpaintnet(inpaintnet_file, device)
            final_pred = run_inpaintnet(
                pred_dict,
                inpaintnet,
                inpaint_seq_len,
                batch_size,
                device,
                num_workers,
            )
        else:
            print(f"[TrackNetV3] InpaintNet checkpoint not found: {inpaintnet_file}")
            print("[TrackNetV3] Continuing with raw TrackNet predictions.")
            final_pred = {
                key: pred_dict[key] for key in ("Frame", "Visibility", "X", "Y")
            }
    else:
        final_pred = {key: pred_dict[key] for key in ("Frame", "Visibility", "X", "Y")}

    final_pred = normalize_prediction_length(final_pred, frame_count)
    return final_pred, width, height, fps, frame_count


def write_prediction_csv(pred_dict, fps, csv_path):
    """Write shuttle predictions to a HitNet-format CSV: Frame,Visibility,X,Y,Time.

    This is the dense, per-frame ball trajectory format HitNet's training
    pipeline expects (matching the `_ball_predicted.csv` convention) -- one
    row for every frame, no gaps, with an added Time column (seconds from
    the start of the clip) beyond the original shuttle_tracker.py schema.

    Args:
        pred_dict: Dict with "Frame", "Visibility", "X", "Y" arrays/lists
            (same shape as track_shuttle()'s first return value).
        fps: Frame rate of the source video, used to compute Time = Frame / fps.
        csv_path: Output CSV path.

    Returns:
        None. Writes the CSV to csv_path.
    """
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Frame", "Visibility", "X", "Y", "Time"])
        for frame, vis, x, y in zip(
            pred_dict["Frame"],
            pred_dict["Visibility"],
            pred_dict["X"],
            pred_dict["Y"],
        ):
            writer.writerow([frame, vis, x, y, round(frame / fps, 6)])


def build_shuttle_features(
    final_pred,
    sample_rate,
    offsets=(-3, -2, -1, 0, 1, 2, 3),
):
    """
    Build shuttle temporal features.

    Parameters
    ----------
    final_pred : dict
        {
            "Frame": [...],
            "Visibility": [...],
            "X": [...],
            "Y": [...]
        }

    sample_rate : int
        Sampling interval used during preprocessing.

    Returns
    -------
    pandas.DataFrame
        One row per sampled frame.
    """

    frames = np.asarray(final_pred["Frame"], dtype=int)
    visibility = np.asarray(final_pred["Visibility"], dtype=np.uint8)
    px = np.asarray(final_pred["X"], dtype=float)
    py = np.asarray(final_pred["Y"], dtype=float)

    n_frames = len(frames)

    rows = []

    for idx in range(0, n_frames, sample_rate):

        row = {
            "frame": int(frames[idx]),
        }

        for offset in offsets:

            j = idx + offset
            prefix = f"t{offset:+d}"

            if 0 <= j < n_frames:
                row[f"{prefix}_visible"] = int(visibility[j])
                row[f"{prefix}_x"] = float(px[j]) if not np.isnan(px[j]) else np.nan
                row[f"{prefix}_y"] = float(py[j]) if not np.isnan(py[j]) else np.nan
            else:
                row[f"{prefix}_visible"] = 0
                row[f"{prefix}_x"] = np.nan
                row[f"{prefix}_y"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


class VideoIterableDataset(IterableDataset):
    """Inference-only video dataset for sliding TrackNet windows."""

    def __init__(
        self,
        video_file,
        seq_len=8,
        sliding_step=1,
        bg_mode="",
        max_sample_num=50, # ADJUSTED LOWER BC INPUT VIDEOS CUT TO PURELY SINGLE RALLY VIDEO
        video_range=None,
        median=None,
    ):
        self.video_file = str(video_file)
        self.cap = cv2.VideoCapture(self.video_file)
        self.video_len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.cap.release()

        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.bg_mode = bg_mode
        self.median = None
        if self.bg_mode:
            self.median = (
                median
                if median is not None
                else self._generate_median(max_sample_num, video_range)
            )

    def __iter__(self):
        # Each raw frame is preprocessed once and cached; consecutive sliding
        # windows reuse seq_len-1 of those cached results.
        cap = cv2.VideoCapture(self.video_file)
        processed = deque(maxlen=self.seq_len)
        frame_ids = deque(maxlen=self.seq_len)
        next_id = 0

        while len(processed) < self.seq_len:
            success, frame = cap.read()
            if not success:
                break
            processed.append(self._process_single(frame))
            frame_ids.append(next_id)
            next_id += 1

        while processed:
            if len(processed) < self.seq_len:
                pad = self.seq_len - len(processed)
                window = list(processed) + [processed[-1]] * pad
                ids = list(frame_ids) + [frame_ids[-1]] * pad
                yield np.array([(0, i) for i in ids]), self._stack(window)
                break

            yield (
                np.array([(0, i) for i in frame_ids]),
                self._stack(list(processed)),
            )

            for _ in range(self.sliding_step):
                success, frame = cap.read()
                if success:
                    processed.append(self._process_single(frame))
                    frame_ids.append(next_id)
                    next_id += 1
                else:
                    processed.popleft()
                    frame_ids.popleft()

        cap.release()

    def _generate_median(self, max_sample_num, video_range):
        print("[TrackNetV3] Generating median image for background mode...")
        cap = cv2.VideoCapture(self.video_file)
        if video_range is None:
            start_frame, end_frame = 0, self.video_len
        else:
            start_frame = max(0, video_range[0] * self.fps)
            end_frame = min(video_range[1] * self.fps, self.video_len)

        video_segment_len = end_frame - start_frame
        sample_step = max(1, video_segment_len // max_sample_num)
        frame_list = []

        for frame_idx in range(start_frame, end_frame, sample_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if not success:
                break
            frame_list.append(frame)

        cap.release()
        if not frame_list:
            raise RuntimeError(f"Could not sample frames for median image: {self.video_file}")

        median = np.median(frame_list, axis=0)[..., ::-1]
        if self.bg_mode == "concat":
            median = Image.fromarray(median.astype("uint8"))
            median = np.array(median.resize(size=(TN_WIDTH, TN_HEIGHT)))
            median = np.moveaxis(median, -1, 0)

        print("[TrackNetV3] Median image generated.")
        return median

    def _process_single(self, frame_bgr):
        img_arr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_arr)

        if self.bg_mode == "subtract":
            diff = np.sum(np.absolute(img_arr - self.median), axis=2).astype("uint8")
            processed = np.array(
                Image.fromarray(diff).resize(size=(TN_WIDTH, TN_HEIGHT))
            )
            processed = processed.reshape(1, TN_HEIGHT, TN_WIDTH)
        elif self.bg_mode == "subtract_concat":
            diff = np.sum(np.absolute(img_arr - self.median), axis=2).astype("uint8")
            diff = np.array(Image.fromarray(diff).resize(size=(TN_WIDTH, TN_HEIGHT)))
            diff = diff.reshape(1, TN_HEIGHT, TN_WIDTH)
            resized = np.moveaxis(
                np.array(img.resize(size=(TN_WIDTH, TN_HEIGHT))), -1, 0
            )
            processed = np.concatenate((resized, diff), axis=0)
        else:
            processed = np.moveaxis(
                np.array(img.resize(size=(TN_WIDTH, TN_HEIGHT))), -1, 0
            )

        return processed.astype(np.float32)

    def _stack(self, window):
        frames = np.concatenate(window, axis=0)
        if self.bg_mode == "concat":
            frames = np.concatenate(
                (self.median.astype(np.float32), frames), axis=0
            )
        frames /= 255.0
        return frames


class CoordinateDataset(Dataset):
    """Inference-only coordinate dataset for InpaintNet."""

    def __init__(self, pred_dict, seq_len=16, sliding_step=1):
        self.pred_dict = pred_dict
        self.seq_len = seq_len
        self.sliding_step = sliding_step
        self.data = self._build_data()

    def _build_data(self):
        ids = []
        coordinates = []
        inpaint_masks = []

        x_pred = self.pred_dict["X"]
        y_pred = self.pred_dict["Y"]
        vis_pred = self.pred_dict["Visibility"]
        inpaint = self.pred_dict["Inpaint_Mask"]
        if not (len(x_pred) == len(y_pred) == len(vis_pred) == len(inpaint)):
            raise ValueError("Prediction coordinate and mask lengths do not match.")

        for start_idx in range(0, len(inpaint), self.sliding_step):
            idx_seq = []
            coor_seq = []
            mask_seq = []

            for frame_offset in range(self.seq_len):
                pred_idx = start_idx + frame_offset
                if pred_idx >= len(inpaint):
                    break
                idx_seq.append((0, pred_idx))
                coor_seq.append((x_pred[pred_idx], y_pred[pred_idx]))
                mask_seq.append(inpaint[pred_idx])

            if len(idx_seq) == self.seq_len:
                ids.append(idx_seq)
                coordinates.append(coor_seq)
                inpaint_masks.append(mask_seq)

        return {
            "id": np.array(ids, dtype=np.int32),
            "coor_pred": np.array(coordinates, dtype=np.float32),
            "inpaint_mask": np.array(inpaint_masks, dtype=np.float32),
        }

    def __len__(self):
        return len(self.data["id"])

    def __getitem__(self, idx):
        data_idx = self.data["id"][idx]
        coor_pred = self.data["coor_pred"][idx].copy()
        inpaint = self.data["inpaint_mask"][idx].reshape(-1, 1)

        width, height = self.pred_dict["Img_shape"]
        coor_pred[:, 0] = coor_pred[:, 0] / width
        coor_pred[:, 1] = coor_pred[:, 1] / height
        return data_idx, coor_pred, inpaint
