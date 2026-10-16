"""Video -> sampled COCO hip/knee/ankle points (PLAN C3).

No angle calculation, UI, tracking, or output files. Missing observations stay
None; confidence filtering for measurements belongs to metrics.py.
"""

from functools import lru_cache
from math import isfinite
from pathlib import Path
from threading import Lock

import cv2
import numpy as np


_JOINTS = {"left": (11, 13, 15), "right": (12, 14, 16)}
_MODEL_LOCK = Lock()


@lru_cache(maxsize=1)
def _get_model():
    # Lazy loading lets callers validate input without downloading weights.
    from ultralytics import YOLO

    return YOLO(str(Path(__file__).with_name("yolo11n-pose.pt")))


def _empty_frame(time_s: float) -> dict:
    return {
        "time_s": float(time_s),
        "hip": None,
        "knee": None,
        "ankle": None,
        "hip_conf": 0.0,
        "knee_conf": 0.0,
        "ankle_conf": 0.0,
    }


def _extract_points(result, side: str, time_s: float) -> dict:
    record = _empty_frame(time_s)
    if result.boxes is None or result.keypoints is None:
        return record
    boxes = result.boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return record
    sizes = np.maximum(boxes[:, 2:] - boxes[:, :2], 0)
    areas = sizes[:, 0] * sizes[:, 1]
    areas = np.where(np.isfinite(areas), areas, 0)
    person = int(np.argmax(areas))
    if areas[person] <= 0:
        return record

    # Use raw data: some library versions mask low-confidence xy to zero.
    points = result.keypoints.data.cpu().numpy()
    if points.ndim != 3 or person >= len(points) or points.shape[2] < 3:
        return record
    for name, index in zip(("hip", "knee", "ankle"), _JOINTS[side]):
        if index >= points.shape[1]:
            continue
        x, y, confidence = (float(v) for v in points[person, index, :3])
        if (
            not all(isfinite(v) for v in (x, y, confidence))
            or not 0 < confidence <= 1
            or x < 0 or y < 0
            or (x == 0 and y == 0)
        ):
            continue
        record[name] = [x, y]
        record[f"{name}_conf"] = confidence
    return record


def extract_video_pose(
    video_path: str,
    side: str = "right",
    target_fps: float = 5.0,
) -> list[dict]:
    """Return one C3 record per sampled frame, including missing detections.

    Samples frame zero then the first frame at/after each target time, without
    duplicating frames when target_fps exceeds source FPS. Uses frame_idx/FPS;
    if FPS is unavailable, requires increasing decoder timestamps instead of
    inventing a frame rate. Raises ValueError for invalid/unreadable inputs,
    RuntimeError for model failures. The caller can show these errors in UI.
    """
    if side not in _JOINTS:
        raise ValueError("side must be 'left' or 'right'.")
    try:
        target_fps = float(target_fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("target_fps must be finite and greater than zero.") from exc
    if not isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be finite and greater than zero.")
    path = Path(video_path)
    if not path.is_file():
        raise ValueError(f"Video file does not exist: {path}")

    capture = cv2.VideoCapture(str(path))
    records = []
    frame_idx = 0
    next_sample = 0.0
    previous_time = -1.0
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        has_fps = isfinite(fps) and fps > 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if has_fps:
                time_s = frame_idx / fps
            else:
                time_s = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if not isfinite(time_s) or time_s < 0 or time_s <= previous_time:
                    raise ValueError("Video has no usable FPS or increasing timestamps.")
            previous_time = time_s
            frame_idx += 1
            if time_s + 1e-9 < next_sample:
                continue
            try:
                # One cached predictor can be called by multiple Streamlit sessions.
                with _MODEL_LOCK:
                    result = _get_model().predict(frame, verbose=False, save=False)[0]
            except Exception as exc:
                raise RuntimeError(
                    "YOLO pose inference failed. Check ultralytics/torch installation "
                    "and access to yolo11n-pose.pt weights."
                ) from exc
            records.append(_extract_points(result, side, time_s))
            next_sample = (int((time_s + 1e-9) * target_fps) + 1) / target_fps
        if frame_idx == 0:
            raise ValueError(f"Video contains no decodable frames: {path}")
        return records
    finally:
        capture.release()
