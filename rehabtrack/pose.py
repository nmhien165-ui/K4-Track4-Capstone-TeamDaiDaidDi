"""Pose extraction for VinMec RehabTrack (simulated demo, not a medical device).

Runs a FROZEN pretrained Ultralytics YOLO11n-pose model over a video and returns one
record per SAMPLED frame with the hip / knee / ankle of one leg. No training, no
fine-tuning, no tracking. This module does not compute angles or ROM (see metrics.py),
does not call Streamlit and does not write files.

Record contract (one dict per sampled frame):
    time_s      float          frame_idx / video_fps (real timestamp, never rescaled)
    hip, knee, ankle           [x, y] image pixels (x right, y down) or None
    hip_conf, knee_conf, ankle_conf   float, 0.0 whenever the joint is None
    edge_rejected  list[str]   OPTIONAL EXTRA key: joints dropped because they sit at the
                               frame border (see _reject_edge_joints); [] when none.

Edge-of-frame rule: a keypoint at or beyond the image border is extrapolated by the model
(the leg is cut off by the frame), not observed. Such joints become None with conf 0.0.
"""

from __future__ import annotations

import math
import os
import threading
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

# COCO-17 keypoint indices (hip, knee, ankle) per anatomical side.
_SIDE_INDICES = {
    "left": (11, 13, 15),
    "right": (12, 14, 16),
}

_HERE = Path(__file__).resolve().parent
_DEFAULT_WEIGHTS = _HERE / "models" / "yolo11n-pose.pt"
_FALLBACK_MODEL_NAME = "yolo11n-pose.pt"  # plain name: Ultralytics auto-downloads it
_ENV_MODEL_VAR = "REHABTRACK_MODEL"

# MVP convention (NOT a medical threshold): joints closer than this fraction of the frame
# width/height to the border are treated as not observed. 0 disables the rule.
DEFAULT_EDGE_MARGIN = 0.03
_JOINT_NAMES = ("hip", "knee", "ankle")

# The UI may run several sessions (threads) in one process:
#  - _load_model is lru_cache'd on the weights path, so the model is loaded once per path
#    (clear it with _load_model.cache_clear(), e.g. in tests);
#  - _LOAD_LOCK stops two threads that both miss the cache from loading the weights twice;
#  - _INFERENCE_LOCK serialises every model.predict call (one shared model is not assumed
#    to be safe for concurrent inference).
_LOAD_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.Lock()


# --------------------------------------------------------------------------- model


def _resolve_model_path() -> str:
    """Env var REHABTRACK_MODEL > <dir of pose.py>/models/yolo11n-pose.pt > plain name."""
    env_value = os.environ.get(_ENV_MODEL_VAR, "").strip()
    if env_value:
        return env_value
    if _DEFAULT_WEIGHTS.exists():
        return str(_DEFAULT_WEIGHTS)
    return _FALLBACK_MODEL_NAME


def get_model_name() -> str:
    """Basename of the weights actually used (or that will be used) - for reports."""
    return os.path.basename(_resolve_model_path())


def _get_device() -> str:
    """'cuda:0' if torch sees a GPU, else 'cpu'. torch is imported lazily."""
    try:
        import torch  # noqa: WPS433 (lazy on purpose)

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:  # torch missing or broken -> CPU
        pass
    return "cpu"


@lru_cache(maxsize=2)
def _load_model(weights_path: str):
    """Load the Ultralytics model for `weights_path`; cached per path (call
    `_load_model.cache_clear()` to reset). ultralytics is imported lazily so
    `import pose` works even when ultralytics is not installed. A failed load is not
    cached, so the next call retries."""
    from ultralytics import YOLO  # noqa: WPS433 (lazy on purpose)

    return YOLO(weights_path)


def _get_model():
    """The cached model for the currently resolved weights path (loaded on first use)."""
    weights_path = _resolve_model_path()
    with _LOAD_LOCK:
        return _load_model(weights_path)


def _inference_error(weights_path: str) -> RuntimeError:
    """The single error callers see when loading the model or running predict fails.
    Raise it `from exc` so the original cause stays attached."""
    return RuntimeError(
        "YOLO pose inference failed. Check the ultralytics/torch installation and "
        f"access to the weights file {weights_path}."
    )


# ------------------------------------------------------------------- pure helpers


def _sample_stride(fps: float, target_fps: float) -> int:
    """Every how many video frames to sample: round(fps / target_fps), minimum 1.

    Raises ValueError for fps or target_fps that is not a finite positive number;
    an fps is never invented.
    """
    try:
        fps_value = float(fps)
    except (TypeError, ValueError):
        raise ValueError(f"FPS của video không hợp lệ: {fps!r}") from None
    if not math.isfinite(fps_value) or fps_value <= 0:
        raise ValueError(
            f"Không đọc được FPS hợp lệ từ video (nhận được {fps_value}); "
            "không tự đoán FPS."
        )
    try:
        target_value = float(target_fps)
    except (TypeError, ValueError):
        raise ValueError(f"target_fps không hợp lệ: {target_fps!r}") from None
    if not math.isfinite(target_value) or target_value <= 0:
        raise ValueError(f"target_fps phải là số dương hữu hạn (nhận được {target_value}).")
    return max(1, int(round(fps_value / target_value)))


def _select_largest_person(boxes_xyxy) -> int | None:
    """Index of the detection with the largest bounding-box area, or None if there is
    no usable box. Boxes are [x1, y1, x2, y2]; non-finite boxes are skipped."""
    if boxes_xyxy is None:
        return None
    boxes = np.asarray(boxes_xyxy, dtype=float)
    if boxes.ndim != 2 or boxes.shape[0] == 0 or boxes.shape[1] < 4:
        return None
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    areas = np.where(np.isfinite(areas), areas, -np.inf)
    best = int(np.argmax(areas))
    if not np.isfinite(areas[best]):
        return None
    return best


def _empty_joints() -> dict:
    return {
        "hip": None,
        "knee": None,
        "ankle": None,
        "hip_conf": 0.0,
        "knee_conf": 0.0,
        "ankle_conf": 0.0,
    }


def _joint_or_none(kpts_xy, kpts_conf, index: int):
    """([x, y], conf) for one keypoint, or (None, 0.0) when it is not usable:
    conf <= 0 / non-finite, coords non-finite, or coords exactly (0, 0) (Ultralytics
    writes (0, 0) for undetected keypoints)."""
    try:
        x = float(kpts_xy[index][0])
        y = float(kpts_xy[index][1])
        conf = float(kpts_conf[index])
    except (TypeError, ValueError, IndexError):
        return None, 0.0
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(conf)):
        return None, 0.0
    if conf <= 0.0:
        return None, 0.0
    if x == 0.0 and y == 0.0:
        return None, 0.0
    return [x, y], conf


def _joints_for_side(kpts_xy, kpts_conf, side: str) -> dict:
    """Map one person's 17 COCO keypoints to the six hip/knee/ankle contract fields."""
    if side not in _SIDE_INDICES:
        raise ValueError(f"side phải là 'left' hoặc 'right', nhận được {side!r}")
    result = _empty_joints()
    if kpts_xy is None or kpts_conf is None:
        return result
    for name, index in zip(_JOINT_NAMES, _SIDE_INDICES[side]):
        point, conf = _joint_or_none(kpts_xy, kpts_conf, index)
        result[name] = point
        result[f"{name}_conf"] = conf
    return result


def _check_edge_margin(edge_margin) -> float:
    """Validated edge margin: a finite number in [0, 0.5). 0 disables the rule; 0.5 or
    more would reject every joint of the frame, so it is an error, not a setting."""
    try:
        margin = float(edge_margin)
    except (TypeError, ValueError):
        raise ValueError(f"edge_margin không hợp lệ: {edge_margin!r}") from None
    if not math.isfinite(margin) or margin < 0.0 or margin >= 0.5:
        raise ValueError(
            f"edge_margin phải nằm trong [0, 0.5) (0 = tắt); nhận được {margin}."
        )
    return margin


def _reject_edge_joints(joints: dict, width, height, margin) -> tuple[dict, list[str]]:
    """Drop joints that sit at the frame border. Pure: no model, no video.

    A joint (hip, knee or ankle) is rejected - set to None with conf 0.0 - when
    x < margin*W, x > W - margin*W, y < margin*H or y > H - margin*H (a point exactly
    on the margin line is kept). `margin` is a fraction of the width/height; 0 disables.
    Reason: keypoints at or outside the image border are extrapolated by the model.

    Returns (new joints dict, names of the joints rejected by this rule). The input dict
    is not modified; joints that are already None stay None and are not listed.
    """
    cleaned = dict(joints)
    rejected: list[str] = []
    try:
        margin_value = float(margin)
        width_value = float(width)
        height_value = float(height)
    except (TypeError, ValueError):
        return cleaned, rejected
    if not all(math.isfinite(v) for v in (margin_value, width_value, height_value)):
        return cleaned, rejected
    if margin_value <= 0.0 or width_value <= 0.0 or height_value <= 0.0:
        return cleaned, rejected

    x_min = margin_value * width_value
    x_max = width_value - x_min
    y_min = margin_value * height_value
    y_max = height_value - y_min
    for name in _JOINT_NAMES:
        point = cleaned.get(name)
        if point is None:
            continue
        x, y = point
        if x < x_min or x > x_max or y < y_min or y > y_max:
            cleaned[name] = None
            cleaned[f"{name}_conf"] = 0.0
            rejected.append(name)
    return cleaned, rejected


def _to_numpy(value):
    """Torch tensor (possibly on GPU) or array-like -> numpy array; None stays None."""
    if value is None:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _record_from_detections(
    time_s: float,
    boxes_xyxy,
    kpts_xy,
    kpts_conf,
    side: str,
    width=None,
    height=None,
    edge_margin: float = 0.0,
) -> dict:
    """Build one contract record from raw per-frame detections (all persons).

    `width`/`height` are the DECODED frame size in pixels; with edge_margin > 0 the joints
    at the frame border are dropped (see _reject_edge_joints) and listed in "edge_rejected".
    """
    record = {"time_s": float(time_s)}
    record.update(_empty_joints())
    record["edge_rejected"] = []
    person = _select_largest_person(boxes_xyxy)
    if person is None or kpts_xy is None or kpts_conf is None:
        return record
    if person >= len(kpts_xy) or person >= len(kpts_conf):
        return record
    joints = _joints_for_side(kpts_xy[person], kpts_conf[person], side)
    joints, rejected = _reject_edge_joints(joints, width, height, edge_margin)
    record.update(joints)
    record["edge_rejected"] = rejected
    return record


def _record_from_result(
    time_s: float, result, side: str, width=None, height=None, edge_margin: float = 0.0
) -> dict:
    """Convert one Ultralytics Results object into a contract record."""
    boxes = getattr(result, "boxes", None)
    keypoints = getattr(result, "keypoints", None)
    boxes_xyxy = _to_numpy(getattr(boxes, "xyxy", None)) if boxes is not None else None
    kpts_xy = _to_numpy(getattr(keypoints, "xy", None)) if keypoints is not None else None
    kpts_conf = _to_numpy(getattr(keypoints, "conf", None)) if keypoints is not None else None
    return _record_from_detections(
        time_s, boxes_xyxy, kpts_xy, kpts_conf, side, width, height, edge_margin
    )


# ------------------------------------------------------------------------ public


def extract_video_pose(
    video_path: str,
    side: str = "right",
    target_fps: float = 5.0,
    *,
    edge_margin: float = DEFAULT_EDGE_MARGIN,
) -> list[dict]:
    """Return one record per SAMPLED frame (frames without a usable person included).

    `side` ("left" or "right") is the subject's anatomical leg as LABELLED BY THE
    MODEL (COCO left = indices 11/13/15, right = 12/14/16). It is not guaranteed to be
    the same as the screen side, especially for mirrored or frontal footage.

    Exactly one person is used per frame: the detection with the largest bounding-box
    area (no tracking). Time stamps are real: time_s = frame_idx / video_fps; sampling
    every round(video_fps / target_fps) frames never rescales time.

    `edge_margin` (fraction of the DECODED frame width/height, default 0.03, 0 = off):
    hip/knee/ankle closer to the border than that are set to None (conf 0.0) and named
    in the record's "edge_rejected" list, because the model extrapolates keypoints at or
    beyond the image border instead of observing them.

    Raises:
        ValueError: invalid `side` / `edge_margin`, unreadable/empty video, or no valid FPS.
        FileNotFoundError: `video_path` does not exist.
        RuntimeError: loading the YOLO model or running predict failed (original exception
            kept as `__cause__`). Input errors above are never wrapped in it.
    """
    if side not in _SIDE_INDICES:
        raise ValueError(f"side phải là 'left' hoặc 'right', nhận được {side!r}")
    margin = _check_edge_margin(edge_margin)
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Không tìm thấy file video: {video_path}")

    capture = cv2.VideoCapture(video_path)
    try:
        if not capture.isOpened():
            raise ValueError(f"Không đọc được video (file hỏng hoặc codec không hỗ trợ): {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        stride = _sample_stride(fps, target_fps)  # ValueError when fps is 0/NaN

        records: list[dict] = []
        frame_idx = 0
        device = None
        model = None
        weights_path = None
        while capture.grab():
            if frame_idx % stride == 0:
                ok, frame = capture.retrieve()
                if ok and frame is not None:
                    if model is None:
                        weights_path = _resolve_model_path()
                        try:
                            model = _get_model()
                            device = _get_device()
                        except Exception as exc:  # KeyboardInterrupt is not an Exception
                            raise _inference_error(weights_path) from exc
                    try:
                        with _INFERENCE_LOCK:
                            results = model.predict(frame, device=device, verbose=False)
                    except Exception as exc:
                        raise _inference_error(weights_path) from exc
                    time_s = frame_idx / float(fps)
                    result = results[0] if len(results) else None
                    frame_height, frame_width = frame.shape[:2]  # decoded (rotation-aware)
                    records.append(
                        _record_from_result(
                            time_s, result, side, frame_width, frame_height, margin
                        )
                    )
            frame_idx += 1
    finally:
        capture.release()

    if not records:
        raise ValueError(f"Video rỗng hoặc không giải mã được frame nào: {video_path}")
    return records
