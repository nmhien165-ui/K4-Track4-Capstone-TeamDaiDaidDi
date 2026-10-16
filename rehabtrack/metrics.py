"""Knee-flexion metrics for VinMec RehabTrack (simulated demo, not a medical device).

Pure numpy math: no Streamlit, no model, no file IO.

Definitions (spec A3):
    interior angle  = angle between (hip - knee) and (ankle - knee), 0..180 degrees
    flexion_deg     = 180 - interior angle  (straight leg ~ 0 degrees)
    best_extension_flexion_deg = P10 of the smoothed valid flexion values
    observed_rom_deg           = P90 - P10 of the same values
    delta_extension_deg        = pre.best_extension - post.best_extension
        positive delta = the knee looked straighter in the SECOND session (post).
"""

from __future__ import annotations

import math

import numpy as np

# MVP conventions (NOT medical thresholds). Chosen to reject clips where the leg is cut
# off by the frame (the model then extrapolates keypoints); not clinically validated.
MIN_VALID_FRAMES = 10  # at 5 fps this is 2 s of usable data (was 5)
MIN_VALID_RATIO = 0.60  # share of sampled frames with hip+knee+ankle usable (was 0.50)
WARN_VALID_RATIO = 0.7
MIN_VECTOR_NORM = 1e-6
SMOOTHING_WINDOW = 3

_JOINTS = ("hip", "knee", "ankle")
_JOINT_LABELS_VI = {"hip": "hip (hông)", "knee": "knee (đầu gối)", "ankle": "ankle (cổ chân)"}

_SAGITTAL_WARNING = (
    "Confidence của keypoint không phát hiện được camera đặt lệch khỏi mặt phẳng dọc "
    "(sagittal); góc 2D có thể bị sai lệch do phép chiếu."
)
_UNQUANTIFIED_ERROR_WARNING = (
    "Sai số đo chưa được định lượng (không có ground truth từ thước đo góc/goniometer); "
    "không được hiểu là độ chính xác theo độ."
)
_SAME_SETUP_WARNING = (
    "Phép so sánh chỉ có ý nghĩa khi hai video cùng một người, cùng một chân và cùng "
    "vị trí đặt camera."
)


# --------------------------------------------------------------------- coercion


def _to_float(value) -> float | None:
    """float(value) when it is a finite number, else None (never raises)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _to_point(point) -> tuple[float, float] | None:
    """(x, y) floats, or None for None / wrong shape / non-numeric / NaN."""
    if point is None:
        return None
    try:
        x_raw, y_raw = point
    except (TypeError, ValueError):
        return None
    x = _to_float(x_raw)
    y = _to_float(y_raw)
    if x is None or y is None:
        return None
    return (x, y)


# ---------------------------------------------------------------------- geometry


def interior_angle_deg(hip, knee, ankle) -> float | None:
    """Angle at the knee between (hip - knee) and (ankle - knee), in [0, 180] degrees.
    None when any point is missing/NaN or a vector is (almost) zero - never 0."""
    hip_p, knee_p, ankle_p = _to_point(hip), _to_point(knee), _to_point(ankle)
    if hip_p is None or knee_p is None or ankle_p is None:
        return None
    v1 = np.array([hip_p[0] - knee_p[0], hip_p[1] - knee_p[1]], dtype=float)
    v2 = np.array([ankle_p[0] - knee_p[0], ankle_p[1] - knee_p[1]], dtype=float)
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 < MIN_VECTOR_NORM or norm2 < MIN_VECTOR_NORM:
        return None
    cosine = float(np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def flexion_deg_from_points(hip, knee, ankle) -> float | None:
    """flexion = 180 - interior angle, always within [0, 180]; None if not computable."""
    interior = interior_angle_deg(hip, knee, ankle)
    if interior is None:
        return None
    return float(min(180.0, max(0.0, 180.0 - interior)))


# ----------------------------------------------------------------------- session


def _frame_flexion(frame, min_conf: float) -> float | None:
    """Flexion for one frame record; None if a joint or its confidence is unusable."""
    if not isinstance(frame, dict):
        return None
    for joint in _JOINTS:
        if _to_point(frame.get(joint)) is None:
            return None
        conf = _to_float(frame.get(f"{joint}_conf"))
        if conf is None or conf < min_conf:
            return None
    return flexion_deg_from_points(frame["hip"], frame["knee"], frame["ankle"])


def _edge_rejected_names(frame) -> list[str]:
    """Joint names that pose.py excluded from this frame because they sit at the frame
    border. Tolerates older frame dicts without the key and malformed values."""
    if not isinstance(frame, dict):
        return []
    names = frame.get("edge_rejected")
    if not isinstance(names, (list, tuple)):
        return []
    return [name for name in names if name in _JOINTS]


def _edge_rejection_summary(frame_list: list) -> tuple[int, list[str]]:
    """(number of frames with edge-rejected joints, joint names involved in
    hip/knee/ankle order)."""
    frames_affected = 0
    joints_seen: set[str] = set()
    for frame in frame_list:
        names = _edge_rejected_names(frame)
        if names:
            frames_affected += 1
            joints_seen.update(names)
    return frames_affected, [joint for joint in _JOINTS if joint in joints_seen]


def _median_smooth_runs(values: list[float | None]) -> list[float | None]:
    """Median window 3 over runs of CONSECUTIVE valid values only.

    None stays None and nothing is interpolated across a gap. The first and last value
    of each run have no full window and are kept unchanged.
    """
    smoothed: list[float | None] = list(values)
    index = 0
    count = len(values)
    while index < count:
        if values[index] is None:
            index += 1
            continue
        end = index
        while end + 1 < count and values[end + 1] is not None:
            end += 1
        for position in range(index + 1, end):  # interior of the run only
            window = values[position - 1 : position + 2]
            smoothed[position] = float(np.median(window))
        index = end + 1
    return smoothed


def analyze_session(frames: list[dict], min_conf: float = 0.40) -> dict:
    """Angles, P10/P90 and quality for one video. Never raises on bad input."""
    try:
        frame_list = list(frames) if frames is not None else []
    except TypeError:
        frame_list = []

    time_s = [_to_float(f.get("time_s")) if isinstance(f, dict) else None for f in frame_list]
    raw = [_frame_flexion(f, min_conf) for f in frame_list]
    smoothed = _median_smooth_runs(raw)

    total = len(frame_list)
    valid_values = [v for v in smoothed if v is not None]
    valid_count = len(valid_values)
    valid_ratio = (valid_count / total) if total else 0.0

    edge_frames, edge_joints = _edge_rejection_summary(frame_list)
    edge_joint_text = ", ".join(_JOINT_LABELS_VI[joint] for joint in edge_joints)

    warnings: list[str] = []
    # MVP convention thresholds, not medical ones.
    insufficient = valid_count < MIN_VALID_FRAMES or valid_ratio < MIN_VALID_RATIO
    if insufficient:
        if total == 0:
            warnings.append("Không có frame nào để phân tích.")
        message = (
            f"Thiếu dữ liệu: chỉ {valid_count}/{total} frame có đủ keypoint hợp lệ "
            f"(cần ít nhất {MIN_VALID_FRAMES} frame và {MIN_VALID_RATIO:.0%} tổng số frame; "
            "ngưỡng quy ước MVP, không phải ngưỡng y khoa). Không đưa ra số đo."
        )
        if edge_frames:
            message += (
                f" Nguyên nhân: {edge_frames} frame bị loại vì khớp {edge_joint_text} "
                "nằm ở mép khung hình."
            )
        warnings.append(message)
    elif valid_ratio < WARN_VALID_RATIO:
        warnings.append(
            f"Chỉ {valid_ratio:.0%} frame có đủ keypoint (dưới {WARN_VALID_RATIO:.0%}, "
            "ngưỡng quy ước MVP); số đo kém ổn định."
        )
    if edge_frames:
        warnings.append(
            f"{edge_frames}/{total} frame bị loại một phần dữ liệu vì khớp {edge_joint_text} "
            "nằm sát hoặc ngoài mép khung hình (keypoint ở mép là do mô hình ngoại suy, "
            "không phải quan sát thật). Hãy đặt camera/đối tượng sao cho hip, knee và bàn "
            "chân nằm trọn trong khung hình."
        )
    warnings.append(_SAGITTAL_WARNING)

    if insufficient:
        best_extension = None
        observed_rom = None
    else:
        p10 = float(np.percentile(valid_values, 10))
        p90 = float(np.percentile(valid_values, 90))
        best_extension = p10
        observed_rom = p90 - p10

    return {
        "status": "insufficient_data" if insufficient else "ok",
        "time_s": time_s,
        "flexion_deg": smoothed,
        "flexion_raw_deg": raw,
        "best_extension_flexion_deg": best_extension,
        "observed_rom_deg": observed_rom,
        "valid_ratio": float(valid_ratio),
        "valid_frame_count": valid_count,
        "total_frame_count": total,
        "valid_frames": valid_count,  # alias of valid_frame_count (two specs, two names)
        "total_frames": total,  # alias of total_frame_count
        "warnings": warnings,
    }


# ---------------------------------------------------------------------- compare


def _usable_p10(session) -> float | None:
    """P10 of a session result, only when its status is ok and the value is finite."""
    if not isinstance(session, dict) or session.get("status") != "ok":
        return None
    return _to_float(session.get("best_extension_flexion_deg"))


def _interpret_delta(delta: float | None) -> str:
    if delta is None:
        return "Chưa đủ dữ liệu hợp lệ để so sánh hai video; không đưa ra số đo."
    shown = round(delta, 1)  # wording follows the number actually displayed
    if shown == 0:
        return "Góc gập tại vùng duỗi tối đa không đổi giữa hai video mô phỏng."
    if delta > 0:
        return f"Góc gập tại vùng duỗi tối đa giảm khoảng {delta:.1f}° giữa hai video mô phỏng."
    return f"Góc gập tại vùng duỗi tối đa tăng khoảng {abs(delta):.1f}° giữa hai video mô phỏng."


def compare_sessions(pre: dict, post: dict) -> dict:
    """Sign-only comparison of two analyze_session results. No clinical conclusion,
    no threshold, no percentage. delta = pre P10 - post P10; positive means the knee
    looked straighter in the second video."""
    pre_p10 = _usable_p10(pre)
    post_p10 = _usable_p10(post)

    warnings: list[str] = []
    if pre_p10 is None:
        warnings.append("Video thứ nhất (trước) chưa đủ dữ liệu hợp lệ.")
    if post_p10 is None:
        warnings.append("Video thứ hai (sau) chưa đủ dữ liệu hợp lệ.")
    warnings.append(_UNQUANTIFIED_ERROR_WARNING)
    warnings.append(_SAME_SETUP_WARNING)

    if pre_p10 is None or post_p10 is None:
        delta = None
        status = "insufficient_data"
    else:
        delta = float(pre_p10 - post_p10)
        status = "ok"

    return {
        "status": status,
        "delta_extension_deg": delta,
        "interpretation": _interpret_delta(delta),
        "warnings": warnings,
    }
