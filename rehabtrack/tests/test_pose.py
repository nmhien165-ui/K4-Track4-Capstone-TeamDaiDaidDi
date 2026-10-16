"""Tests for pose.py. Pure helpers and mock-based tests need NO ultralytics and run no
real inference. The single integration test is skipped unless ultralytics and the
weights file are available."""

import contextlib
import math
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pose  # noqa: E402  (must work without ultralytics installed)

# Captured at import: tests monkeypatch pose._load_model, but the cache that must be reset
# between tests is the real lru_cache object.
_REAL_LOAD_MODEL = pose._load_model


@pytest.fixture(autouse=True)
def _fresh_model_cache():
    """Every test starts and ends with an empty model cache (tests stay independent)."""
    _REAL_LOAD_MODEL.cache_clear()
    yield
    _REAL_LOAD_MODEL.cache_clear()


REQUIRED_KEYS = {"time_s", "hip", "knee", "ankle", "hip_conf", "knee_conf", "ankle_conf"}
RECORD_KEYS = REQUIRED_KEYS | {"edge_rejected"}  # the ONE optional extra key


def make_keypoints(overrides=None, conf=0.9):
    """17 COCO keypoints with distinct non-zero coordinates and a uniform confidence."""
    xy = np.array([[10.0 + i, 20.0 + 2 * i] for i in range(17)])
    confs = np.full(17, conf)
    for index, (x, y, c) in (overrides or {}).items():
        xy[index] = [x, y]
        confs[index] = c
    return xy, confs


# ------------------------------------------------------------------ _sample_stride


def test_sample_stride_rounds_and_has_minimum_one():
    assert pose._sample_stride(30.0, 5.0) == 6
    assert pose._sample_stride(15.0, 5.0) == 3
    assert pose._sample_stride(29.97, 5.0) == 6
    assert pose._sample_stride(5.0, 5.0) == 1
    assert pose._sample_stride(2.0, 5.0) == 1  # video slower than target: never 0


@pytest.mark.parametrize("bad_fps", [0, 0.0, -5.0, float("nan"), float("inf"), None, "abc"])
def test_sample_stride_rejects_invalid_fps(bad_fps):
    with pytest.raises(ValueError):
        pose._sample_stride(bad_fps, 5.0)


@pytest.mark.parametrize("bad_target", [0, -1.0, float("nan")])
def test_sample_stride_rejects_invalid_target_fps(bad_target):
    with pytest.raises(ValueError):
        pose._sample_stride(30.0, bad_target)


# ------------------------------------------------------- _select_largest_person


def test_select_largest_person_by_box_area():
    boxes = [[0, 0, 10, 10], [0, 0, 50, 20], [5, 5, 30, 30]]  # areas 100, 1000, 625
    assert pose._select_largest_person(boxes) == 1
    assert pose._select_largest_person(np.array(boxes, dtype=float)) == 1


def test_select_largest_person_handles_empty_and_bad_boxes():
    assert pose._select_largest_person([]) is None
    assert pose._select_largest_person(None) is None
    assert pose._select_largest_person(np.zeros((0, 4))) is None
    assert pose._select_largest_person([[float("nan")] * 4]) is None
    assert pose._select_largest_person([[float("nan")] * 4, [0, 0, 2, 2]]) == 1


# -------------------------------------------------------------- _joints_for_side


def test_joints_for_side_index_mapping():
    xy, conf = make_keypoints()
    right = pose._joints_for_side(xy, conf, "right")
    assert right["hip"] == [10.0 + 12, 20.0 + 24]
    assert right["knee"] == [10.0 + 14, 20.0 + 28]
    assert right["ankle"] == [10.0 + 16, 20.0 + 32]
    left = pose._joints_for_side(xy, conf, "left")
    assert left["hip"] == [10.0 + 11, 20.0 + 22]
    assert left["knee"] == [10.0 + 13, 20.0 + 26]
    assert left["ankle"] == [10.0 + 15, 20.0 + 30]
    for joints in (left, right):
        assert set(joints) == REQUIRED_KEYS - {"time_s"}
        assert joints["hip_conf"] == pytest.approx(0.9)


def test_joints_zero_zero_and_low_conf_become_none_with_zero_conf():
    xy, conf = make_keypoints(
        {
            14: (0.0, 0.0, 0.8),  # right knee at (0,0): undetected placeholder
            16: (50.0, 60.0, 0.0),  # right ankle with conf 0
            12: (30.0, 40.0, -0.1),  # right hip with negative conf
        }
    )
    joints = pose._joints_for_side(xy, conf, "right")
    assert joints["knee"] is None and joints["knee_conf"] == 0.0
    assert joints["ankle"] is None and joints["ankle_conf"] == 0.0
    assert joints["hip"] is None and joints["hip_conf"] == 0.0


def test_joints_nan_and_missing_inputs_become_none():
    xy, conf = make_keypoints({14: (float("nan"), 5.0, 0.9)})
    joints = pose._joints_for_side(xy, conf, "right")
    assert joints["knee"] is None and joints["knee_conf"] == 0.0
    assert joints["hip"] is not None  # unaffected joints survive
    empty = pose._joints_for_side(None, None, "left")
    assert empty["hip"] is None and empty["ankle_conf"] == 0.0


def test_invalid_side_raises_value_error():
    xy, conf = make_keypoints()
    with pytest.raises(ValueError):
        pose._joints_for_side(xy, conf, "middle")
    with pytest.raises(ValueError):
        pose.extract_video_pose("does_not_matter.mp4", side="front")


# --------------------------------------------------------- per-frame record builder


def test_record_uses_largest_person_and_no_person_gives_all_none():
    xy_small, conf_small = make_keypoints(conf=0.9)
    xy_big, conf_big = make_keypoints({14: (111.0, 222.0, 0.77)}, conf=0.9)
    boxes = [[0, 0, 10, 10], [0, 0, 100, 200]]
    record = pose._record_from_detections(
        0.4, boxes, np.stack([xy_small, xy_big]), np.stack([conf_small, conf_big]), "right"
    )
    assert set(record) == RECORD_KEYS
    assert record["time_s"] == pytest.approx(0.4)
    assert record["knee"] == [111.0, 222.0]
    assert record["knee_conf"] == pytest.approx(0.77)

    none_record = pose._record_from_detections(0.6, np.zeros((0, 4)), None, None, "right")
    assert set(none_record) == RECORD_KEYS
    assert none_record["hip"] is None and none_record["knee"] is None and none_record["ankle"] is None
    assert none_record["hip_conf"] == none_record["knee_conf"] == none_record["ankle_conf"] == 0.0
    assert none_record["edge_rejected"] == []


def test_record_from_detections_applies_edge_rule_only_when_size_and_margin_given():
    xy, conf = make_keypoints({16: (50.0, 199.0, 0.9)})  # right ankle at 99.5% of H=200
    args = ([[0, 0, 90, 190]], np.stack([xy]), np.stack([conf]), "right")
    without = pose._record_from_detections(0.0, *args)  # no frame size, margin 0
    assert without["ankle"] == [50.0, 199.0] and without["edge_rejected"] == []
    with_rule = pose._record_from_detections(0.0, *args, width=100, height=200, edge_margin=0.03)
    assert with_rule["ankle"] is None and with_rule["ankle_conf"] == 0.0
    assert with_rule["edge_rejected"] == ["ankle"]
    assert with_rule["hip"] == [22.0, 44.0]  # interior joints untouched


# ----------------------------------------------------- _reject_edge_joints (pure)

# Frame 100 wide x 200 high, margin 0.03 -> keep x in [3, 97], y in [6, 194].
_W, _H, _MARGIN = 100, 200, 0.03


def _joints(**overrides):
    joints = {
        "hip": [50.0, 100.0],
        "knee": [50.0, 120.0],
        "ankle": [50.0, 150.0],
        "hip_conf": 0.9,
        "knee_conf": 0.8,
        "ankle_conf": 0.7,
    }
    joints.update(overrides)
    return joints


def test_edge_helper_keeps_interior_joints_untouched():
    joints = _joints()
    cleaned, rejected = pose._reject_edge_joints(joints, _W, _H, _MARGIN)
    assert rejected == []
    assert cleaned == joints
    assert cleaned is not joints  # pure: returns a copy


@pytest.mark.parametrize(
    "point",
    [
        [3.0, 100.0],  # exactly on the left margin line
        [97.0, 100.0],  # exactly on the right margin line
        [50.0, 6.0],  # exactly on the top margin line
        [50.0, 194.0],  # exactly on the bottom margin line
        [3.0, 6.0],  # top-left margin corner
        [97.0, 194.0],  # bottom-right margin corner
    ],
)
def test_edge_helper_keeps_point_exactly_on_margin(point):
    cleaned, rejected = pose._reject_edge_joints(_joints(ankle=point), _W, _H, _MARGIN)
    assert rejected == []
    assert cleaned["ankle"] == point and cleaned["ankle_conf"] == 0.7


@pytest.mark.parametrize(
    "point",
    [
        [2.9, 100.0],  # left of the margin
        [97.1, 100.0],  # right of the margin
        [50.0, 5.9],  # above the margin
        [50.0, 194.1],  # below the margin
        [-5.0, 100.0],  # outside the image (negative x)
        [50.0, 250.0],  # outside the image (below the bottom edge)
        [150.0, 100.0],  # outside the image (right of the right edge)
        [50.0, -1.0],  # outside the image (above the top edge)
    ],
)
def test_edge_helper_rejects_point_outside_margin_on_each_side(point):
    cleaned, rejected = pose._reject_edge_joints(_joints(ankle=point), _W, _H, _MARGIN)
    assert rejected == ["ankle"]
    assert cleaned["ankle"] is None and cleaned["ankle_conf"] == 0.0
    assert cleaned["hip"] == [50.0, 100.0] and cleaned["hip_conf"] == 0.9  # others kept
    assert cleaned["knee"] == [50.0, 120.0] and cleaned["knee_conf"] == 0.8


def test_edge_helper_rejects_each_joint_name_and_lists_them_in_order():
    joints = _joints(hip=[1.0, 100.0], knee=[50.0, 199.0], ankle=[99.0, 100.0])
    cleaned, rejected = pose._reject_edge_joints(joints, _W, _H, _MARGIN)
    assert rejected == ["hip", "knee", "ankle"]
    for name in ("hip", "knee", "ankle"):
        assert cleaned[name] is None and cleaned[f"{name}_conf"] == 0.0


def test_edge_helper_margin_zero_disables_the_rule():
    joints = _joints(ankle=[50.0, 250.0], hip=[-5.0, 100.0])  # even outside the image
    cleaned, rejected = pose._reject_edge_joints(joints, _W, _H, 0.0)
    assert rejected == []
    assert cleaned == joints


def test_edge_helper_margin_scales_with_width_and_height_separately():
    # margin 0.10 -> keep x in [10, 90], y in [20, 180] for a 100 x 200 frame
    inside, rejected = pose._reject_edge_joints(_joints(ankle=[10.0, 20.0]), _W, _H, 0.10)
    assert rejected == [] and inside["ankle"] == [10.0, 20.0]
    # (5, 100) is only 5 px from the left edge: 5% of the width -> out for margin 10%
    _, rejected = pose._reject_edge_joints(_joints(ankle=[5.0, 100.0]), _W, _H, 0.10)
    assert rejected == ["ankle"]
    # (50, 190) is 10 px from the bottom edge: only 5% of the height -> out for margin 10%
    _, rejected = pose._reject_edge_joints(_joints(ankle=[50.0, 190.0]), _W, _H, 0.10)
    assert rejected == ["ankle"]


def test_edge_helper_none_joints_stay_none_and_are_not_reported():
    joints = _joints(knee=None, knee_conf=0.0, ankle=None, ankle_conf=0.0)
    cleaned, rejected = pose._reject_edge_joints(joints, _W, _H, _MARGIN)
    assert rejected == []
    assert cleaned["knee"] is None and cleaned["knee_conf"] == 0.0
    assert cleaned["ankle"] is None and cleaned["ankle_conf"] == 0.0
    assert cleaned["hip"] == [50.0, 100.0]
    # no (0, 0) fakes are ever produced
    assert all(cleaned[name] != [0.0, 0.0] for name in ("hip", "knee", "ankle"))


@pytest.mark.parametrize("bad_size", [(None, 200), (100, None), (0, 200), (100, -1), (float("nan"), 200)])
def test_edge_helper_unknown_frame_size_does_not_reject_anything(bad_size):
    joints = _joints(ankle=[50.0, 250.0])
    cleaned, rejected = pose._reject_edge_joints(joints, bad_size[0], bad_size[1], _MARGIN)
    assert rejected == [] and cleaned == joints


@pytest.mark.parametrize("bad_margin", [-0.1, 0.5, 0.9, float("nan"), float("inf"), "abc", None])
def test_extract_rejects_invalid_edge_margin(bad_margin):
    with pytest.raises(ValueError, match="edge_margin"):
        pose.extract_video_pose("does_not_matter.mp4", edge_margin=bad_margin)


# ------------------------------------------------------------- file / video errors


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        pose.extract_video_pose(str(tmp_path / "nope.mp4"))


def test_unreadable_file_raises_value_error(tmp_path):
    bad = tmp_path / "garbage.mp4"
    bad.write_bytes(b"this is definitely not a video")
    with pytest.raises(ValueError):
        pose.extract_video_pose(str(bad))
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(ValueError):
        pose.extract_video_pose(str(empty))


class _FakeCapture:
    """Stand-in for cv2.VideoCapture: `n_frames` black frames at `fps`. The mock provides
    the DECODED frame size (width x height) that the edge rule reads from frame.shape."""

    def __init__(self, fps, n_frames, width=100, height=200):
        self._fps = fps
        self._left = n_frames
        self._width = width
        self._height = height

    def isOpened(self):
        return True

    def get(self, _prop):
        return self._fps

    def grab(self):
        if self._left <= 0:
            return False
        self._left -= 1
        return True

    def retrieve(self):
        return True, np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def release(self):
        pass


def _fake_video_file(tmp_path):
    path = tmp_path / "fake.mp4"
    path.write_bytes(b"placeholder - cv2.VideoCapture is mocked")
    return str(path)


@pytest.mark.parametrize("bad_fps", [0.0, float("nan")])
def test_zero_or_nan_fps_raises_value_error(tmp_path, monkeypatch, bad_fps):
    monkeypatch.setattr(pose.cv2, "VideoCapture", lambda _p: _FakeCapture(bad_fps, 10))
    with pytest.raises(ValueError, match="FPS"):
        pose.extract_video_pose(_fake_video_file(tmp_path))


def test_zero_frame_video_raises_value_error(tmp_path, monkeypatch):
    monkeypatch.setattr(pose.cv2, "VideoCapture", lambda _p: _FakeCapture(30.0, 0))
    with pytest.raises(ValueError):
        pose.extract_video_pose(_fake_video_file(tmp_path))


def test_extract_with_mocked_capture_and_model(tmp_path, monkeypatch):
    """Whole extract_video_pose loop with a FAKE model: no ultralytics, no inference."""
    fps = 30.0
    monkeypatch.setattr(pose.cv2, "VideoCapture", lambda _p: _FakeCapture(fps, 62))

    xy, conf = make_keypoints({14: (0.0, 0.0, 0.9)})  # knee undetected -> None

    class FakeModel:
        calls = 0

        def predict(self, frame, device=None, verbose=None):
            FakeModel.calls += 1
            assert verbose is False
            result = SimpleNamespace(
                boxes=SimpleNamespace(xyxy=np.array([[0.0, 0.0, 5.0, 5.0], [0.0, 0.0, 90.0, 190.0]])),
                keypoints=SimpleNamespace(xy=np.stack([xy * 0, xy]), conf=np.stack([conf * 0, conf])),
            )
            return [result]

    monkeypatch.setattr(pose, "_load_model", lambda _path: FakeModel())
    monkeypatch.setattr(pose, "_get_device", lambda: "cpu")

    records = pose.extract_video_pose(_fake_video_file(tmp_path), side="right", target_fps=5.0)
    # stride = 6 -> frames 0, 6, 12, ... 60 -> 11 records, time = frame_idx / fps
    assert len(records) == 11
    assert FakeModel.calls == 11
    assert [r["time_s"] for r in records] == pytest.approx([i * 6 / fps for i in range(11)])
    for record in records:
        assert set(record) == RECORD_KEYS
        assert record["knee"] is None and record["knee_conf"] == 0.0
        assert record["hip"] == [10.0 + 12, 20.0 + 24]  # person 2 (largest box) was used
        assert record["ankle"] == [10.0 + 16, 20.0 + 32]  # interior of the 100 x 200 frame
        assert record["edge_rejected"] == []  # the (0, 0) knee is "undetected", not "edge"


class _ScriptedModel:
    """Fake model that returns one prepared (xy, conf) person per call, in order."""

    def __init__(self, persons):
        self._persons = list(persons)
        self.calls = 0

    def predict(self, frame, device=None, verbose=None):
        xy, conf = self._persons[self.calls]
        self.calls += 1
        result = SimpleNamespace(
            boxes=SimpleNamespace(xyxy=np.array([[0.0, 0.0, 90.0, 190.0]])),
            keypoints=SimpleNamespace(xy=np.stack([xy]), conf=np.stack([conf])),
        )
        return [result]


def test_extract_edge_ankle_rejected_but_interior_frame_untouched(tmp_path, monkeypatch):
    """Frame 100 x 200 from the mock: a right ankle at y = 0.99*H is extrapolated by the
    model at the bottom border -> None / conf 0.0 / edge_rejected == ["ankle"]."""
    width, height = 100, 200
    monkeypatch.setattr(pose.cv2, "VideoCapture", lambda _p: _FakeCapture(5.0, 2, width, height))
    edge_xy, edge_conf = make_keypoints({16: (26.0, 0.99 * height, 0.95)})
    inside_xy, inside_conf = make_keypoints({16: (26.0, 0.50 * height, 0.95)})
    model = _ScriptedModel([(edge_xy, edge_conf), (inside_xy, inside_conf)])
    monkeypatch.setattr(pose, "_load_model", lambda _path: model)
    monkeypatch.setattr(pose, "_get_device", lambda: "cpu")

    edge_record, inside_record = pose.extract_video_pose(
        _fake_video_file(tmp_path), side="right", target_fps=5.0
    )
    assert model.calls == 2
    assert set(edge_record) == RECORD_KEYS and set(inside_record) == RECORD_KEYS

    assert edge_record["ankle"] is None
    assert edge_record["ankle_conf"] == 0.0
    assert edge_record["edge_rejected"] == ["ankle"]
    assert edge_record["hip"] == [22.0, 44.0] and edge_record["hip_conf"] == pytest.approx(0.9)
    assert edge_record["knee"] == [24.0, 48.0] and edge_record["knee_conf"] == pytest.approx(0.9)

    assert inside_record["ankle"] == [26.0, 0.50 * height]
    assert inside_record["ankle_conf"] == pytest.approx(0.95)
    assert inside_record["edge_rejected"] == []


def test_extract_edge_margin_zero_keeps_the_edge_ankle(tmp_path, monkeypatch):
    width, height = 100, 200
    monkeypatch.setattr(pose.cv2, "VideoCapture", lambda _p: _FakeCapture(5.0, 1, width, height))
    edge_xy, edge_conf = make_keypoints({16: (26.0, 0.99 * height, 0.95)})
    monkeypatch.setattr(pose, "_load_model", lambda _path: _ScriptedModel([(edge_xy, edge_conf)]))
    monkeypatch.setattr(pose, "_get_device", lambda: "cpu")

    (record,) = pose.extract_video_pose(
        _fake_video_file(tmp_path), side="right", target_fps=5.0, edge_margin=0.0
    )
    assert record["ankle"] == [26.0, 0.99 * height]
    assert record["ankle_conf"] == pytest.approx(0.95)
    assert record["edge_rejected"] == []


def test_extract_edge_rule_uses_decoded_frame_shape_for_rotated_video(tmp_path, monkeypatch):
    """The same keypoint is interior in a landscape frame (200 x 100) and at the right
    border of a portrait one (100 x 200): size comes from the decoded frame, not metadata."""
    xy, conf = make_keypoints({16: (190.0, 50.0, 0.95)})
    for width, height, expected in ((200, 100, []), (100, 200, ["ankle"])):
        monkeypatch.setattr(
            pose.cv2, "VideoCapture", lambda _p, w=width, h=height: _FakeCapture(5.0, 1, w, h)
        )
        monkeypatch.setattr(pose, "_load_model", lambda _path: _ScriptedModel([(xy, conf)]))
        monkeypatch.setattr(pose, "_get_device", lambda: "cpu")
        (record,) = pose.extract_video_pose(_fake_video_file(tmp_path), side="right", target_fps=5.0)
        assert record["edge_rejected"] == expected


# ------------------------------------------------------------------- model helper


def test_model_name_follows_env_var(monkeypatch):
    monkeypatch.setenv("REHABTRACK_MODEL", "C:/somewhere/custom-weights.pt")
    assert pose._resolve_model_path() == "C:/somewhere/custom-weights.pt"
    assert pose.get_model_name() == "custom-weights.pt"
    monkeypatch.delenv("REHABTRACK_MODEL")
    assert pose.get_model_name().endswith("yolo11n-pose.pt")


# ------------------------------- model cache, inference lock, error wrapper (all mocked)

_FAKE_WEIGHTS = "C:/fake-weights-dir/yolo11n-pose.pt"
_N_RECORDS = 4  # _FakeCapture(30.0, 20) at target_fps 5 -> frames 0, 6, 12, 18


def _use_fake_video(tmp_path, monkeypatch, fps=30.0, n_frames=20):
    """Patch cv2.VideoCapture with a fresh fake per call; return the (placeholder) path."""
    monkeypatch.setattr(pose.cv2, "VideoCapture", lambda _p: _FakeCapture(fps, n_frames))
    return _fake_video_file(tmp_path)


def _person_result():
    """One detected person, right leg interior of the 100 x 200 fake frame."""
    xy, conf = make_keypoints()
    return SimpleNamespace(
        boxes=SimpleNamespace(xyxy=np.array([[0.0, 0.0, 90.0, 190.0]])),
        keypoints=SimpleNamespace(xy=np.stack([xy]), conf=np.stack([conf])),
    )


class _OverlapProbeModel:
    """Fake model whose predict() sleeps and records how many calls ran at the same time."""

    def __init__(self, delay=0.02):
        self._delay = delay
        self._guard = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0
        self.calls = 0

    def predict(self, frame, device=None, verbose=None):
        with self._guard:
            self.in_flight += 1
            self.calls += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            time.sleep(self._delay)
        finally:
            with self._guard:
                self.in_flight -= 1
        return [_person_result()]


def _extract_in_threads(video_path, n_threads):
    """Call extract_video_pose from n_threads threads released together by a barrier."""
    barrier = threading.Barrier(n_threads)
    outputs = [None] * n_threads
    errors = []

    def worker(index):
        try:
            barrier.wait(timeout=10)
            outputs[index] = pose.extract_video_pose(video_path, side="right", target_fps=5.0)
        except BaseException as exc:  # reported to the test, never swallowed
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads), "worker thread hung"
    return outputs, errors


class _CountingYolo:
    """Stand-in for ultralytics.YOLO: counts constructions and remembers the paths."""

    instances: list = []

    def __init__(self, path):
        type(self).instances.append(path)
        self.path = path

    def predict(self, frame, device=None, verbose=None):
        return [_person_result()]


@pytest.fixture()
def fake_ultralytics(monkeypatch):
    """A fake `ultralytics` module so the REAL _load_model / lru_cache runs, no inference."""
    _CountingYolo.instances = []
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=_CountingYolo))
    monkeypatch.setattr(pose, "_get_device", lambda: "cpu")
    monkeypatch.setenv("REHABTRACK_MODEL", _FAKE_WEIGHTS)
    return _CountingYolo


def test_import_pose_does_not_import_ultralytics():
    assert not hasattr(pose, "YOLO")  # imported lazily inside _load_model only


def test_loader_is_asked_once_per_extract_call_not_once_per_frame(tmp_path, monkeypatch):
    video = _use_fake_video(tmp_path, monkeypatch)
    monkeypatch.setattr(pose, "_get_device", lambda: "cpu")
    monkeypatch.setenv("REHABTRACK_MODEL", _FAKE_WEIGHTS)
    asked = []
    model = _OverlapProbeModel(delay=0.0)

    def mocked_loader(path):
        asked.append(path)
        return model

    monkeypatch.setattr(pose, "_load_model", mocked_loader)
    pose.extract_video_pose(video, side="right", target_fps=5.0)
    assert model.calls == _N_RECORDS and asked == [_FAKE_WEIGHTS]
    pose.extract_video_pose(video, side="right", target_fps=5.0)
    assert asked == [_FAKE_WEIGHTS, _FAKE_WEIGHTS]  # reuse across calls is the lru_cache's job


def test_real_cache_loads_yolo_once_per_weights_path(tmp_path, monkeypatch, fake_ultralytics):
    video = _use_fake_video(tmp_path, monkeypatch)
    for _ in range(3):
        pose.extract_video_pose(video, side="right", target_fps=5.0)
    assert fake_ultralytics.instances == [_FAKE_WEIGHTS]  # cache hit on calls 2 and 3
    assert _REAL_LOAD_MODEL.cache_info().hits >= 2

    monkeypatch.setenv("REHABTRACK_MODEL", "C:/fake-weights-dir/other.pt")  # new key
    pose.extract_video_pose(video, side="right", target_fps=5.0)
    assert fake_ultralytics.instances == [_FAKE_WEIGHTS, "C:/fake-weights-dir/other.pt"]

    _REAL_LOAD_MODEL.cache_clear()  # the reset hook for tests
    monkeypatch.setenv("REHABTRACK_MODEL", _FAKE_WEIGHTS)
    pose.extract_video_pose(video, side="right", target_fps=5.0)
    assert fake_ultralytics.instances[-1] == _FAKE_WEIGHTS and len(fake_ultralytics.instances) == 3


def test_failed_load_is_not_cached_and_next_call_retries(tmp_path, monkeypatch, fake_ultralytics):
    video = _use_fake_video(tmp_path, monkeypatch)
    state = {"fail": True}
    original_init = _CountingYolo.__init__

    def flaky_init(self, path):
        if state["fail"]:
            raise OSError("weights temporarily unreadable")
        original_init(self, path)

    monkeypatch.setattr(_CountingYolo, "__init__", flaky_init)
    with pytest.raises(RuntimeError):
        pose.extract_video_pose(video, side="right", target_fps=5.0)
    state["fail"] = False
    records = pose.extract_video_pose(video, side="right", target_fps=5.0)
    assert len(records) == _N_RECORDS


def test_concurrent_threads_never_run_predict_at_the_same_time(tmp_path, monkeypatch):
    video = _use_fake_video(tmp_path, monkeypatch)
    monkeypatch.setattr(pose, "_get_device", lambda: "cpu")
    model = _OverlapProbeModel()
    monkeypatch.setattr(pose, "_load_model", lambda _path: model)

    reference = pose.extract_video_pose(video, side="right", target_fps=5.0)
    model.calls = 0

    outputs, errors = _extract_in_threads(video, n_threads=3)
    assert errors == []
    assert model.calls == 3 * _N_RECORDS
    assert model.max_in_flight == 1  # the lock serialised every predict
    for records in outputs:
        assert len(records) == _N_RECORDS
        assert records == reference  # every thread got the correct, identical records


def test_control_without_the_lock_the_probe_does_see_overlap(tmp_path, monkeypatch):
    """Proves the probe in the test above can fail: with a no-op lock, calls overlap."""
    video = _use_fake_video(tmp_path, monkeypatch)
    monkeypatch.setattr(pose, "_get_device", lambda: "cpu")
    model = _OverlapProbeModel()
    monkeypatch.setattr(pose, "_load_model", lambda _path: model)
    monkeypatch.setattr(pose, "_INFERENCE_LOCK", contextlib.nullcontext())

    outputs, errors = _extract_in_threads(video, n_threads=3)
    assert errors == []
    assert model.max_in_flight >= 2


def test_concurrent_cold_start_loads_the_weights_only_once(tmp_path, monkeypatch, fake_ultralytics):
    video = _use_fake_video(tmp_path, monkeypatch)
    slow_init = fake_ultralytics.__init__

    def slow_load(self, path):
        time.sleep(0.05)  # widen the window in which a second thread could also miss the cache
        slow_init(self, path)

    monkeypatch.setattr(_CountingYolo, "__init__", slow_load)
    outputs, errors = _extract_in_threads(video, n_threads=3)
    assert errors == []
    assert fake_ultralytics.instances == [_FAKE_WEIGHTS]
    assert all(len(records) == _N_RECORDS for records in outputs)


def test_predict_failure_becomes_runtime_error_with_cause_and_path(tmp_path, monkeypatch):
    video = _use_fake_video(tmp_path, monkeypatch)
    monkeypatch.setattr(pose, "_get_device", lambda: "cpu")
    monkeypatch.setenv("REHABTRACK_MODEL", _FAKE_WEIGHTS)
    boom = ValueError("CUDA out of memory (simulated)")

    class ExplodingModel:
        def predict(self, frame, device=None, verbose=None):
            raise boom

    monkeypatch.setattr(pose, "_load_model", lambda _path: ExplodingModel())
    with pytest.raises(RuntimeError) as caught:
        pose.extract_video_pose(video, side="right", target_fps=5.0)
    assert caught.value.__cause__ is boom
    message = str(caught.value)
    assert message.startswith("YOLO pose inference failed.")
    assert "ultralytics/torch installation" in message
    assert _FAKE_WEIGHTS in message
    assert pose._INFERENCE_LOCK.acquire(blocking=False)  # lock released after the failure
    pose._INFERENCE_LOCK.release()


@pytest.mark.parametrize(
    "load_error",
    [ImportError("No module named 'ultralytics'"), FileNotFoundError("weights file missing")],
)
def test_loader_failure_becomes_runtime_error_with_cause(tmp_path, monkeypatch, load_error):
    video = _use_fake_video(tmp_path, monkeypatch)
    monkeypatch.setattr(pose, "_get_device", lambda: "cpu")
    monkeypatch.setenv("REHABTRACK_MODEL", _FAKE_WEIGHTS)

    def failing_loader(_path):
        raise load_error

    monkeypatch.setattr(pose, "_load_model", failing_loader)
    with pytest.raises(RuntimeError) as caught:
        pose.extract_video_pose(video, side="right", target_fps=5.0)
    assert caught.value.__cause__ is load_error
    assert _FAKE_WEIGHTS in str(caught.value)


def test_missing_ultralytics_package_becomes_runtime_error(tmp_path, monkeypatch):
    """The REAL lazy import: ultralytics absent (None in sys.modules) -> ImportError cause."""
    video = _use_fake_video(tmp_path, monkeypatch)
    monkeypatch.setattr(pose, "_get_device", lambda: "cpu")
    monkeypatch.setenv("REHABTRACK_MODEL", _FAKE_WEIGHTS)
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    with pytest.raises(RuntimeError) as caught:
        pose.extract_video_pose(video, side="right", target_fps=5.0)
    assert isinstance(caught.value.__cause__, ImportError)


def test_keyboard_interrupt_is_not_wrapped(tmp_path, monkeypatch):
    video = _use_fake_video(tmp_path, monkeypatch)
    monkeypatch.setattr(pose, "_get_device", lambda: "cpu")

    class InterruptedModel:
        def predict(self, frame, device=None, verbose=None):
            raise KeyboardInterrupt

    monkeypatch.setattr(pose, "_load_model", lambda _path: InterruptedModel())
    with pytest.raises(KeyboardInterrupt):
        pose.extract_video_pose(video, side="right", target_fps=5.0)
    assert pose._INFERENCE_LOCK.acquire(blocking=False)
    pose._INFERENCE_LOCK.release()


def test_input_errors_stay_their_own_type_and_never_load_the_model(tmp_path, monkeypatch):
    """Bad side / edge_margin / target_fps / fps, missing file, unreadable or empty video:
    the original exception type is kept (NOT RuntimeError) and no model load is attempted."""

    def must_not_load(_path):
        raise AssertionError("model must not be loaded for an input error")

    monkeypatch.setattr(pose, "_load_model", must_not_load)
    monkeypatch.setattr(pose, "_get_device", lambda: "cpu")
    video = _use_fake_video(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="side"):
        pose.extract_video_pose(video, side="front")
    with pytest.raises(ValueError, match="edge_margin"):
        pose.extract_video_pose(video, edge_margin=0.7)
    with pytest.raises(ValueError, match="target_fps"):
        pose.extract_video_pose(video, target_fps=0)
    with pytest.raises(FileNotFoundError):
        pose.extract_video_pose(str(tmp_path / "nope.mp4"))

    monkeypatch.setattr(pose.cv2, "VideoCapture", lambda _p: _FakeCapture(0.0, 10))
    with pytest.raises(ValueError, match="FPS"):
        pose.extract_video_pose(video)
    monkeypatch.setattr(pose.cv2, "VideoCapture", lambda _p: _FakeCapture(30.0, 0))
    with pytest.raises(ValueError):
        pose.extract_video_pose(video)


def test_unreadable_real_file_is_value_error_not_runtime_error(tmp_path, monkeypatch):
    monkeypatch.setattr(pose, "_load_model", lambda _path: pytest.fail("no model load expected"))
    bad = tmp_path / "garbage.mp4"
    bad.write_bytes(b"this is definitely not a video")
    with pytest.raises(ValueError) as caught:
        pose.extract_video_pose(str(bad))
    assert not isinstance(caught.value, RuntimeError)


# ---------------------------------------------------- integration (needs weights)


def test_integration_blank_synthetic_clip_has_no_person(tmp_path):
    """SYNTHETIC clip (blank black frames, no person) - not real footage. Real YOLO
    inference: skipped unless ultralytics and the weights file are available."""
    pytest.importorskip("ultralytics")
    weights = pose._resolve_model_path()
    if not os.path.exists(weights):
        pytest.skip(f"weights not available at {weights}")

    import cv2

    video_path = str(tmp_path / "blank_synthetic.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (320, 240))
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write mp4v on this machine")
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(45):  # 3 s at 15 fps
        writer.write(blank)
    writer.release()

    records = pose.extract_video_pose(video_path, side="right", target_fps=5.0)
    assert len(records) >= 10
    times = [r["time_s"] for r in records]
    assert all(b > a for a, b in zip(times, times[1:]))
    assert all(math.isclose(b - a, 0.2, abs_tol=0.03) for a, b in zip(times, times[1:]))
    for record in records:
        assert set(record) == RECORD_KEYS
        assert record["hip"] is None and record["knee"] is None and record["ankle"] is None
        assert record["hip_conf"] == record["knee_conf"] == record["ankle_conf"] == 0.0
