"""Tests for metrics.py. Pure math on SYNTHETIC keypoints - not model output."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics import (  # noqa: E402
    MIN_VALID_FRAMES,
    MIN_VALID_RATIO,
    analyze_session,
    compare_sessions,
    flexion_deg_from_points,
    interior_angle_deg,
)

ANALYZE_KEYS = {
    "status",
    "time_s",
    "flexion_deg",
    "best_extension_flexion_deg",
    "observed_rom_deg",
    "valid_ratio",
    "valid_frame_count",
    "total_frame_count",
    "warnings",
    "valid_frames",
    "total_frames",
    "flexion_raw_deg",
}
COMPARE_KEYS = {"status", "delta_extension_deg", "interpretation", "warnings"}
FORBIDDEN_WORDS = ("recovery", "recovered", "improvement", "outcome", "%", "hồi phục", "cải thiện")


def make_frame(time_s, flexion_deg, conf=0.9):
    """Synthetic frame: knee at origin, hip straight up, ankle placed so that the
    flexion at the knee equals `flexion_deg` (interior = 180 - flexion)."""
    interior = math.radians(180.0 - flexion_deg)
    return {
        "time_s": time_s,
        "hip": [0.0, 1.0],
        "knee": [0.0, 0.0],
        "ankle": [math.sin(interior), math.cos(interior)],
        "hip_conf": conf,
        "knee_conf": conf,
        "ankle_conf": conf,
    }


def make_sequence(flexions, conf=0.9):
    return [make_frame(i * 0.2, f, conf) for i, f in enumerate(flexions)]


def make_session(best_extension, status="ok"):
    return {"status": status, "best_extension_flexion_deg": best_extension}


# ------------------------------------------------------------------ geometry


def test_straight_leg_is_180_interior_0_flexion():
    assert interior_angle_deg((0, 1), (0, 0), (0, -1)) == pytest.approx(180.0)
    assert flexion_deg_from_points((0, 1), (0, 0), (0, -1)) == pytest.approx(0.0, abs=1e-6)


def test_right_angle_is_90_interior_90_flexion():
    assert interior_angle_deg((0, 1), (0, 0), (1, 0)) == pytest.approx(90.0)
    assert flexion_deg_from_points((0, 1), (0, 0), (1, 0)) == pytest.approx(90.0)


def test_flexion_always_within_0_180():
    for ankle in [(0, 1), (0, -1), (1, 0), (-1, 0), (3, -0.2), (0.001, 0.999)]:
        value = flexion_deg_from_points((0, 1), (0, 0), ankle)
        assert 0.0 <= value <= 180.0


def test_missing_point_or_degenerate_vector_returns_none_not_zero():
    assert flexion_deg_from_points((0, 1), None, (0, -1)) is None
    assert flexion_deg_from_points(None, (0, 0), (0, -1)) is None
    assert flexion_deg_from_points((0, 1), (0, 0), None) is None
    assert flexion_deg_from_points((0, 1), (0, 0), (0, 0)) is None  # zero-length vector
    assert flexion_deg_from_points((float("nan"), 1), (0, 0), (0, -1)) is None
    assert flexion_deg_from_points("abc", (0, 0), (0, -1)) is None
    assert interior_angle_deg((0, 1), (0, 0), (0, 0)) is None


# ----------------------------------------------------------------- analyze_session


def test_gate_constants_are_the_documented_mvp_values():
    assert MIN_VALID_FRAMES == 10  # 2 s of usable data at 5 fps
    assert MIN_VALID_RATIO == pytest.approx(0.60)


def test_all_required_keys_present_and_lists_aligned():
    result = analyze_session(make_sequence([70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 22, 20]))
    assert ANALYZE_KEYS <= set(result)
    assert result["status"] == "ok"
    assert len(result["time_s"]) == len(result["flexion_deg"]) == len(result["flexion_raw_deg"]) == 12
    assert result["valid_frames"] == result["valid_frame_count"] == 12
    assert result["total_frames"] == result["total_frame_count"] == 12
    assert result["valid_ratio"] == pytest.approx(1.0)
    assert result["best_extension_flexion_deg"] is not None
    assert result["observed_rom_deg"] is not None
    assert result["observed_rom_deg"] >= 0.0
    assert any("sagittal" in w for w in result["warnings"])  # 2D projection caveat


def test_none_keypoint_is_excluded_not_zero():
    frames = make_sequence([60, 55, 50, 45, 40, 35, 30, 25])
    frames[3]["knee"] = None
    frames[3]["knee_conf"] = 0.0
    result = analyze_session(frames)
    assert result["flexion_raw_deg"][3] is None
    assert result["flexion_deg"][3] is None
    assert result["valid_frame_count"] == 7
    assert sum(v is None for v in result["flexion_deg"]) == 1  # a gap, not a 0 deg value


def test_low_confidence_frame_is_excluded():
    frames = make_sequence([60, 55, 50, 45, 40, 35, 30, 25])
    frames[2]["ankle_conf"] = 0.1
    result = analyze_session(frames, min_conf=0.40)
    assert result["flexion_raw_deg"][2] is None
    assert result["valid_frame_count"] == 7


def _sequence_with_valid_count(total, valid):
    """`total` frames of which the first `valid` are usable (knee conf 0 on the rest)."""
    frames = make_sequence([50.0 + (i % 5) for i in range(total)])
    for index in range(valid, total):
        frames[index]["knee_conf"] = 0.0
    return frames


def test_too_few_valid_frames_gives_insufficient_data_without_fabricated_rom():
    frames = make_sequence([60, 55, 50, 45, 40, 35, 30, 25])
    for index in range(4, 8):  # only 4 valid frames left (< 10)
        frames[index]["hip"] = None
        frames[index]["hip_conf"] = 0.0
    result = analyze_session(frames)
    assert result["status"] == "insufficient_data"
    assert result["best_extension_flexion_deg"] is None
    assert result["observed_rom_deg"] is None
    assert result["valid_frame_count"] == 4
    assert any("Thiếu dữ liệu" in w for w in result["warnings"])


def test_insufficient_warning_quotes_the_actual_constants():
    result = analyze_session(_sequence_with_valid_count(20, 4))
    text = next(w for w in result["warnings"] if "Thiếu dữ liệu" in w)
    assert f"cần ít nhất {MIN_VALID_FRAMES} frame" in text
    assert f"{MIN_VALID_RATIO:.0%} tổng số frame" in text
    assert "4/20" in text
    assert "ngưỡng quy ước MVP" in text and "không phải ngưỡng y khoa" in text


def test_low_valid_ratio_gives_insufficient_data():
    frames = _sequence_with_valid_count(20, 11)  # 0.55 < 0.60 although count >= 10
    result = analyze_session(frames)
    assert result["valid_frame_count"] == 11
    assert result["valid_ratio"] == pytest.approx(0.55)
    assert result["status"] == "insufficient_data"
    assert result["best_extension_flexion_deg"] is None
    assert result["observed_rom_deg"] is None


def test_nine_valid_frames_is_insufficient_even_at_full_ratio():
    result = analyze_session(_sequence_with_valid_count(9, 9))  # ratio 1.0, count 9 < 10
    assert result["valid_ratio"] == pytest.approx(1.0)
    assert result["valid_frame_count"] == 9
    assert result["status"] == "insufficient_data"
    assert result["best_extension_flexion_deg"] is None


def test_exactly_meeting_both_gates_is_ok():
    # exactly MIN_VALID_FRAMES valid frames (ratio 10/16 = 0.625 >= 0.60)
    by_count = analyze_session(_sequence_with_valid_count(16, 10))
    assert by_count["valid_frame_count"] == MIN_VALID_FRAMES
    assert by_count["status"] == "ok"
    assert by_count["best_extension_flexion_deg"] is not None
    # exactly MIN_VALID_RATIO (12/20 = 0.60) with 12 valid frames (>= 10)
    by_ratio = analyze_session(_sequence_with_valid_count(20, 12))
    assert by_ratio["valid_ratio"] == pytest.approx(MIN_VALID_RATIO)
    assert by_ratio["status"] == "ok"
    assert by_ratio["best_extension_flexion_deg"] is not None
    assert by_ratio["observed_rom_deg"] is not None
    # one frame fewer on either gate flips it to insufficient_data
    assert analyze_session(_sequence_with_valid_count(16, 9))["status"] == "insufficient_data"
    assert analyze_session(_sequence_with_valid_count(20, 11))["status"] == "insufficient_data"


def test_valid_ratio_below_070_adds_warning_but_stays_ok():
    frames = _sequence_with_valid_count(20, 13)  # 13 of 20 = 0.65: >= 0.60 gate, < 0.70 warn
    result = analyze_session(frames)
    assert result["status"] == "ok"
    assert result["valid_ratio"] == pytest.approx(0.65)
    assert any("dưới 70%" in w for w in result["warnings"])


# ------------------------------------------------------ edge-of-frame exclusions


def _with_edge_rejection(frames, indices, joint="ankle"):
    """Mimic pose.py: those frames lost `joint` to the frame-border rule."""
    for index in indices:
        frames[index][joint] = None
        frames[index][f"{joint}_conf"] = 0.0
        frames[index]["edge_rejected"] = [joint]
    return frames


def test_edge_rejected_frames_add_a_warning_naming_the_joint_and_the_framing_advice():
    frames = _with_edge_rejection(make_sequence([50.0 + (i % 5) for i in range(20)]), range(3))
    result = analyze_session(frames)
    assert result["status"] == "ok"  # 17/20 valid
    assert result["valid_frame_count"] == 17
    edge_warnings = [w for w in result["warnings"] if "mép khung hình" in w]
    assert len(edge_warnings) == 1
    text = edge_warnings[0]
    assert "3/20" in text  # how many frames
    assert "ankle" in text  # which joint
    assert "hip" in text and "knee" in text  # framing advice names the joints to keep in frame
    assert "trọn trong khung hình" in text
    # status ok: the cause clause belongs to the insufficient message only
    assert not any("Nguyên nhân" in w for w in result["warnings"])


def test_edge_rejected_joints_are_listed_in_hip_knee_ankle_order():
    frames = make_sequence([50.0 + (i % 5) for i in range(20)])
    _with_edge_rejection(frames, [0], joint="ankle")
    _with_edge_rejection(frames, [1], joint="hip")
    text = next(w for w in analyze_session(frames)["warnings"] if "mép khung hình" in w)
    assert "2/20" in text
    assert text.index("hip") < text.index("ankle")
    assert "knee (đầu gối)" not in text.split("Hãy đặt")[0]  # knee was not rejected


def test_insufficient_with_edge_rejected_frames_names_edge_as_cause():
    frames = _with_edge_rejection(make_sequence([50.0 + (i % 5) for i in range(20)]), range(12))
    result = analyze_session(frames)  # 8/20 valid
    assert result["status"] == "insufficient_data"
    assert result["best_extension_flexion_deg"] is None
    insufficient = next(w for w in result["warnings"] if "Thiếu dữ liệu" in w)
    assert "Nguyên nhân" in insufficient
    assert "12 frame" in insufficient and "ankle" in insufficient
    assert "mép khung hình" in insufficient
    assert any("trọn trong khung hình" in w for w in result["warnings"])


def test_insufficient_without_edge_rejection_has_no_edge_cause():
    result = analyze_session(_sequence_with_valid_count(20, 4))
    assert result["status"] == "insufficient_data"
    assert not any("mép khung hình" in w for w in result["warnings"])
    assert not any("Nguyên nhân" in w for w in result["warnings"])


def test_empty_or_malformed_edge_rejected_values_are_ignored():
    frames = make_sequence([50.0 + (i % 5) for i in range(12)])
    frames[0]["edge_rejected"] = []
    frames[1]["edge_rejected"] = None
    frames[2]["edge_rejected"] = "ankle"  # not a list
    frames[3]["edge_rejected"] = ["not_a_joint", 5]
    result = analyze_session(frames)
    assert result["status"] == "ok"
    assert not any("mép khung hình" in w for w in result["warnings"])


def test_old_style_frames_without_edge_rejected_still_analyse():
    frames = make_sequence([70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 22, 20])
    assert all("edge_rejected" not in frame for frame in frames)
    result = analyze_session(frames)
    assert result["status"] == "ok"
    assert result["valid_frame_count"] == 12
    assert result["best_extension_flexion_deg"] is not None
    assert not any("mép khung hình" in w for w in result["warnings"])


def test_p10_robust_to_single_outlier_frame():
    clean_flexions = [60, 58, 55, 50, 45, 40, 35, 30, 28, 27, 28, 30]
    clean = analyze_session(make_sequence(clean_flexions))
    outlier_flexions = list(clean_flexions)
    outlier_flexions[6] = 0.0  # one wildly wrong frame (impossible straight-leg spike)
    noisy = analyze_session(make_sequence(outlier_flexions))
    assert noisy["status"] == "ok"
    # a single mid-sequence outlier must not drag the P10 anywhere near 0
    assert noisy["best_extension_flexion_deg"] == pytest.approx(
        clean["best_extension_flexion_deg"], abs=3.0
    )
    assert noisy["best_extension_flexion_deg"] > 20.0
    assert noisy["flexion_raw_deg"][6] == pytest.approx(0.0, abs=1e-6)  # raw is untouched


def test_smoothing_never_crosses_none_gaps():
    frames = make_sequence([10, 90, 10, 50, 50, 50, 50])
    frames[3]["knee"] = None
    frames[3]["knee_conf"] = 0.0
    result = analyze_session(frames)
    assert result["flexion_deg"][3] is None  # None stays None, no interpolation
    # run 0..2: middle frame (90) is replaced by the median of its run neighbours (10)
    assert result["flexion_deg"][1] == pytest.approx(10.0, abs=1e-6)
    # frame 4 is the start of a new run: it must NOT be smoothed with frame 2's values
    assert result["flexion_deg"][4] == pytest.approx(50.0, abs=1e-6)


def test_empty_frames_list_does_not_crash():
    result = analyze_session([])
    assert ANALYZE_KEYS <= set(result)
    assert result["status"] == "insufficient_data"
    assert result["valid_ratio"] == 0.0
    assert result["total_frame_count"] == 0
    assert result["time_s"] == [] and result["flexion_deg"] == []
    assert result["best_extension_flexion_deg"] is None


def test_malformed_frames_do_not_crash():
    frames = [
        {},  # missing keys
        None,  # not a dict
        {"time_s": 0.2, "hip": "abc", "knee": [0, 0], "ankle": [0, -1],
         "hip_conf": 0.9, "knee_conf": 0.9, "ankle_conf": 0.9},  # non-numeric point
        {"time_s": 0.4, "hip": [0, 1], "knee": [float("nan"), 0], "ankle": [0, -1],
         "hip_conf": 0.9, "knee_conf": 0.9, "ankle_conf": 0.9},  # NaN
        {"time_s": 0.6, "hip": [0, 1], "knee": [0, 0], "ankle": [0, -1],
         "hip_conf": "high", "knee_conf": float("nan"), "ankle_conf": None},  # bad conf
        {"time_s": "x", "hip": [0, 1]},
    ]
    result = analyze_session(frames)
    assert result["status"] == "insufficient_data"
    assert result["valid_frame_count"] == 0
    assert result["total_frame_count"] == len(frames)
    assert result["flexion_deg"] == [None] * len(frames)
    assert result["time_s"][0] is None  # missing time is None, never invented
    assert analyze_session(None)["status"] == "insufficient_data"


# ---------------------------------------------------------------- compare_sessions


def test_compare_sign_from_hand_made_sessions():
    result = compare_sessions(make_session(40.0), make_session(15.0))
    assert COMPARE_KEYS <= set(result)
    assert result["status"] == "ok"
    assert result["delta_extension_deg"] == pytest.approx(25.0)
    assert "giảm khoảng 25.0°" in result["interpretation"]

    reverse = compare_sessions(make_session(15.0), make_session(40.0))
    assert reverse["delta_extension_deg"] == pytest.approx(-25.0)
    assert "tăng khoảng 25.0°" in reverse["interpretation"]

    same = compare_sessions(make_session(30.0), make_session(30.0))
    assert same["delta_extension_deg"] == pytest.approx(0.0)
    assert "không đổi" in same["interpretation"]


def test_compare_sign_through_analyze_session():
    # A: best extension around 40 deg flexion. B: around 15 deg.
    session_a = analyze_session(make_sequence([90, 80, 70, 60, 50, 45, 42, 40, 40, 40, 41, 45]))
    session_b = analyze_session(make_sequence([90, 70, 50, 35, 25, 18, 15, 15, 15, 15, 16, 20]))
    assert session_a["status"] == session_b["status"] == "ok"
    result = compare_sessions(session_a, session_b)
    expected = session_a["best_extension_flexion_deg"] - session_b["best_extension_flexion_deg"]
    assert result["status"] == "ok"
    assert result["delta_extension_deg"] == pytest.approx(expected)
    assert result["delta_extension_deg"] > 0  # straighter knee in B -> positive


def test_compare_sign_a_40_b_15_is_plus_25_under_the_stricter_gate():
    # constant sequences of 12 frames: P10 equals the flexion exactly
    session_a = analyze_session(make_sequence([40.0] * 12))
    session_b = analyze_session(make_sequence([15.0] * 12))
    assert session_a["status"] == session_b["status"] == "ok"
    result = compare_sessions(session_a, session_b)
    assert result["status"] == "ok"
    assert result["delta_extension_deg"] == pytest.approx(25.0, abs=1e-6)
    assert "giảm khoảng 25.0°" in result["interpretation"]


def test_compare_ignores_edge_warning_but_refuses_an_edge_failed_session():
    good = analyze_session(make_sequence([15.0] * 12))
    cut_off = analyze_session(
        _with_edge_rejection(make_sequence([40.0] * 20), range(12))  # 8/20 valid
    )
    assert cut_off["status"] == "insufficient_data"
    result = compare_sessions(cut_off, good)
    assert result["status"] == "insufficient_data"
    assert result["delta_extension_deg"] is None


def test_session_warnings_have_no_recovery_wording():
    forbidden = [w for w in FORBIDDEN_WORDS if w != "%"]  # "%" is used for the ratio gates
    frames = _with_edge_rejection(make_sequence([50.0 + (i % 5) for i in range(20)]), range(12))
    for session in (analyze_session(frames), analyze_session(make_sequence([50.0] * 12))):
        for message in session["warnings"]:
            for word in forbidden:
                assert word not in message.lower(), f"{word!r} found in {message!r}"


def test_compare_with_insufficient_input_gives_insufficient_and_no_delta():
    good = analyze_session(make_sequence([70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 22, 20]))
    assert good["status"] == "ok"
    bad = analyze_session(make_sequence([60, 55]))
    for pre, post in [(good, bad), (bad, good), (bad, bad)]:
        result = compare_sessions(pre, post)
        assert result["status"] == "insufficient_data"
        assert result["delta_extension_deg"] is None
        assert "Chưa đủ dữ liệu" in result["interpretation"]
    # status ok but P10 missing must also not produce a delta
    assert compare_sessions(make_session(None), make_session(10.0))["delta_extension_deg"] is None
    assert compare_sessions({}, {})["status"] == "insufficient_data"


def test_compare_wording_has_no_forbidden_words_and_adds_required_warnings():
    cases = [
        compare_sessions(make_session(40.0), make_session(15.0)),
        compare_sessions(make_session(15.0), make_session(40.0)),
        compare_sessions(make_session(30.0), make_session(30.0)),
        compare_sessions(make_session(None), make_session(30.0)),
    ]
    for result in cases:
        assert set(result) >= COMPARE_KEYS
        text = result["interpretation"].lower()
        for word in FORBIDDEN_WORDS:
            assert word not in text, f"{word!r} found in {result['interpretation']!r}"
        joined = " ".join(result["warnings"])
        assert "chưa được định lượng" in joined  # error unquantified, no goniometer
        assert "cùng một người" in joined and "cùng vị trí đặt camera" in joined
