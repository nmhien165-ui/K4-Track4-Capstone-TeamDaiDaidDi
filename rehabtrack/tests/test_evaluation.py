"""Tests for evaluate.py (data / QA worker).

Evaluator logic is tested with injected fake pose/analyze/compare functions, so
it needs neither YOLO nor metrics.py. Tests that use the real metrics.py are
skipped with a clear reason when it is not on disk. Nothing here runs YOLO and
nothing here is a real-video result; tiny empty placeholder files stand in for
clips because pose_fn is always injected.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import evaluate  # noqa: E402

HEADER = ",".join(evaluate.MANIFEST_COLUMNS)


def manifest_line(clip_id, path, label, group="pair_01", side="right", view="side",
                  source="staged_real", consent="yes", participant="participant_01",
                  exercise="seated_knee_extension"):
    return ",".join([clip_id, path, participant, label, exercise, side, view, source, consent, group, "test"])


def make_project(tmp_path, lines, files=("a.mp4", "b.mp4", "f.mp4"), header=HEADER):
    (tmp_path / "data").mkdir(exist_ok=True)
    for name in files:
        (tmp_path / "data" / name).write_bytes(b"")
    manifest = tmp_path / "data" / "manifest.csv"
    manifest.write_text("\n".join([header] + list(lines)) + "\n", encoding="utf-8")
    return manifest


def good_lines():
    return [manifest_line("clip_a", "data/a.mp4", "A"),
            manifest_line("clip_b", "data/b.mp4", "B"),
            manifest_line("clip_f", "data/f.mp4", "failure_bad_angle", group="", view="oblique")]


# Fake pipeline: the "frames" carry the best-extension value picked per file name.
BEST_BY_FILE = {"a.mp4": 40.0, "b.mp4": 15.0, "f.mp4": 33.0}


def fake_pose(path, side="right"):
    return [{"_best": BEST_BY_FILE[Path(path).name], "_side": side}] * 6


def fake_analyze(frames):
    best = frames[0]["_best"]
    return {"status": "ok", "time_s": [0.0], "flexion_deg": [best] * len(frames),
            "best_extension_flexion_deg": best, "observed_rom_deg": 10.0, "valid_ratio": 1.0,
            "valid_frame_count": len(frames), "total_frame_count": len(frames), "warnings": []}


def fake_compare(pre, post):
    return {"status": "ok", "delta_extension_deg": pre["best_extension_flexion_deg"] - post["best_extension_flexion_deg"],
            "warnings": []}


def run(tmp_path, lines, **kwargs):
    manifest = make_project(tmp_path, lines, header=kwargs.pop("header", HEADER))
    kwargs.setdefault("pose_fn", fake_pose)
    kwargs.setdefault("analyze_fn", fake_analyze)
    kwargs.setdefault("compare_fn", fake_compare)
    return evaluate.run_evaluation(manifest, tmp_path / "out", **kwargs)


def error_text(report):
    return " ".join(report.get("errors", []))


# ---- manifest validation -------------------------------------------------

def test_missing_columns(tmp_path):
    report, code = run(tmp_path, ["x,y"], header="clip_id,relative_path")
    assert code == 2 and "missing required column" in error_text(report)


def test_missing_file_is_readable_error(tmp_path):
    manifest = make_project(tmp_path, good_lines(), files=("b.mp4",))
    report, code = evaluate.run_evaluation(manifest, tmp_path / "out", pose_fn=fake_pose,
                                           analyze_fn=fake_analyze, compare_fn=fake_compare)
    assert code == 2
    assert "MISSING FILE" in error_text(report) and "a.mp4" in error_text(report)
    assert not (tmp_path / "out" / "report.json").exists()


def test_bad_label(tmp_path):
    report, code = run(tmp_path, [manifest_line("clip_a", "data/a.mp4", "C")])
    assert code == 2 and "invalid session_label" in error_text(report)


def test_bad_enum_values(tmp_path):
    line = manifest_line("clip_a", "data/a.mp4", "A", view="top", source="web", consent="maybe",
                         exercise="squat", side="middle")
    report, code = run(tmp_path, [line, manifest_line("clip_b", "data/b.mp4", "B")])
    text = error_text(report)
    assert code == 2
    for fragment in ("invalid camera_view", "invalid source_type", "invalid consent_confirmed",
                     "invalid exercise", "invalid side"):
        assert fragment in text


def test_duplicate_a(tmp_path):
    lines = good_lines() + [manifest_line("clip_a2", "data/a.mp4", "A")]
    report, code = run(tmp_path, lines)
    assert code == 2 and "duplicate session A" in error_text(report)


def test_missing_b_and_failure_clip_is_not_b(tmp_path):
    lines = [manifest_line("clip_a", "data/a.mp4", "A"),
             manifest_line("clip_f", "data/f.mp4", "failure_occluded", group="pair_01")]
    report, code = run(tmp_path, lines)
    assert code == 2 and "no session B" in error_text(report)


def test_side_mismatch_between_pair(tmp_path):
    lines = [manifest_line("clip_a", "data/a.mp4", "A", side="left"),
             manifest_line("clip_b", "data/b.mp4", "B", side="right")]
    report, code = run(tmp_path, lines)
    assert code == 2 and "side mismatch" in error_text(report)


def test_consent_not_confirmed_for_staged_real(tmp_path):
    lines = [manifest_line("clip_a", "data/a.mp4", "A", consent="no"), manifest_line("clip_b", "data/b.mp4", "B")]
    report, code = run(tmp_path, lines)
    assert code == 2 and "consent not confirmed" in error_text(report)


def test_oblique_or_unknown_view_is_warning_not_error(tmp_path):
    lines = [manifest_line("clip_a", "data/a.mp4", "A", view="oblique"),
             manifest_line("clip_b", "data/b.mp4", "B", view="unknown")]
    report, code = run(tmp_path, lines)
    assert code == 0
    assert any("camera_view" in w for w in report["validation_warnings"])


# ---- evaluation logic with injected functions ------------------------------

def test_delta_is_plus_25_for_40_and_15_and_failure_never_paired(tmp_path):
    report, code = run(tmp_path, good_lines())
    assert code == 0
    assert report["evaluation_type"] == "synthetic_only" and report["real_videos_processed"] == 0
    assert len(report["pairs"]) == 1
    pair = report["pairs"][0]
    assert pair["b_clip_id"] == "clip_b" and pair["a_clip_id"] == "clip_a"
    assert pair["delta_extension_deg"] == pytest.approx(25.0)
    ids = [c["clip_id"] for c in report["clips"]]
    assert "clip_f" in ids and "clip_f" not in (pair["a_clip_id"], pair["b_clip_id"])
    failure = next(c for c in report["clips"] if c["clip_id"] == "clip_f")
    assert "valid_ratio does not prove" in failure["note"]


def test_per_clip_exception_isolation(tmp_path):
    def flaky_pose(path, side="right"):
        if Path(path).name == "a.mp4":
            raise RuntimeError("unreadable video")
        return fake_pose(path, side)

    report, code = run(tmp_path, good_lines(), pose_fn=flaky_pose)
    assert code == 0  # B and the failure clip still evaluated
    clip_a = next(c for c in report["clips"] if c["clip_id"] == "clip_a")
    assert clip_a["status"] == "error" and "unreadable video" in clip_a["error"]
    assert clip_a["best_extension_flexion_deg"] is None  # no fabricated metrics
    pair = report["pairs"][0]
    assert pair["status"] == "insufficient_data" and pair["delta_extension_deg"] is None
    assert report["errors"] and report["errors"][0]["clip_id"] == "clip_a"


def test_all_clips_failing_is_nothing_evaluable(tmp_path):
    def broken_pose(path, side="right"):
        raise RuntimeError("boom")

    report, code = run(tmp_path, good_lines(), pose_fn=broken_pose)
    assert code == 2 and report["status"] == "nothing_evaluable"


def test_missing_dependency_is_blocking_exit_3(tmp_path):
    def import_failing_pose(path, side="right"):
        raise ImportError("No module named ultralytics")

    report, code = run(tmp_path, good_lines(), pose_fn=import_failing_pose)
    assert code == 3 and report["status"] == "blocked_integration"


def test_manifest_side_is_authoritative_and_blank_uses_fallback(tmp_path):
    seen = {}

    def recording_pose(path, side="right"):
        seen[Path(path).name] = side
        return fake_pose(path, side)

    lines = [manifest_line("clip_a", "data/a.mp4", "A", side=""), manifest_line("clip_b", "data/b.mp4", "B", side="")]
    report, code = run(tmp_path, lines, side="left", pose_fn=recording_pose)
    assert code == 0 and seen == {"a.mp4": "left", "b.mp4": "left"}

    seen.clear()
    lines = [manifest_line("clip_a", "data/a.mp4", "A", side="right"), manifest_line("clip_b", "data/b.mp4", "B", side="right")]
    run(tmp_path, lines, side="left", pose_fn=recording_pose)
    assert seen == {"a.mp4": "right", "b.mp4": "right"}


def test_outputs_written_with_exact_columns(tmp_path):
    report, code = run(tmp_path, good_lines())
    out = tmp_path / "out"
    with open(out / "clip_metrics.csv", newline="", encoding="utf-8") as handle:
        clip_rows = list(csv.reader(handle))
    with open(out / "pair_comparison.csv", newline="", encoding="utf-8") as handle:
        pair_rows = list(csv.reader(handle))
    assert clip_rows[0] == [
        "clip_id", "session_label", "participant_id", "source_type", "side", "status",
        "total_frames", "valid_frames", "valid_ratio", "best_extension_flexion_deg",
        "observed_rom_deg", "warnings", "error", "edge_rejected_frames",
    ]
    assert pair_rows[0] == [
        "pair_id", "a_clip_id", "b_clip_id", "a_best_extension_flexion_deg",
        "b_best_extension_flexion_deg", "delta_extension_deg", "status", "warnings",
    ]
    assert len(clip_rows) == 4 and len(pair_rows) == 2  # header + 3 clips / header + 1 pair
    delta_index = pair_rows[0].index("delta_extension_deg")
    assert float(pair_rows[1][delta_index]) == pytest.approx(25.0)
    saved = json.loads((out / "report.json").read_text(encoding="utf-8"))
    for key in ("timestamp_utc", "model_name", "evaluation_type", "methodology", "clips", "pairs", "errors", "limitations"):
        assert key in saved


def test_edge_rejected_frames_are_counted(tmp_path):
    def pose_with_edges(path, side="right"):
        frames = fake_pose(path, side)
        if Path(path).name == "a.mp4":
            frames = [dict(f, edge_rejected=["ankle"] if i < 4 else []) for i, f in enumerate(frames)]
        return frames

    report, code = run(tmp_path, good_lines(), pose_fn=pose_with_edges)
    by_id = {c["clip_id"]: c for c in report["clips"]}
    assert by_id["clip_a"]["edge_rejected_frames"] == 4
    assert by_id["clip_b"]["edge_rejected_frames"] is None  # records carry no edge_rejected key: unknown, not 0
    with open(tmp_path / "out" / "clip_metrics.csv", newline="", encoding="utf-8") as handle:
        rows = {r["clip_id"]: r for r in csv.DictReader(handle)}
    assert rows["clip_a"]["edge_rejected_frames"] == "4" and rows["clip_b"]["edge_rejected_frames"] == ""


def test_compare_delta_mismatch_is_flagged(tmp_path):
    def wrong_compare(pre, post):
        return {"status": "ok", "delta_extension_deg": -25.0, "warnings": []}

    report, code = run(tmp_path, good_lines(), compare_fn=wrong_compare)
    assert any("contract mismatch" in w for w in report["pairs"][0]["warnings"])


def test_valid_frame_alias_fields_are_accepted():
    counts = evaluate._frame_counts({"valid_frames": 4, "total_frames": 6, "flexion_deg": [1, None]}, 99)
    assert counts == (6, 4)
    counts = evaluate._frame_counts({"valid_frame_count": 3, "total_frame_count": 5, "valid_frames": 1, "total_frames": 2}, 99)
    assert counts == (5, 3)
    assert evaluate._frame_counts({"flexion_deg": [10.0, None, 20.0]}, 99) == (3, 2)


# ---- CLI -----------------------------------------------------------------

def test_pair_requires_consent_flag(capsys):
    code = evaluate.main(["--pair", "x.mp4", "y.mp4"])
    assert code == 2 and "--consent-confirmed" in capsys.readouterr().out


def test_shipped_manifest_exits_nonzero_while_files_missing(capsys):
    manifest = PROJECT_ROOT / "data" / "manifest.csv"
    if not manifest.is_file():
        pytest.skip("data/manifest.csv not present")
    if (PROJECT_ROOT / "data" / "session_before.mp4").exists():
        pytest.skip("real clips were added; shipped-placeholder behaviour no longer applies")
    code = evaluate.main(["--manifest", str(manifest), "--output-dir", str(PROJECT_ROOT / "outputs" / "_unused")])
    out = capsys.readouterr().out
    assert code == 2 and "MISSING FILE" in out and "session_before.mp4" in out and "Traceback" not in out


def test_help_works_without_ultralytics_import():
    result = subprocess.run([sys.executable, "-c",
                             "import sys, evaluate; assert 'ultralytics' not in sys.modules; "
                             "assert 'cv2' not in sys.modules; evaluate.build_parser().format_help()"],
                            cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_smoke_test_reports_each_contract_function(capsys):
    code = evaluate.main(["--smoke-test"])
    out = capsys.readouterr().out
    assert code in (0, 3)
    for name in ("extract_video_pose", "analyze_session", "compare_sessions"):
        assert name in out


# ---- real metrics.py (skipped when absent) --------------------------------

@pytest.fixture
def metrics():
    return pytest.importorskip("metrics", reason="metrics.py not on disk yet (backend worker): real-metrics tests skipped")


def frames_of(metrics_module, hip, knee, ankle, count=None):
    """Constant-pose frames; the default count follows metrics.MIN_VALID_FRAMES (no hard-coded gate)."""
    if count is None:
        count = evaluate.synthetic_frame_count(metrics_module)
    return evaluate._constant_frames(hip, knee, ankle, count)


def test_synthetic_frame_count_clears_the_metrics_gate(metrics):
    gate = getattr(metrics, "MIN_VALID_FRAMES", 10)
    assert evaluate.synthetic_frame_count(metrics) >= gate
    result = metrics.analyze_session(frames_of(metrics, (0, 1), (0, 0), (1, 0)))
    assert result["status"] == "ok"


def test_real_metrics_geometry_180_and_90(metrics):
    straight = metrics.analyze_session(frames_of(metrics, (0, 1), (0, 0), (0, -1)))
    right_angle = metrics.analyze_session(frames_of(metrics, (0, 1), (0, 0), (1, 0)))
    assert straight["best_extension_flexion_deg"] == pytest.approx(0.0, abs=0.5)
    assert right_angle["best_extension_flexion_deg"] == pytest.approx(90.0, abs=0.5)


def test_real_metrics_excludes_missing_and_low_confidence(metrics):
    count = 2 * getattr(metrics, "MIN_VALID_FRAMES", 10)  # 2 invalid frames still clear the gate
    frames = frames_of(metrics, (0, 1), (0, 0), (1, 0), count=count)
    frames[2]["knee"], frames[2]["knee_conf"] = None, 0.0
    frames[5]["knee_conf"] = 0.1
    result = metrics.analyze_session(frames)
    assert result["flexion_deg"][2] is None and result["flexion_deg"][5] is None
    assert result["valid_ratio"] == pytest.approx((count - 2) / count, abs=0.01)


def test_real_metrics_p10_robust_to_single_outlier(metrics):
    frames = evaluate.build_synthetic_frames(40.0)
    frames[10]["ankle"] = [300.0, 300.0]  # one frame with flexion 0
    result = metrics.analyze_session(frames)
    assert result["best_extension_flexion_deg"] == pytest.approx(40.0, abs=2.0)


def test_real_metrics_delta_sign_40_vs_15(metrics, tmp_path):
    frames_by_file = {"a.mp4": evaluate.build_synthetic_frames(40.0), "b.mp4": evaluate.build_synthetic_frames(15.0)}

    def synthetic_pose(path, side="right"):
        return frames_by_file[Path(path).name]

    manifest = make_project(tmp_path, good_lines()[:2], files=("a.mp4", "b.mp4"))
    report, code = evaluate.run_evaluation(manifest, tmp_path / "out", pose_fn=synthetic_pose)
    assert code == 0 and report["evaluation_type"] == "synthetic_only"
    assert report["pairs"][0]["status"] == "ok"
    assert report["pairs"][0]["delta_extension_deg"] == pytest.approx(25.0, abs=1.0)


def test_synthetic_test_cli(metrics, capsys):
    code = evaluate.main(["--synthetic-test"])
    assert code == 0 and "SYNTHETIC ONLY - not model inference" in capsys.readouterr().out
