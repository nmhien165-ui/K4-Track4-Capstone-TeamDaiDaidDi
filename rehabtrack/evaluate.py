"""VinMec RehabTrack - evaluation CLI (data / QA worker).

Runs the project pipeline over the clips listed in a manifest:

    extract_video_pose (pose.py) -> analyze_session -> compare_sessions (metrics.py)

and writes clip_metrics.csv, pair_comparison.csv and report.json.

Design rules
- Only the standard library is imported at module level, so ``--help``,
  ``--smoke-test`` and manifest validation work without ultralytics / opencv.
  pose.py and metrics.py are imported lazily, inside functions.
- Nothing here fabricates a measurement: a clip that cannot be processed is
  recorded with status "error" or "insufficient_data" and no numbers.
- Synthetic keypoints are for testing the math only and are always labelled
  "SYNTHETIC ONLY - not model inference".

Exit codes
    0  evaluation finished and at least one clip produced status "ok"
    1  --synthetic-test ran and an assertion failed
    2  invalid manifest / invalid options, or nothing evaluable (no clip "ok")
    3  blocking integration error (pose.py or metrics.py missing / import fails)

Manifest path rule: relative_path is resolved against the project root, which
is the parent of the manifest's folder when that folder is named "data"
(data/manifest.csv -> project root = folder containing data/), otherwise the
manifest's own folder. Absolute paths are used as given.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

MANIFEST_COLUMNS = [
    "clip_id", "relative_path", "participant_id", "session_label", "exercise",
    "side", "camera_view", "source_type", "consent_confirmed", "paired_group",
    "notes",
]
CLIP_METRICS_COLUMNS = [
    "clip_id", "session_label", "participant_id", "source_type", "side",
    "status", "total_frames", "valid_frames", "valid_ratio",
    "best_extension_flexion_deg", "observed_rom_deg", "warnings", "error",
    "edge_rejected_frames",
]
PAIR_COMPARISON_COLUMNS = [
    "pair_id", "a_clip_id", "b_clip_id", "a_best_extension_flexion_deg",
    "b_best_extension_flexion_deg", "delta_extension_deg", "status",
    "warnings",
]

PAIRED_LABELS = ("A", "B")
FAILURE_LABELS = ("failure_occluded", "failure_bad_angle")
VALID_LABELS = PAIRED_LABELS + FAILURE_LABELS
VALID_SOURCE_TYPES = ("staged_real", "synthetic_test", "public_external")
VALID_EXERCISES = ("seated_knee_extension",)
VALID_CAMERA_VIEWS = ("side", "oblique", "unknown")
VALID_CONSENT = ("yes", "no", "not_applicable")
VALID_SIDES = ("left", "right")
REAL_SOURCE_TYPES = ("staged_real", "public_external")

EXIT_OK, EXIT_SYNTHETIC_FAIL, EXIT_INVALID, EXIT_BLOCKED = 0, 1, 2, 3

SYNTHETIC_BANNER = "SYNTHETIC ONLY - not model inference"

METHODOLOGY = [
    "Pose: frozen pretrained Ultralytics YOLO11n-pose via pose.extract_video_pose; no training, no fine-tuning.",
    "Geometry: interior angle between (hip - knee) and (ankle - knee); flexion_deg = 180 - interior angle.",
    "Per clip: best_extension_flexion_deg = P10 of valid flexion angles; observed_rom_deg = P90 - P10.",
    "Comparison: delta_extension_deg = A best extension - B best extension; positive means the knee appeared straighter in B.",
    "Only clips with status ok are paired; failure clips are analysed separately and never paired.",
]
LIMITATIONS = [
    "2D projection only: the angle is measured in the image plane, not in 3D.",
    "Camera angle is not detectable from keypoint confidence; a camera rotated off the side view can give high confidence and a biased angle.",
    "Occlusion or missing keypoints reduce valid frames or yield insufficient_data; they are not corrected.",
    "Joints within the frame-border margin (pose.py edge_margin, default 0.03 of the frame size) are excluded as extrapolated, so hip, knee and foot must stay fully inside the frame; the validity gate (MIN_VALID_FRAMES / MIN_VALID_RATIO) lives in metrics.py.",
    "The border rule and valid_ratio cannot detect a camera that is off the side-on plane or the wrong exercise performed fully inside the frame.",
    "No goniometer or motion-capture ground truth: absolute angular accuracy is unknown and is not claimed.",
    "Sessions are simulated by a volunteer; they are not surgical or clinical outcomes.",
    "No recovery percentage is computed or implied; this is not a diagnosis.",
]
FAILURE_NOTE = (
    "Failure clip, reported as observed only. valid_ratio does not prove that a "
    "bad camera angle is detected: an oblique view can still give confident "
    "keypoints and a biased 2D angle."
)


class BlockingIntegrationError(Exception):
    """pose.py / metrics.py cannot be imported or lack a contract function."""


# --------------------------------------------------------------------------
# Manifest loading and validation
# --------------------------------------------------------------------------

def _project_root_for(manifest_path: Path) -> Path:
    parent = manifest_path.parent
    return parent.parent if parent.name.lower() == "data" else parent


def read_manifest(manifest_path: Path):
    """Return (rows, errors). Rows are dicts with stripped string values."""
    errors = []
    if not manifest_path.is_file():
        return [], [f"Manifest not found: {manifest_path.as_posix()}. "
                    "Fix: pass --manifest <path> or create data/manifest.csv."]
    try:
        with open(manifest_path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            header = [(name or "").strip() for name in (reader.fieldnames or [])]
            missing = [c for c in MANIFEST_COLUMNS if c not in header]
            if missing:
                return [], ["Manifest is missing required column(s): "
                            + ", ".join(missing)
                            + ". Expected header: " + ",".join(MANIFEST_COLUMNS)]
            rows = []
            for line_number, raw in enumerate(reader, start=2):
                row = {(k or "").strip(): (v or "").strip()
                       for k, v in raw.items() if k is not None}
                if not any(row.values()):
                    continue
                row["_line"] = line_number
                rows.append(row)
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        return [], [f"Cannot read manifest {manifest_path.as_posix()}: {exc}"]
    return rows, errors


def validate_manifest(rows, project_root: Path, default_side: str):
    """Validate rows. Returns dict(errors, warnings, clips, pairs).

    clips: normalised clip dicts in manifest order (with resolved_path,
    effective side, file_exists). pairs: {paired_group: {"A": clip, "B": clip}}
    for groups that validated as complete pairs.
    """
    errors, warnings, clips = [], [], []
    seen_ids = {}

    for row in rows:
        where = f"row {row.get('_line', '?')} (clip_id={row.get('clip_id') or '?'})"
        clip = dict(row)
        label = row["session_label"]
        if label.lower() in ("a", "b"):
            label = label.upper()
        clip["session_label"] = label
        for key in ("exercise", "camera_view", "source_type", "consent_confirmed", "side"):
            clip[key] = row[key].lower()

        for key in ("clip_id", "relative_path", "participant_id"):
            if not row[key]:
                errors.append(f"{where}: required value '{key}' is empty.")
        if row["clip_id"]:
            if row["clip_id"] in seen_ids:
                errors.append(f"{where}: duplicate clip_id (also on row {seen_ids[row['clip_id']]}).")
            seen_ids[row["clip_id"]] = row.get("_line", "?")
        if label not in VALID_LABELS:
            errors.append(f"{where}: invalid session_label '{row['session_label']}'; use one of {', '.join(VALID_LABELS)}.")
        if clip["exercise"] not in VALID_EXERCISES:
            errors.append(f"{where}: invalid exercise '{row['exercise']}'; use {', '.join(VALID_EXERCISES)}.")
        if clip["camera_view"] not in VALID_CAMERA_VIEWS:
            errors.append(f"{where}: invalid camera_view '{row['camera_view']}'; use one of {', '.join(VALID_CAMERA_VIEWS)}.")
        if clip["source_type"] not in VALID_SOURCE_TYPES:
            errors.append(f"{where}: invalid source_type '{row['source_type']}'; use one of {', '.join(VALID_SOURCE_TYPES)}.")
        if clip["consent_confirmed"] not in VALID_CONSENT:
            errors.append(f"{where}: invalid consent_confirmed '{row['consent_confirmed']}'; use one of {', '.join(VALID_CONSENT)}.")

        if clip["side"] == "":
            clip["effective_side"], clip["side_source"] = default_side, "cli_fallback"
        elif clip["side"] in VALID_SIDES:
            clip["effective_side"], clip["side_source"] = clip["side"], "manifest"
        else:
            clip["effective_side"], clip["side_source"] = None, "invalid"
            errors.append(f"{where}: invalid side '{row['side']}'; use left or right (or leave blank to use --side).")

        if clip["source_type"] == "staged_real" and clip["consent_confirmed"] != "yes":
            errors.append(f"{where}: consent not confirmed (consent_confirmed='{row['consent_confirmed']}') for a staged_real clip. "
                          "Fix: ask the participant, then set consent_confirmed to yes. Consent is never assumed.")

        if row["relative_path"]:
            raw_path = Path(row["relative_path"])
            resolved = raw_path if raw_path.is_absolute() else project_root / raw_path
        else:
            resolved = None
        clip["resolved_path"] = resolved
        clip["file_exists"] = bool(resolved and resolved.is_file())
        if resolved is not None and not clip["file_exists"]:
            message = (f"MISSING FILE for {where}: expected at {resolved.as_posix()}. "
                       "Fix: record or copy the clip there (see data/VIDEO_GUIDE.md) or correct relative_path in the manifest.")
            if label in FAILURE_LABELS:
                warnings.append(message + " Failure clips are optional, so it is skipped.")
            else:
                errors.append(message)
        if label in PAIRED_LABELS and clip["camera_view"] in ("oblique", "unknown"):
            warnings.append(f"{where}: camera_view is '{clip['camera_view']}'. Comparison assumes a standardised side view; "
                            "the 2D angle may be biased and confidence will not reveal it.")
        clips.append(clip)

    groups = {}
    for clip in clips:
        if clip["session_label"] in PAIRED_LABELS:
            if not clip["paired_group"]:
                errors.append(f"clip_id={clip['clip_id'] or '?'}: session_label {clip['session_label']} needs a paired_group.")
                continue
            groups.setdefault(clip["paired_group"], {"A": [], "B": []})[clip["session_label"]].append(clip)

    pairs = {}
    for group, members in groups.items():
        a_list, b_list = members["A"], members["B"]
        group_ok = True
        for name, members_of in (("A", a_list), ("B", b_list)):
            if len(members_of) > 1:
                ids = ", ".join(c["clip_id"] for c in members_of)
                errors.append(f"paired_group '{group}': duplicate session {name} ({ids}); exactly one {name} per pair.")
                group_ok = False
            if not members_of:
                present = "A" if name == "B" else "B"
                errors.append(f"paired_group '{group}': has session {present} but no session {name}; "
                              "each pair needs exactly one A and one B. A failure clip never counts as B.")
                group_ok = False
        if not group_ok:
            continue
        clip_a, clip_b = a_list[0], b_list[0]
        for field, label in (("participant_id", "participant_id"), ("exercise", "exercise")):
            if clip_a[field] != clip_b[field]:
                errors.append(f"paired_group '{group}': {label} mismatch (A={clip_a[field]}, B={clip_b[field]}).")
                group_ok = False
        if clip_a["effective_side"] != clip_b["effective_side"]:
            errors.append(f"paired_group '{group}': side mismatch between pair (A={clip_a['effective_side']}, "
                          f"B={clip_b['effective_side']}). Both sessions must use the same anatomical leg.")
            group_ok = False
        if clip_a["camera_view"] != clip_b["camera_view"]:
            warnings.append(f"paired_group '{group}': camera_view differs (A={clip_a['camera_view']}, B={clip_b['camera_view']}); comparison may be biased.")
        if clip_a["source_type"] != clip_b["source_type"]:
            warnings.append(f"paired_group '{group}': source_type differs (A={clip_a['source_type']}, B={clip_b['source_type']}).")
        if group_ok:
            pairs[group] = {"A": clip_a, "B": clip_b}
    return {"errors": errors, "warnings": warnings, "clips": clips, "pairs": pairs}


# --------------------------------------------------------------------------
# Lazy integration with pose.py / metrics.py
# --------------------------------------------------------------------------

def _import_project_module(name: str):
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        return importlib.import_module(name)
    except Exception as exc:  # ImportError, SyntaxError, or errors at import time
        raise BlockingIntegrationError(f"cannot import {name}.py: {type(exc).__name__}: {exc}") from exc


def _require_function(module, module_name: str, function_name: str):
    function = getattr(module, function_name, None)
    if not callable(function):
        raise BlockingIntegrationError(f"{module_name}.py has no callable {function_name}")
    return function


def resolve_pipeline(pose_fn, analyze_fn, compare_fn):
    """Fill in defaults lazily. Returns (pose_fn, analyze_fn, compare_fn, model_name, uses_real_pose)."""
    model_name = "none (injected pose_fn)"
    uses_real_pose = pose_fn is None
    if pose_fn is None:
        pose_module = _import_project_module("pose")
        pose_fn = _require_function(pose_module, "pose", "extract_video_pose")
        getter = getattr(pose_module, "get_model_name", None)
        try:
            model_name = str(getter()) if callable(getter) else "unknown"
        except Exception:
            model_name = "unknown"
    if analyze_fn is None or compare_fn is None:
        metrics_module = _import_project_module("metrics")
        if analyze_fn is None:
            analyze_fn = _require_function(metrics_module, "metrics", "analyze_session")
        if compare_fn is None:
            compare_fn = _require_function(metrics_module, "metrics", "compare_sessions")
    return pose_fn, analyze_fn, compare_fn, model_name, uses_real_pose


# --------------------------------------------------------------------------
# Per-clip processing
# --------------------------------------------------------------------------

def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _frame_counts(analysis: dict, frame_total: int):
    """Contract names *_count first, then the *_frames alias, then derive."""
    series = analysis.get("flexion_deg")
    total = analysis.get("total_frame_count")
    if total is None:
        total = analysis.get("total_frames")
    if total is None:
        total = len(series) if isinstance(series, (list, tuple)) else frame_total
    valid = analysis.get("valid_frame_count")
    if valid is None:
        valid = analysis.get("valid_frames")
    if valid is None and isinstance(series, (list, tuple)):
        valid = sum(1 for item in series if item is not None)
    return total, valid


def _empty_clip_result(clip: dict) -> dict:
    return {
        "clip_id": clip["clip_id"], "session_label": clip["session_label"],
        "participant_id": clip["participant_id"], "source_type": clip["source_type"],
        "side": clip["effective_side"], "side_source": clip["side_source"],
        "camera_view": clip["camera_view"], "paired_group": clip["paired_group"],
        "relative_path": clip["relative_path"], "status": "not_processed",
        "error": "", "total_frames": None, "valid_frames": None,
        "valid_ratio": None, "best_extension_flexion_deg": None,
        "observed_rom_deg": None, "warnings": [], "time_s": [], "flexion_deg": [],
        "edge_rejected_frames": None,
        "note": FAILURE_NOTE if clip["session_label"] in FAILURE_LABELS else "",
    }


def _count_edge_rejected_frames(frames):
    """Frames whose pose record lists edge-rejected joints; None when no record carries the key."""
    if not isinstance(frames, (list, tuple)):
        return None
    records = [f for f in frames if isinstance(f, dict) and "edge_rejected" in f]
    if not records:
        return None
    return sum(1 for f in records if f.get("edge_rejected"))


def process_clip(clip: dict, pose_fn, analyze_fn):
    """Return (result_dict, pose_returned). Never raises except for ImportError."""
    result = _empty_clip_result(clip)
    pose_returned = False
    try:
        frames = pose_fn(str(clip["resolved_path"]), side=clip["effective_side"])
        pose_returned = True
        analysis = analyze_fn(frames)
    except ImportError as exc:
        raise BlockingIntegrationError(f"pose/metrics dependency missing while processing {clip['clip_id']}: {exc}") from exc
    except Exception as exc:
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result, pose_returned

    if not isinstance(analysis, dict) or analysis.get("status") not in ("ok", "insufficient_data"):
        result["status"] = "error"
        result["error"] = "analyze_session returned no usable 'status' (expected 'ok' or 'insufficient_data')"
        return result, pose_returned

    total, valid = _frame_counts(analysis, len(frames) if hasattr(frames, "__len__") else 0)
    result.update({
        "status": analysis["status"], "total_frames": total, "valid_frames": valid,
        "valid_ratio": analysis.get("valid_ratio"),
        "best_extension_flexion_deg": analysis.get("best_extension_flexion_deg"),
        "observed_rom_deg": analysis.get("observed_rom_deg"),
        "warnings": _as_list(analysis.get("warnings")),
        "time_s": list(analysis.get("time_s") or []),
        "flexion_deg": list(analysis.get("flexion_deg") or []),
        "edge_rejected_frames": _count_edge_rejected_frames(frames),
    })
    if analysis["status"] == "ok" and result["best_extension_flexion_deg"] is None:
        result["status"] = "insufficient_data"
        result["warnings"].append("analysis reported ok but gave no best_extension_flexion_deg; treated as insufficient_data")
    if analysis["status"] == "insufficient_data":
        result["best_extension_flexion_deg"] = None
        result["observed_rom_deg"] = None
    return result, pose_returned


def compare_pair(group: str, clip_a: dict, clip_b: dict, result_a: dict, result_b: dict, compare_fn) -> dict:
    pair = {
        "pair_id": group, "participant_id": clip_a["participant_id"],
        "side": clip_a["effective_side"], "a_clip_id": clip_a["clip_id"],
        "b_clip_id": clip_b["clip_id"],
        "a_best_extension_flexion_deg": result_a["best_extension_flexion_deg"],
        "b_best_extension_flexion_deg": result_b["best_extension_flexion_deg"],
        "delta_extension_deg": None, "status": "insufficient_data",
        "warnings": [], "cause": "", "interpretation": "",
    }
    not_ok = [f"{r['clip_id']} (status={r['status']})" for r in (result_a, result_b) if r["status"] != "ok"]
    if not_ok:
        pair["cause"] = "both sessions must have status ok; not ok: " + ", ".join(not_ok)
        return pair
    try:
        comparison = compare_fn(_analysis_view(result_a), _analysis_view(result_b))
    except ImportError as exc:
        raise BlockingIntegrationError(f"compare_sessions dependency missing: {exc}") from exc
    except Exception as exc:
        pair["status"] = "error"
        pair["cause"] = f"compare_sessions failed: {type(exc).__name__}: {exc}"
        return pair
    if not isinstance(comparison, dict) or comparison.get("status") not in ("ok", "insufficient_data"):
        pair["status"] = "error"
        pair["cause"] = "compare_sessions returned no usable 'status'"
        return pair
    pair["status"] = comparison["status"]
    pair["delta_extension_deg"] = comparison.get("delta_extension_deg") if comparison["status"] == "ok" else None
    pair["warnings"] = _as_list(comparison.get("warnings"))
    pair["interpretation"] = str(comparison.get("interpretation") or "")
    if pair["status"] == "ok" and pair["delta_extension_deg"] is None:
        pair["status"] = "insufficient_data"
        pair["cause"] = "compare_sessions reported ok but gave no delta_extension_deg"
    if pair["status"] == "ok":
        expected = result_a["best_extension_flexion_deg"] - result_b["best_extension_flexion_deg"]
        if abs(float(pair["delta_extension_deg"]) - expected) > 0.01:
            pair["warnings"].append(f"contract mismatch: delta_extension_deg={pair['delta_extension_deg']} but A - B = {expected:.3f}")
    return pair


def _analysis_view(result: dict) -> dict:
    """Rebuild the analyze_session-shaped dict from a clip result for compare_sessions."""
    return {
        "status": result["status"], "time_s": result["time_s"],
        "flexion_deg": result["flexion_deg"],
        "best_extension_flexion_deg": result["best_extension_flexion_deg"],
        "observed_rom_deg": result["observed_rom_deg"],
        "valid_ratio": result["valid_ratio"],
        "valid_frame_count": result["valid_frames"],
        "total_frame_count": result["total_frames"],
        "valid_frames": result["valid_frames"], "total_frames": result["total_frames"],
        "warnings": list(result["warnings"]),
    }


# --------------------------------------------------------------------------
# Output files
# --------------------------------------------------------------------------

def _clean(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "tolist"):
        return _clean(value.tolist())
    return str(value)


def _cell(value):
    value = _clean(value)
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        return " | ".join(str(v) for v in value)
    return value


def write_outputs(output_dir: Path, clip_results, pair_results, report: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "clip_metrics.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CLIP_METRICS_COLUMNS)
        for result in clip_results:
            writer.writerow([_cell(result.get(column)) for column in CLIP_METRICS_COLUMNS])
    with open(output_dir / "pair_comparison.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PAIR_COMPARISON_COLUMNS)
        for pair in pair_results:
            writer.writerow([_cell(pair.get(column)) for column in PAIR_COMPARISON_COLUMNS])
    with open(output_dir / "report.json", "w", encoding="utf-8") as handle:
        json.dump(_clean(report), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


# --------------------------------------------------------------------------
# Core entry point (testable)
# --------------------------------------------------------------------------

def run_evaluation(manifest_path, output_dir, side="right", pose_fn=None,
                   analyze_fn=None, compare_fn=None, rows=None, project_root=None):
    """Evaluate a manifest. Returns (report_dict, exit_code).

    pose_fn(video_path, side=...) -> frames; analyze_fn(frames) -> dict;
    compare_fn(pre, post) -> dict. Any left as None is imported lazily from
    pose.py / metrics.py. rows/project_root let the CLI evaluate an in-memory
    manifest (--pair); manifest_path is then used only for the report.
    """
    output_dir = Path(output_dir)
    if rows is None:
        manifest = Path(manifest_path).resolve()
        rows, read_errors = read_manifest(manifest)
        root = _project_root_for(manifest)
        manifest_label = manifest.as_posix()
        if read_errors:
            return _invalid_report(manifest_label, read_errors, []), EXIT_INVALID
    else:
        root = Path(project_root) if project_root else Path.cwd()
        manifest_label = str(manifest_path) if manifest_path else "in-memory manifest (--pair)"

    validation = validate_manifest(rows, root, side)
    if validation["errors"]:
        return _invalid_report(manifest_label, validation["errors"], validation["warnings"]), EXIT_INVALID

    try:
        pose_fn, analyze_fn, compare_fn, model_name, uses_real_pose = resolve_pipeline(pose_fn, analyze_fn, compare_fn)
        clip_results, real_processed = {}, 0
        for clip in validation["clips"]:
            if not clip["file_exists"]:
                result = _empty_clip_result(clip)
                result["status"] = "skipped_missing_file"
                result["error"] = "optional failure clip file not found; skipped"
                clip_results[clip["clip_id"]] = result
                continue
            result, pose_returned = process_clip(clip, pose_fn, analyze_fn)
            clip_results[clip["clip_id"]] = result
            if uses_real_pose and pose_returned and clip["source_type"] in REAL_SOURCE_TYPES:
                real_processed += 1
        pair_results = [
            compare_pair(group, members["A"], members["B"], clip_results[members["A"]["clip_id"]],
                         clip_results[members["B"]["clip_id"]], compare_fn)
            for group, members in validation["pairs"].items()
        ]
    except BlockingIntegrationError as exc:
        return {"status": "blocked_integration", "error": str(exc),
                "validation_warnings": validation["warnings"]}, EXIT_BLOCKED

    ordered = list(clip_results.values())
    ok_count = sum(1 for r in ordered if r["status"] == "ok")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": model_name,
        "evaluation_type": "real_video" if real_processed > 0 else "synthetic_only",
        "real_videos_processed": real_processed,
        "manifest": manifest_label,
        "side_fallback": side,
        "methodology": METHODOLOGY,
        "summary": {
            "clips_in_manifest": len(ordered),
            "files_found": sum(1 for c in validation["clips"] if c["file_exists"]),
            "clips_processed": sum(1 for r in ordered if r["status"] in ("ok", "insufficient_data", "error")),
            "clips_ok": ok_count,
            "clips_insufficient_data": sum(1 for r in ordered if r["status"] == "insufficient_data"),
            "clips_error": sum(1 for r in ordered if r["status"] == "error"),
            "pairs": len(pair_results),
        },
        "clips": ordered,
        "pairs": pair_results,
        "errors": [{"clip_id": r["clip_id"], "error": r["error"]} for r in ordered if r["status"] == "error"],
        "validation_warnings": validation["warnings"],
        "limitations": LIMITATIONS,
        "ground_truth": "none: no goniometer reference, so absolute accuracy is not claimed",
    }
    exit_code = EXIT_OK if ok_count > 0 else EXIT_INVALID
    if exit_code == EXIT_INVALID:
        report["status"] = "nothing_evaluable"
    else:
        report["status"] = "evaluated"
    write_outputs(output_dir, ordered, pair_results, report)
    return report, exit_code


def _invalid_report(manifest_label, errors, warnings):
    return {"status": "invalid_manifest", "manifest": manifest_label,
            "errors": list(errors), "validation_warnings": list(warnings)}


def print_summary(report: dict, exit_code: int, output_dir: Path):
    print("RehabTrack evaluation summary")
    if report.get("status") == "invalid_manifest":
        print(f"  manifest: {report.get('manifest')}")
        print(f"  INVALID MANIFEST: {len(report['errors'])} problem(s). Nothing was evaluated and no output files were written.")
        for message in report["errors"]:
            print(f"   - {message}")
        for message in report.get("validation_warnings", []):
            print(f"   * warning: {message}")
        return
    if report.get("status") == "blocked_integration":
        print(f"  BLOCKED: {report['error']}")
        print("  Fix: make pose.py / metrics.py importable (and install requirements) before running a real evaluation.")
        return
    s = report["summary"]
    print(f"  manifest: {report['manifest']}")
    print(f"  evaluation_type: {report['evaluation_type']} (real videos processed: {report['real_videos_processed']}, model: {report['model_name']})")
    print(f"  clips in manifest: {s['clips_in_manifest']} | files found: {s['files_found']} | processed: {s['clips_processed']} "
          f"| ok: {s['clips_ok']} | insufficient_data: {s['clips_insufficient_data']} | failed: {s['clips_error']}")
    for r in report["clips"]:
        detail = r["error"] or ""
        ratio = "n/a" if r["valid_ratio"] is None else f"{float(r['valid_ratio']):.2f}"
        edge = f" edge_rejected_frames={r['edge_rejected_frames']}" if r.get("edge_rejected_frames") else ""
        print(f"   - {r['clip_id']} [{r['session_label']}] status={r['status']} valid_ratio={ratio}{edge} {detail}".rstrip())
    for p in report["pairs"]:
        if p["status"] == "ok":
            delta = float(p["delta_extension_deg"])
            print(f"  pair {p['pair_id']}: A={float(p['a_best_extension_flexion_deg']):.1f} deg, B={float(p['b_best_extension_flexion_deg']):.1f} deg, "
                  f"delta={delta:+.1f} deg (positive = knee looked straighter in B)")
        else:
            print(f"  pair {p['pair_id']}: {p['status']} - {p['cause']}")
    if not report["pairs"]:
        print("  no A/B pair was compared.")
    for message in report["validation_warnings"]:
        print(f"  warning: {message}")
    print("  Limitations: 2D projection; camera angle not detected by confidence; occlusion reduces data; no goniometer ground truth,")
    print("  so absolute accuracy is unknown; simulated sessions are not surgical outcomes; no recovery percentage.")
    print(f"  Outputs: {output_dir.as_posix()}/clip_metrics.csv, pair_comparison.csv, report.json")


# --------------------------------------------------------------------------
# Synthetic geometry test and smoke test
# --------------------------------------------------------------------------

def build_synthetic_frames(best_extension_deg: float, rest_flexion_deg: float = 90.0,
                           repetitions: int = 3, fps: float = 5.0):
    """Keypoint frames (no video) whose flexion repeatedly goes rest -> best_extension_deg -> rest.

    Geometry in image coordinates: hip=(100,300), knee=(200,300); the ankle is
    knee + 100*(cos f, sin f), which gives flexion f exactly (f=0 straight leg,
    f=90 shank pointing down). Each repetition holds 6 of 15 frames at the
    extreme so the P10 equals best_extension_deg.
    """
    cycle = [rest_flexion_deg] * 3
    cycle += [rest_flexion_deg + (best_extension_deg - rest_flexion_deg) * k / 3 for k in (1, 2)]
    cycle += [best_extension_deg] * 6
    cycle += [best_extension_deg + (rest_flexion_deg - best_extension_deg) * k / 3 for k in (1, 2)]
    cycle += [rest_flexion_deg] * 2
    flexions = cycle * repetitions
    frames = []
    for index, flexion in enumerate(flexions):
        radians = math.radians(flexion)
        frames.append({
            "time_s": index / fps, "hip": [100.0, 300.0], "knee": [200.0, 300.0],
            "ankle": [200.0 + 100.0 * math.cos(radians), 300.0 + 100.0 * math.sin(radians)],
            "hip_conf": 0.9, "knee_conf": 0.9, "ankle_conf": 0.9,
        })
    return frames


def synthetic_frame_count(metrics_module=None, margin: int = 2) -> int:
    """Frames needed so synthetic clips clear the validity gate: follows metrics.MIN_VALID_FRAMES
    when the module defines it (fallback 10), never fewer than 12."""
    minimum = getattr(metrics_module, "MIN_VALID_FRAMES", 10) if metrics_module is not None else 10
    return max(12, int(minimum) + margin)


def _constant_frames(hip, knee, ankle, count=12):
    return [{"time_s": i * 0.2, "hip": list(hip), "knee": list(knee), "ankle": list(ankle),
             "hip_conf": 0.9, "knee_conf": 0.9, "ankle_conf": 0.9} for i in range(count)]


def run_synthetic_test(tolerance_deg: float = 1.0):
    """Returns (exit_code, lines). Uses the real metrics module if importable."""
    lines = [SYNTHETIC_BANNER]
    try:
        metrics_module = _import_project_module("metrics")
        analyze = _require_function(metrics_module, "metrics", "analyze_session")
        compare = _require_function(metrics_module, "metrics", "compare_sessions")
    except BlockingIntegrationError as exc:
        return EXIT_BLOCKED, lines + [f"BLOCKED: metrics.py missing or not importable ({exc})"]

    failures = []
    frame_count = synthetic_frame_count(metrics_module)

    def check(name, actual, expected, tolerance=tolerance_deg):
        passed = actual is not None and abs(float(actual) - expected) <= tolerance
        lines.append(f"  [{'PASS' if passed else 'FAIL'}] {name}: got {actual}, expected {expected} +/- {tolerance}")
        if not passed:
            failures.append(name)

    try:
        straight = analyze(_constant_frames((0, 1), (0, 0), (0, -1), frame_count))
        check("hip=(0,1) knee=(0,0) ankle=(0,-1) -> flexion 0", straight.get("best_extension_flexion_deg"), 0.0)
        right_angle = analyze(_constant_frames((0, 1), (0, 0), (1, 0), frame_count))
        check("hip=(0,1) knee=(0,0) ankle=(1,0) -> flexion 90", right_angle.get("best_extension_flexion_deg"), 90.0)
        session_a = analyze(build_synthetic_frames(40.0))
        session_b = analyze(build_synthetic_frames(15.0))
        check("session A best extension flexion", session_a.get("best_extension_flexion_deg"), 40.0)
        check("session B best extension flexion", session_b.get("best_extension_flexion_deg"), 15.0)
        comparison = compare(session_a, session_b)
        check("delta_extension_deg (A 40 - B 15)", comparison.get("delta_extension_deg"), 25.0)
    except Exception as exc:
        return EXIT_SYNTHETIC_FAIL, lines + [f"FAIL: metrics raised {type(exc).__name__}: {exc}"]

    lines.append("synthetic test: " + ("PASSED" if not failures else f"FAILED ({len(failures)} check(s))"))
    lines.append(SYNTHETIC_BANNER)
    return (EXIT_OK if not failures else EXIT_SYNTHETIC_FAIL), lines


def run_smoke_test():
    """Check imports and contract functions only; no video, no weights. Returns (exit_code, lines)."""
    import importlib.util
    lines = ["Smoke test: imports and API presence only (no video opened, no weights loaded)"]
    blocked = False
    for package in ("numpy", "cv2", "ultralytics"):
        found = importlib.util.find_spec(package) is not None
        lines.append(f"  dependency {package}: {'available' if found else 'NOT INSTALLED (BLOCKED for real inference)'}")
    for module_name, functions in (("pose", ["extract_video_pose"]), ("metrics", ["analyze_session", "compare_sessions"])):
        try:
            module = _import_project_module(module_name)
        except BlockingIntegrationError as exc:
            blocked = True
            for function_name in functions:
                lines.append(f"  BLOCKED: {module_name}.{function_name} unavailable ({exc})")
            continue
        for function_name in functions:
            present = callable(getattr(module, function_name, None))
            blocked = blocked or not present
            lines.append(f"  {module_name}.{function_name}: {'available' if present else 'BLOCKED: missing'}")
        if module_name == "pose":
            lines.append(f"  pose.get_model_name: {'available' if callable(getattr(module, 'get_model_name', None)) else 'absent (report will say unknown)'}")
    lines.append("smoke test: " + ("BLOCKED" if blocked else "all contract functions available"))
    return (EXIT_BLOCKED if blocked else EXIT_OK), lines


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate staged seated-knee-extension clips with the frozen pretrained pose pipeline.",
        epilog="Exit codes: 0 ok; 1 synthetic assertion failed; 2 invalid manifest or nothing evaluable; 3 blocking integration error. "
               "relative_path is resolved against the project root (parent of data/ when the manifest is in data/).")
    parser.add_argument("--manifest", default="data/manifest.csv", help="manifest CSV (default: data/manifest.csv)")
    parser.add_argument("--output-dir", default="outputs/evaluation", help="output folder (generated, not source data)")
    parser.add_argument("--side", choices=VALID_SIDES, default="right", help="fallback leg when a manifest row leaves side blank")
    parser.add_argument("--smoke-test", action="store_true", help="check imports / API presence only; no video, no weights")
    parser.add_argument("--synthetic-test", action="store_true", help="validate geometry with generated keypoints (not model inference)")
    parser.add_argument("--pair", nargs=2, metavar=("PRE_A", "POST_B"), help="evaluate two clips without a manifest (requires --consent-confirmed)")
    parser.add_argument("--consent-confirmed", action="store_true", help="the participant has confirmed consent; required with --pair")
    return parser


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)

    if args.smoke_test:
        code, lines = run_smoke_test()
        print("\n".join(lines))
        return code
    if args.synthetic_test:
        code, lines = run_synthetic_test()
        print("\n".join(lines))
        return code

    output_dir = Path(args.output_dir)
    if args.pair:
        if not args.consent_confirmed:
            print("REFUSED: --pair needs --consent-confirmed. Confirm with the participant first; consent is never assumed.")
            return EXIT_INVALID
        base = {"participant_id": "participant_01", "exercise": "seated_knee_extension", "side": args.side,
                "camera_view": "unknown", "source_type": "staged_real", "consent_confirmed": "yes",
                "paired_group": "pair_cli", "notes": "built from --pair; camera view assumed unknown", "_line": 0}
        rows = [dict(base, clip_id="cli_session_a", relative_path=args.pair[0], session_label="A"),
                dict(base, clip_id="cli_session_b", relative_path=args.pair[1], session_label="B")]
        report, code = run_evaluation(None, output_dir, args.side, rows=rows, project_root=Path.cwd())
    else:
        report, code = run_evaluation(args.manifest, output_dir, args.side)
    print_summary(report, code, output_dir)
    return code


if __name__ == "__main__":
    sys.exit(main())
