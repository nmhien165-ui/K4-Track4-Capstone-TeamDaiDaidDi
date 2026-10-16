# Alternatives (reference only)

Teammate implementations that were merged into the repo history and kept here for comparison. **Nothing in this folder is imported by `app.py`, `evaluate.py` or the tests.** Each file is preserved unmodified from the branch it came from; the active code is `rehabtrack/pose.py` and `rehabtrack/metrics.py`.

| File | Owner | Source branch | What it is |
|---|---|---|---|
| `hien_pose.py` | Hien | `nguyenminhhien` (`rehabtrack/pose.py`) | YOLO11n-pose keypoint extraction from video, with a cached model load. |
| `kien_metrics.py` | Kien | `kien` (root `metrics.py`) | Knee-angle metrics and session comparison. |
| `kien_demo_MOCK_chart.png` | Kien | `kien` (root `rehab_comparison_demo.png`) | **MOCK chart with invented numbers. It is not a measurement** and must not be quoted as a result. |

## How the active code differs (from the audit on real clips)

- `hien_pose.py`: no rejection of joints at the frame border, and time-based sampling can land on slightly different frames than the active `pose.py`.
- `kien_metrics.py`: a looser gate (50% valid ratio, 5 valid frames) lets wrong-exercise clips through as `ok`; the active code gates at >= 60% and >= 10 valid frames and returns `insufficient_data` instead.
- `kien_metrics.py`: returns NaN for NaN coordinates and raises on malformed frames, instead of marking the frame invalid and continuing.
- The active `pose.py` / `metrics.py` supersede both while keeping Hien's idea of cached, lock-protected model loading and the `RuntimeError` wrapper around inference.
