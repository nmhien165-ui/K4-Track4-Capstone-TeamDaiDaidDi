"""
=============================================================================
VINMEC REHABTRACK - MODULE PHÂN TÍCH HÌNH HỌC & SO SÁNH TIẾN TRIỂN
File sở hữu: metrics.py
Thành viên: Lê Trung Kiên (Geometry Lead)
Tuân thủ API Contract mục C3 - Bản kế hoạch MVP Hackathon Track 4
=============================================================================
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import math
from typing import List, Dict, Optional, Any


def calculate_knee_flexion(hip: Optional[List[float]], 
                           knee: Optional[List[float]], 
                           ankle: Optional[List[float]]) -> Optional[float]:
    """
    Tính góc gập gối (flexion_deg = 180 - interior_angle).
    - interior_angle: góc tạo bởi vector (hip - knee) và (ankle - knee)
    - flexion_deg gần 0°: chân duỗi thẳng hoàn toàn
    - flexion_deg gần 90°: cẳng chân gập vuông góc với đùi
    Trả về None nếu bất kỳ điểm nào thiếu hoặc độ dài vector quá nhỏ.
    """
    if hip is None or knee is None or ankle is None:
        return None

    p_hip = np.array(hip[:2], dtype=float)
    p_knee = np.array(knee[:2], dtype=float)
    p_ankle = np.array(ankle[:2], dtype=float)

    v1 = p_hip - p_knee
    v2 = p_ankle - p_knee

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 < 1e-6 or norm_v2 < 1e-6:
        return None

    cosine_val = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cosine_val = np.clip(cosine_val, -1.0, 1.0)

    interior_angle = math.degrees(math.acos(cosine_val))
    flexion_deg = 180.0 - interior_angle

    # Đảm bảo 0 <= flexion <= 180
    return float(np.clip(flexion_deg, 0.0, 180.0))


def apply_median_smoothing(flexion_list: List[Optional[float]], window_size: int = 3) -> List[Optional[float]]:
    """
    Làm mượt chuỗi góc hợp lệ bằng median filter cửa sổ nhỏ (3-5 frame).
    Không nội suy qua các đoạn mất khớp (None).
    """
    n = len(flexion_list)
    smoothed = list(flexion_list)
    radius = window_size // 2

    for i in range(n):
        if flexion_list[i] is None:
            continue
        # Thu thập các giá trị lân cận hợp lệ
        neighbors = []
        for j in range(max(0, i - radius), min(n, i + radius + 1)):
            if flexion_list[j] is not None:
                neighbors.append(flexion_list[j])
        if len(neighbors) >= 2:
            smoothed[i] = float(np.median(neighbors))

    return smoothed


def analyze_session(frames: List[Dict[str, Any]], min_conf: float = 0.40) -> Dict[str, Any]:
    """
    Tính góc, P10/P90, số đo và chất lượng cho một video.
    
    Yêu cầu API Contract:
    - Input: frames: list các record dict:
        {"time_s": float, "hip": [x,y]|None, "knee": [x,y]|None, "ankle": [x,y]|None,
         "hip_conf": float, "knee_conf": float, "ankle_conf": float}
    - Output: dict chứa:
        status: "ok" hoặc "insufficient_data"
        time_s: list[float]
        flexion_deg: list[float|None] (mỗi frame 1 giá trị)
        best_extension_flexion_deg: float|None (P10 của chuỗi hợp lệ)
        observed_rom_deg: float|None (P90 - P10)
        valid_ratio: float
        valid_frame_count: int
        total_frame_count: int
        warnings: list[str]
    """
    total_frames = len(frames)
    if total_frames == 0:
        return {
            "status": "insufficient_data",
            "time_s": [],
            "flexion_deg": [],
            "best_extension_flexion_deg": None,
            "observed_rom_deg": None,
            "valid_ratio": 0.0,
            "valid_frame_count": 0,
            "total_frame_count": 0,
            "warnings": ["Video không có frame nào được trích xuất."],
        }

    time_s_list = []
    raw_flexions = []
    warnings = []

    for f in frames:
        time_s_list.append(f.get("time_s", 0.0))
        
        # Kiểm tra confidence gate
        c_hip = f.get("hip_conf", 0.0)
        c_knee = f.get("knee_conf", 0.0)
        c_ankle = f.get("ankle_conf", 0.0)

        if c_hip < min_conf or c_knee < min_conf or c_ankle < min_conf:
            raw_flexions.append(None)
            continue

        hip = f.get("hip")
        knee = f.get("knee")
        ankle = f.get("ankle")

        ang = calculate_knee_flexion(hip, knee, ankle)
        raw_flexions.append(round(ang, 1) if ang is not None else None)

    # Làm mượt nhẹ bằng median filter cửa sổ 3 frame trên các đoạn hợp lệ
    smoothed_flexions = apply_median_smoothing(raw_flexions, window_size=3)

    # Thống kê frame hợp lệ
    valid_values = [v for v in smoothed_flexions if v is not None]
    valid_count = len(valid_values)
    valid_ratio = round(valid_count / float(total_frames), 3)

    # Tiêu chí hoàn thành MVP (Mục A4 & C3):
    # Không tạo kết quả nếu có < 5 frame hợp lệ hoặc valid_ratio < 0.5
    if valid_count < 5 or valid_ratio < 0.5:
        if valid_count < 5:
            warnings.append(f"Số frame hợp lệ quá ít ({valid_count} < 5 frame tối thiểu).")
        if valid_ratio < 0.5:
            warnings.append(f"Tỉ lệ frame hợp lệ thấp ({valid_ratio*100:.1f}% < 50.0%).")

        return {
            "status": "insufficient_data",
            "time_s": time_s_list,
            "flexion_deg": smoothed_flexions,
            "best_extension_flexion_deg": None,
            "observed_rom_deg": None,
            "valid_ratio": valid_ratio,
            "valid_frame_count": valid_count,
            "total_frame_count": total_frames,
            "warnings": warnings,
        }

    # Tính P10 và P90 trên các frame hợp lệ
    valid_arr = np.array(valid_values, dtype=float)
    p10 = float(np.percentile(valid_arr, 10))
    p90 = float(np.percentile(valid_arr, 90))

    best_extension_flexion = round(p10, 1)
    observed_rom = round(p90 - p10, 1)

    return {
        "status": "ok",
        "time_s": time_s_list,
        "flexion_deg": smoothed_flexions,
        "best_extension_flexion_deg": best_extension_flexion,
        "observed_rom_deg": observed_rom,
        "valid_ratio": valid_ratio,
        "valid_frame_count": valid_count,
        "total_frame_count": total_frames,
        "warnings": warnings,
    }


def compare_sessions(pre: Dict[str, Any], post: Dict[str, Any]) -> Dict[str, Any]:
    """
    So sánh hai buổi đánh giá: pre (Buổi A) và post (Buổi B).
    
    Yêu cầu API Contract:
    - Input: pre dict, post dict từ analyze_session
    - Output: dict chứa:
        status: "ok" hoặc "insufficient_data"
        delta_extension_deg: float|None (pre.best - post.best)
        interpretation: str
        warnings: list[str]
    """
    warnings = []
    warnings.extend(pre.get("warnings", []))
    warnings.extend(post.get("warnings", []))

    if pre.get("status") != "ok" or post.get("status") != "ok":
        return {
            "status": "insufficient_data",
            "delta_extension_deg": None,
            "interpretation": "Không đủ dữ liệu tin cậy ở một hoặc cả hai video để so sánh.",
            "warnings": warnings,
        }

    pre_best = pre.get("best_extension_flexion_deg")
    post_best = post.get("best_extension_flexion_deg")

    if pre_best is None or post_best is None:
        return {
            "status": "insufficient_data",
            "delta_extension_deg": None,
            "interpretation": "Thiếu chỉ số duỗi tối đa hợp lệ để tính chênh lệch.",
            "warnings": warnings,
        }

    # delta = pre - post (số dương thể hiện buổi 2 góc gập nhỏ hơn -> chân duỗi thẳng hơn)
    delta = round(pre_best - post_best, 1)

    if delta > 0.5:
        interpretation = (
            f"Góc gập tại vùng duỗi tối đa giảm khoảng {delta:.1f}° giữa hai video mô phỏng "
            f"(chân có xu hướng duỗi thẳng hơn ở buổi sau)."
        )
    elif delta < -0.5:
        interpretation = (
            f"Góc gập tại vùng duỗi tối đa tăng khoảng {abs(delta):.1f}° giữa hai video mô phỏng "
            f"(chân ít duỗi thẳng hơn ở buổi sau)."
        )
    else:
        interpretation = "Không ghi nhận thay đổi đáng kể về góc gập ở vùng duỗi tối đa giữa hai video mô phỏng."

    # Lời nhắc kiểm tra điều kiện quay đồng nhất (Mục A2)
    warnings.append("Lưu ý: Chỉ số so sánh chỉ có ý nghĩa khi hai video giữ cùng góc quay camera và cùng bên chân.")

    return {
        "status": "ok",
        "delta_extension_deg": delta,
        "interpretation": interpretation,
        "warnings": warnings,
    }


# =============================================================================
# KHỐI KIỂM THỬ ĐỘC LẬP THEO MỤC F CỦA KẾ HOẠCH (AGENT KIÊN TEST)
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print(" VINMEC REHABTRACK - METRICS MODULE TEST (LÊ TRUNG KIÊN)")
    print(" Kiểm thử hợp đồng API theo đúng mục C3 & F của Kế hoạch MVP...")
    print("="*70)

    # 1. Test hình học cơ bản (Section F)
    # Hip=(0,1), knee=(0,0), ankle=(0,-1) -> thẳng đứng 180° -> flexion 0°
    f1 = calculate_knee_flexion([0.0, 1.0], [0.0, 0.0], [0.0, -1.0])
    print(f"[TEST 1] Chân thẳng đứng: flexion = {f1:.1f}° (Kỳ vọng: 0.0°)")
    assert abs(f1 - 0.0) < 1e-4, "Lỗi Test 1"

    # Hip=(0,1), knee=(0,0), ankle=(1,0) -> vuông góc 90° -> flexion 90°
    f2 = calculate_knee_flexion([0.0, 1.0], [0.0, 0.0], [1.0, 0.0])
    print(f"[TEST 2] Cẳng chân vuông góc: flexion = {f2:.1f}° (Kỳ vọng: 90.0°)")
    assert abs(f2 - 90.0) < 1e-4, "Lỗi Test 2"

    # 2. Test giả lập Video A (Mô phỏng buổi 1: duỗi chân tối đa đạt flexion ~45°)
    mock_frames_A = []
    for i in range(30):
        t = i * 0.2
        # Flexion dao động 45° (duỗi) đến 85° (co)
        flex_deg = 65.0 + 20.0 * math.cos(i * 0.4)
        rad_interior = math.radians(180.0 - flex_deg)
        # Vector v1 = (0, -100). Đặt v2 sao cho dot(v1, v2)/norm = cos(rad_interior)
        mock_frames_A.append({
            "time_s": round(t, 2),
            "hip": [100.0, 50.0],
            "knee": [100.0, 150.0],
            "ankle": [100.0 + 100.0 * math.sin(rad_interior), 150.0 - 100.0 * math.cos(rad_interior)],
            "hip_conf": 0.92,
            "knee_conf": 0.89,
            "ankle_conf": 0.91,
        })

    res_A = analyze_session(mock_frames_A)
    print("\n[TEST 3] Phân tích Session A:")
    print(f"  Status: {res_A['status']} | Valid Ratio: {res_A['valid_ratio']*100:.1f}%")
    print(f"  Best Extension Flexion (P10): {res_A['best_extension_flexion_deg']}°")
    print(f"  Observed ROM: {res_A['observed_rom_deg']}°")

    # 3. Test giả lập Video B (Mô phỏng buổi 2: duỗi chân tốt hơn, flexion đạt ~25°)
    mock_frames_B = []
    for i in range(30):
        t = i * 0.2
        # Flexion dao động 25° (duỗi thẳng hơn) đến 85° (co)
        flex_deg = 55.0 + 30.0 * math.cos(i * 0.4)
        rad_interior = math.radians(180.0 - flex_deg)
        mock_frames_B.append({
            "time_s": round(t, 2),
            "hip": [100.0, 50.0],
            "knee": [100.0, 150.0],
            "ankle": [100.0 + 100.0 * math.sin(rad_interior), 150.0 - 100.0 * math.cos(rad_interior)],
            "hip_conf": 0.94,
            "knee_conf": 0.91,
            "ankle_conf": 0.88,
        })

    res_B = analyze_session(mock_frames_B)
    print("\n[TEST 4] Phân tích Session B:")
    print(f"  Status: {res_B['status']} | Valid Ratio: {res_B['valid_ratio']*100:.1f}%")
    print(f"  Best Extension Flexion (P10): {res_B['best_extension_flexion_deg']}°")
    print(f"  Observed ROM: {res_B['observed_rom_deg']}°")

    # 4. Test hàm compare_sessions
    comp = compare_sessions(res_A, res_B)
    print("\n[TEST 5] Kết quả so sánh hai buổi (compare_sessions):")
    print(f"  Status: {comp['status']}")
    print(f"  Delta Extension: {comp['delta_extension_deg']}° (Số dương = chân duỗi thẳng hơn)")
    print(f"  Interpretation: {comp['interpretation']}")

    # 5. Test trường hợp thiếu dữ liệu (insufficient_data test)
    bad_frames = [{"time_s": 0.0, "hip": None, "knee": None, "ankle": None, 
                   "hip_conf": 0.1, "knee_conf": 0.1, "ankle_conf": 0.1}] * 10
    res_bad = analyze_session(bad_frames)
    print(f"\n[TEST 6] Test xử lý thiếu mốc khớp: Status = {res_bad['status']} (Kỳ vọng: insufficient_data)")
    assert res_bad['status'] == "insufficient_data", "Lỗi Test 6"

    print("\n" + "="*70)
    print(" TOÀN BỘ UNIT TEST CỦA LÊ TRUNG KIÊN (metrics.py) ĐÃ ĐẠT 100%!")
    print("="*70 + "\n")
