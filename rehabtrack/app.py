"""Giao diện của Như Ý. Chạy: streamlit run rehabtrack/app.py."""

import hashlib
import math
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st


def load_pipeline():
    # Import khi chạy ứng dụng để vẫn xem được UI lúc chờ module của nhóm.
    from pose import extract_video_pose
    from metrics import analyze_session, compare_sessions

    return extract_video_pose, analyze_session, compare_sessions


def number(value):
    """None/NaN không phải góc 0°."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def angle(value, signed=False):
    value = number(value)
    if value is None:
        return "—"
    return f"{value:+.1f}°" if signed else f"{value:.1f}°"


def show_warnings(result):
    for warning in result.get("warnings", []):
        st.warning(str(warning))


def render_session(label, result):
    st.subheader(f"Buổi {label}")
    ok = result.get("status") == "ok"
    st.metric(
        "Góc gập tại vùng duỗi tối đa (P10)",
        angle(result.get("best_extension_flexion_deg")) if ok else "—",
    )
    ratio = number(result.get("valid_ratio"))
    st.metric("Khung hình hợp lệ", f"{ratio:.0%}" if ratio is not None else "—")
    valid = result.get("valid_frame_count")
    total = result.get("total_frame_count")
    if valid is not None and total is not None:
        st.caption(f"{valid}/{total} khung hình được lấy mẫu có dữ liệu hợp lệ.")

    if not ok:
        st.warning(
            "INSUFFICIENT_DATA — Không đủ dữ liệu để công bố số đo của buổi này. "
            "Hãy quay lại từ bên hông, thấy rõ hông, gối và cổ chân."
        )
    show_warnings(result)

    times = result.get("time_s", [])
    flexion = result.get("flexion_deg", [])
    if len(times) != len(flexion):
        st.warning("Dữ liệu biểu đồ không khớp: số thời điểm khác số góc. Cần kiểm tra module metrics.")
        return
    if times:
        chart = pd.DataFrame(
            {"Thời gian (giây)": times, "Góc gập gối (°)": [number(x) for x in flexion]}
        ).set_index("Thời gian (giây)")
        if chart["Góc gập gối (°)"].notna().any():
            st.line_chart(chart, x_label="Thời gian (giây)", y_label="Góc gập gối (°)")
            st.caption("Điểm thiếu được giữ trống, không thay bằng 0°.")
        else:
            st.info("Chưa có góc hợp lệ để vẽ biểu đồ.")
    else:
        st.info("Chưa có dữ liệu theo thời gian.")


def analyze_uploads(uploads, side, pipeline, progress):
    extract_video_pose, analyze_session, compare_sessions = pipeline
    paths = []
    results = []
    try:
        for index, upload in enumerate(uploads):
            label = "A" if index == 0 else "B"
            progress.progress(index / 2, text=f"Đang phân tích video buổi {label}…")
            suffix = Path(upload.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                paths.append(Path(handle.name))
                handle.write(upload.getvalue())
            frames = extract_video_pose(str(paths[-1]), side=side, target_fps=5.0)
            results.append(analyze_session(frames, min_conf=0.40))
        comparison = compare_sessions(results[0], results[1])
        progress.progress(1.0, text="Đã phân tích xong hai video.")
        return {"a": results[0], "b": results[1], "comparison": comparison}
    finally:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                st.warning("Một file video tạm chưa xóa được vì đang được sử dụng.")


def main():
    st.set_page_config(page_title="VinMec RehabTrack", page_icon="🦵", layout="wide")
    st.title("VinMec RehabTrack")
    st.caption("So sánh khả năng duỗi gối qua hai video • Bài tập duỗi gối khi ngồi")
    st.warning("DEMO MÔ PHỎNG — KHÔNG DÀNH CHO CHẨN ĐOÁN HAY RA QUYẾT ĐỊNH Y KHOA")
    with st.expander("Hướng dẫn quay để so sánh hai buổi", expanded=True):
        st.markdown(
            "- Quay **cùng người, cùng chân, cùng bài tập**, giữ vị trí camera và ánh sáng tương đồng.\n"
            "- Camera nhìn ngang từ bên hông, thấy rõ **hông – gối – cổ chân**, một người trong khung.\n"
            "- Ngồi trên ghế ổn định; mỗi clip 6–10 giây, thực hiện 2–3 lần duỗi chậm trong phạm vi thoải mái.\n"
            "- Dùng hai buổi mô phỏng của thành viên tự nguyện; tránh để lộ thông tin cá nhân."
        )
        st.caption("Camera lệch có thể làm sai số đo dù AI vẫn nhận diện đủ khớp.")

    side = st.selectbox(
        "Chân phân tích (áp dụng cho cả hai video)",
        options=["right", "left"],
        format_func=lambda value: "Chân phải" if value == "right" else "Chân trái",
    )
    columns = st.columns(2)
    uploads = []
    for label, column in zip(("A", "B"), columns):
        with column:
            upload = st.file_uploader(
                f"Video buổi {label}", type=["mp4", "mov", "avi"], key=f"video_{label}"
            )
            uploads.append(upload)
            if upload is not None:
                st.video(upload.getvalue())
                st.caption(upload.name)
    st.caption("Hai video xem độc lập; biểu đồ chưa căn chỉnh thời điểm bắt đầu từng động tác.")

    signature = (
        side,
        tuple(
            (upload.name, hashlib.sha256(upload.getbuffer()).hexdigest())
            if upload is not None else None
            for upload in uploads
        ),
    )
    if st.session_state.get("input_signature") != signature:
        st.session_state["input_signature"] = signature
        st.session_state.pop("report", None)
        st.session_state.pop("analysis_error", None)

    pipeline = None
    try:
        pipeline = load_pipeline()
    except ImportError as exc:
        st.info(
            "Giao diện đã sẵn sàng. Để phân tích thật, cần pose.py của Hiển và metrics.py "
            "của Kiên trong cùng thư mục với app.py, cùng các thư viện của nhóm."
        )
        with st.expander("Chi tiết module cần bổ sung"):
            st.code(str(exc), language="text")
    except Exception as exc:
        st.error(f"Không khởi tạo được module phân tích: {exc}")

    ready = all(upload is not None and upload.size > 0 for upload in uploads)
    if not ready:
        st.info("Tải đủ hai video có nội dung để bắt đầu phân tích.")

    if st.button("Phân tích", type="primary", disabled=not ready or pipeline is None):
        st.session_state.pop("report", None)
        st.session_state.pop("analysis_error", None)
        progress = st.progress(0, text="Chuẩn bị video…")
        try:
            with st.spinner("Đang nhận diện khớp và tính góc. Lần đầu có thể cần tải mô hình…"):
                st.session_state["report"] = analyze_uploads(uploads, side, pipeline, progress)
        except Exception as exc:
            st.session_state["analysis_error"] = str(exc)
        finally:
            progress.empty()

    if "analysis_error" in st.session_state:
        st.error(
            "Không phân tích được video. Hãy kiểm tra file có phát được, đúng định dạng "
            "và các module/model đã sẵn sàng, sau đó thử lại."
        )
        with st.expander("Chi tiết lỗi để nhóm kiểm tra"):
            st.code(st.session_state["analysis_error"], language="text")

    report = st.session_state.get("report")
    if report is None:
        return
    st.divider()
    st.header("Báo cáo so sánh")
    st.caption(f"Chân đang đo: {'phải' if side == 'right' else 'trái'} • Buổi A → Buổi B")
    for label, column, key in zip(("A", "B"), st.columns(2), ("a", "b")):
        with column:
            render_session(label, report[key])

    comparison = report["comparison"]
    comparable = all(report[key].get("status") == "ok" for key in ("a", "b"))
    comparable = comparable and comparison.get("status") == "ok"
    delta = number(comparison.get("delta_extension_deg")) if comparable else None
    st.metric("Chênh lệch góc gập vùng duỗi tối đa (A − B)", angle(delta, signed=True))
    if delta is None:
        st.warning("INSUFFICIENT_DATA — Chưa đủ dữ liệu để so sánh hai buổi.")
    elif comparison.get("interpretation"):
        st.info(comparison["interpretation"])
    show_warnings(comparison)
    st.caption(
        "Góc gập 0° tương ứng chân gần thẳng. P10 là phân vị 10% của chuỗi góc hợp lệ đã lọc, "
        "giúp giảm ảnh hưởng của điểm ngoại lệ. Chênh lệch dương nghĩa là buổi B có góc gập "
        "vùng duỗi tối đa nhỏ hơn A; số âm là ngược lại. Chỉ diễn giải trong điều kiện quay tương đồng. "
        "Tỉ lệ khung hình hợp lệ không phải độ chính xác hay phần trăm hồi phục."
    )


if __name__ == "__main__":
    main()
