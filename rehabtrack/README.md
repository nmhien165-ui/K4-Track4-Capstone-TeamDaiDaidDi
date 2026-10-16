# VinMec RehabTrack — So sánh độ duỗi gối từ hai video

Proof of concept cho hackathon Computer Vision (Track 4, gắn với VinMec). Hệ thống đo **góc gập gối theo thời gian** từ hai video mô phỏng của cùng một bài tập, rồi so sánh một chỉ số được định nghĩa thống nhất.

> **DEMO MÔ PHỎNG — không phải tư vấn hay chẩn đoán y khoa.** Không dùng dữ liệu bệnh nhân. Không tính và không hiển thị "% hồi phục".

## 1. Bài toán và người dùng

Chuyên viên phục hồi chức năng cần ghi nhận và đối chiếu khả năng duỗi gối của cùng một người ở hai lần đánh giá. Chỉ xem hai video bằng mắt khó cho ra một số đo thống nhất. MVP này đo góc gập gối từ video và so sánh hai buổi bằng một chỉ số cố định. Người dùng giả định là chuyên viên cần số đo hỗ trợ đánh giá; đây không phải AI chẩn đoán.

## 2. Phạm vi MVP

- Một bài tập duy nhất: **tập duỗi gối khi ngồi (seated knee extension)**, nhìn ngang một bên.
- Hai video: buổi **A** (mô phỏng duỗi hạn chế) và buổi **B** (mô phỏng duỗi nhiều hơn), cùng người, cùng chân, cùng camera. Đây là **hai buổi mô phỏng**, không phải trước/sau phẫu thuật thật.
- Stack do kế hoạch của nhóm chốt sẵn: **Ultralytics YOLO11n-pose (pretrained, đóng băng) + OpenCV + numpy**. Không huấn luyện, không fine-tune, không viết code pose riêng.

## 3. Giả định về góc camera

Camera đặt cố định, nhìn **ngang từ bên hông** (mặt phẳng dọc), thấy trọn hông - gối - cổ chân của chân được đo; một người trong khung hình. Góc đo trong ảnh 2D chỉ đúng khi camera vuông góc với hướng duỗi chân. Camera lệch làm góc sai mà **độ tin cậy keypoint không báo được điều này**.

## 4. Không phải chẩn đoán

Số đo là ước lượng từ video 2D, chưa kiểm chứng với thước đo góc (goniometer). Không dùng để ra quyết định y khoa.

## 5. Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`yolo11n-pose.pt` nằm ở `models/` (hoặc Ultralytics tự tải lần đầu chạy: hãy chạy thử trước khi demo để tránh phát sinh lúc trình bày).

## 6. Cách quay clip

Xem `data/VIDEO_GUIDE.md`: giao thức quay, đồng ý của người mẫu, quyền riêng tư, tên file, checklist 10 phút. Tóm tắt: ghế ổn định, camera cố định nhìn ngang, 6-10 giây, 2-3 lần duỗi chậm, cùng chân cho cả hai clip. Video cá nhân không đưa lên Git.

## 7. Manifest

`data/manifest.csv` mô tả từng clip với đúng header:

`clip_id,relative_path,participant_id,session_label,exercise,side,camera_view,source_type,consent_confirmed,paired_group,notes`

- `session_label`: `A`, `B`, `failure_occluded`, `failure_bad_angle`.
- `source_type`: `staged_real`, `synthetic_test`, `public_external`. `camera_view`: `side`, `oblique`, `unknown`. `consent_confirmed`: `yes`, `no`, `not_applicable`.
- `relative_path` tính từ **thư mục gốc dự án** (thư mục cha của `data/` nếu manifest nằm trong `data/`; ngược lại là thư mục chứa manifest).
- Cột `side` là nguồn chính cho từng clip; nếu để trống thì dùng `--side`. Hai clip A/B của một cặp bắt buộc cùng `side`.
- File manifest hiện tại **chỉ có dòng PLACEHOLDER** (clip chưa quay, `consent_confirmed=no`). Công cụ sẽ báo lỗi dễ đọc, liệt kê từng file thiếu, cho đến khi bạn quay clip, bỏ PLACEHOLDER và xác nhận đồng ý.
- Ghép cặp: cùng `participant_id`, `paired_group`, `side`, bài tập; mỗi cặp có đúng một A và một B. Clip lỗi **không bao giờ** là B.

## 8. Chạy giao diện

```bash
streamlit run app.py
```

`app.py` do đồng đội phụ trách giao diện (UI) cung cấp và sẽ được gộp sau; chưa được kiểm tra ở đây (**NOT TESTED**).

## 9. Lệnh đánh giá (`evaluate.py`)

```bash
python evaluate.py --manifest data/manifest.csv --output-dir outputs/evaluation --side right
python evaluate.py --smoke-test        # kiểm tra import và API, không mở video, không tải weights
python evaluate.py --synthetic-test    # kiểm tra toán hình học bằng keypoint sinh sẵn, KHÔNG phải suy luận mô hình
python evaluate.py --pair data/session_before.mp4 data/session_after.mp4 --side right --consent-confirmed
python -m pytest tests/test_evaluation.py -q
```

- `--pair` chỉ chạy khi có `--consent-confirmed` (không bao giờ tự giả định đồng ý); camera được coi là `unknown` nên có cảnh báo.
- Mã thoát: `0` xong và có ít nhất một clip `ok`; `1` kiểm thử synthetic sai; `2` manifest không hợp lệ hoặc không có gì để đánh giá; `3` lỗi tích hợp chặn (pose.py/metrics.py không import được).
- Một clip lỗi không làm hỏng cả lần chạy: clip đó ghi `status=error`, không có số bịa.
- `--help` và kiểm tra manifest chạy được khi chưa cài ultralytics.

## 10. Kết quả đầu ra

Thư mục `outputs/evaluation/` (sinh tự động, không phải dữ liệu nguồn):

- `clip_metrics.csv`: clip, trạng thái, số frame tổng/hợp lệ, `valid_ratio`, góc gập tại vùng duỗi tối đa (P10), biên độ quan sát (P90 - P10), cảnh báo, lỗi, và `edge_rejected_frames` (số frame có khớp bị loại vì nằm ở mép khung hình; để trống nếu bản ghi pose không mang khóa `edge_rejected`).
- `pair_comparison.csv`: cặp, hai clip A/B, hai góc, `delta_extension_deg`, trạng thái, cảnh báo.
- `report.json`: thời điểm, tên model, `evaluation_type` (`real_video` chỉ khi extractor thật đã xử lý ít nhất một clip `staged_real`/`public_external`, còn lại là `synthetic_only`), phương pháp, số đo từng clip, so sánh cặp, lỗi, giới hạn.

## 11. Cách đọc delta

`delta_extension_deg = góc gập tốt nhất của A - góc gập tốt nhất của B` (góc gập tốt nhất = P10 của chuỗi góc hợp lệ; chân thẳng gần 0 độ).

- **Dương**: ở buổi B, gối trông duỗi thẳng hơn A (góc gập nhỏ hơn).
- **Âm**: ngược lại. Ví dụ A = 40 độ, B = 15 độ cho delta = +25 độ (kiểm thử synthetic).
- Đây chỉ là chênh lệch giữa **hai clip mô phỏng**, không phải kết quả lâm sàng. **Không quy đổi thành "% hồi phục"** vì không có ground truth và không có định nghĩa y khoa cho phép làm vậy.

## 12. Kết quả đo

### Cặp A/B tự quay (đo thật, 2026-10-16)

Chạy chính thức: `python evaluate.py --manifest data/manifest.csv --output-dir outputs/evaluation --side right` (YOLO11n-pose pretrained, CPU, 5 fps). Hai clip quay ngồi ghế, camera nhìn ngang, cùng người, cùng chân gần camera, cùng ghế và vị trí camera, 1280x720. Số liệu lấy từ `outputs/evaluation/` (thư mục sinh ra, không đưa vào Git).

| Clip | status | valid_ratio | Góc gập tốt nhất (P10, độ) | Biên độ quan sát (độ) | Ghi chú |
|---|---|---|---|---|---|
| session_before (A, duỗi hạn chế) | ok | 1.00 (87/87) | 36.4 | 42.4 | 17,3 s |
| session_after (B, duỗi gần thẳng) | ok | 0.98 (94/96) | 17.5 | 78.1 | 19,1 s |

**Delta A - B = +18.9 độ** (số dương: gối trông thẳng hơn ở buổi B). Đây là chênh lệch giữa hai clip mô phỏng, không phải kết quả lâm sàng, không quy đổi thành "% hồi phục".

Lưu ý khi đọc:
- Một cặp clip, một người, không có goniometer: sai số tuyệt đối chưa biết.
- Clip B được quay **trước** clip A theo thời gian; nhãn A/B theo động tác thực hiện (A: chủ ý dừng sớm, B: duỗi gần thẳng), không theo giờ quay.
- `side=right` là track chân phải của mô hình; trong góc nhìn này nó bám theo chân gần camera. Hai track trái/phải cho cùng delta 18,9 độ vì chân xa bị che. Bên giải phẫu thực tế chưa được kiểm chứng độc lập.
- Cuối clip B có bàn tay che ống kính (khoảng 1 giây cuối); ảnh hưởng chủ yếu đến biên độ (P90), ít ảnh hưởng P10.
- Tham khảo thô về độ lặp lại: một cặp clip cùng mức duỗi quay trước đó (khung hình khác) cho P10 chân phải 17,2 và 17,8 độ (lệch 0,6 độ). Chạy bằng script tạm, không qua `evaluate.py`, file không nằm trong repo; chỉ là gợi ý, không phải độ chính xác.

### Ca lỗi thật (lần quay đầu, giữ tại `data/attempt1_walking_*.mp4`, chưa ghi vào manifest)

Hai clip đầu tiên quay sai: người **đứng/đi bộ** thay vì ngồi, camera chéo phía trước, cổ chân bị cắt ở mép dưới khung hình, và người đi ra khỏi khung hình giữa clip.

- **Lần chạy đầu (code cũ: ngưỡng 50% và 5 frame, chưa loại khớp ở mép khung hình):** pipeline cho chân phải valid_ratio 0.53 (24/45) và 0.52 (17/33), P10 18,1 và 17,4 độ, delta 0,7 độ. Code cũ **để lọt cả hai clip** với `status: ok`.
- **Chạy lại ngày 2026-10-16 với code mới** (loại khớp nằm trong 3% kích thước khung hình tính từ mép; ngưỡng 60% và 10 frame), chân phải: `attempt1_walking_before` có 0/45 frame hợp lệ, `insufficient_data` (34 frame có khớp ở mép khung hình bị loại); `attempt1_walking_after` có 1/33 frame hợp lệ, `insufficient_data` (24 frame bị loại). `compare_sessions` trả `insufficient_data`, **không có delta**.

Cách đọc đúng: rào chắn mới đã chặn đúng ca lỗi cụ thể này (cổ chân sát mép khung hình). Nhưng `valid_ratio` cộng quy tắc mép **vẫn không phát hiện được** camera lệch khỏi mặt phẳng dọc, hoặc sai bài tập mà cơ thể nằm trọn trong khung hình; phải nhìn lại frame. Con số cũ (delta 0,7 độ) **không dùng được** và không được trích dẫn.

Cặp A/B thật ở trên **không đổi** khi chạy với code mới: A hợp lệ 87/87, P10 36,4; B hợp lệ 94/96, P10 17,5; delta +18,9 độ (đã chạy lại bằng `evaluate.py` chính thức).

### Smoke test bằng clip công khai (KHÔNG phải cặp A/B, không phải dữ liệu bệnh nhân)

Ngày 2026-10-16, chạy pipeline thật (`extract_video_pose` -> `analyze_session`, YOLO11n-pose, CPU) trên **một** clip công khai Pexels 32709558 (giấy phép Pexels; 9,7 giây, 30 fps, 2160x3840, 16,6 MB), lưu local tại `data/external/pexels_32709558/` (không đưa vào Git). Manifest: `data/manifest_smoke_public.csv`, lệnh:

```bash
python evaluate.py --manifest data/manifest_smoke_public.csv --output-dir outputs/smoke_public --side right
```

| Clip | status | valid_ratio | Góc gập tốt nhất (P10, độ) | Biên độ quan sát (độ) |
|---|---|---|---|---|
| Pexels 32709558, side=right | ok | 1.00 (49/49 frame) | 21.2 | 72.1 |
| (cùng clip, side=left, chạy phụ) | ok | 1.00 (49/49 frame) | 18.7 | 73.4 |

Cách đọc đúng: pipeline chạy được trên video người thật (nhiều lần duỗi/gập, confidence gối khoảng 0.88-0.99). **Đây không phải số đo gối hợp lệ**: clip quay trên máy tập gym, camera nằm chéo phía trước (không phải mặt phẳng dọc), thấy cả hai chân. Chân phải/trái cho P10 lệch nhau 2,5 độ, nhưng đó không phải bằng chứng về độ lặp lại hay độ chính xác. Clip này cũng cho thấy `valid_ratio` = 1.00 và confidence cao vẫn không báo được góc camera sai, đúng như giới hạn đã nêu bên dưới. Phép đo thật cần cặp A/B tự quay.

Kiểm tra tích hợp end-to-end (`pose` -> `metrics` -> `evaluate`) trên cặp A/B tự quay: **đã chạy thật ngày 2026-10-16** (kết quả ở phần đầu mục này). `app.py` / `streamlit run`: **NOT TESTED** (UI do thành viên khác làm, sẽ ghép sau).

## 13. Dataset công khai (tùy chọn, không phụ thuộc)

Các dataset tham khảo và tính chất đã kiểm chứng được ghi trong `data/README.md`; README này không nhắc lại chi tiết. Không dataset nào là cặp trước/sau phẫu thuật có kết quả y khoa.

## 14. Giới hạn

- **Chiếu 2D**: góc đo trong mặt phẳng ảnh, không phải 3D.
- **Góc camera**: confidence cao không chứng minh camera đúng góc; `valid_ratio` cao ở clip `bad_angle` không có nghĩa là hệ thống phát hiện được góc lệch.
- **Che khuất**: che cổ chân làm mất keypoint, giảm frame hợp lệ hoặc cho `insufficient_data`.
- **Mép khung hình**: khớp nằm trong 3% kích thước khung hình tính từ mép bị loại vì mô hình ngoại suy keypoint ở mép, không quan sát thật. Hông, gối và bàn chân phải nằm trọn trong khung hình, có chừa khoảng trống phía dưới bàn chân và phía trên hông.
- **Cổng dữ liệu**: cần ít nhất 10 frame hợp lệ và 60% tổng số frame hợp lệ (ngưỡng quy ước MVP, không phải ngưỡng y khoa); thiếu thì trả `insufficient_data` và không đưa số đo. Quy tắc mép và `valid_ratio` vẫn không phát hiện được camera lệch hoặc sai bài tập khi cơ thể nằm trọn trong khung hình.
- **Sai số pose estimation**: YOLO có thể sai khớp; P10/P90 giảm tác động ngoại lệ nhưng không loại bỏ hoàn toàn.
- **Không ground truth**: chưa có goniometer nên chưa biết sai số tuyệt đối, và không công bố độ chính xác theo độ.
- **Không có nhóm bệnh nhân**: chỉ tình nguyện viên trong nhóm, hai buổi mô phỏng.

## 15. Hướng phát triển

Theo dõi nhiều buổi theo thời gian; đối chiếu với goniometer để báo cáo sai số; kết hợp IMU; kiểm chứng đa góc / đa camera / 3D; kiểm tra tổng quát hóa trên người và camera mới trước khi đề xuất triển khai.

## 16. Kịch bản demo 90 giây

1. (0:00-0:15) Nêu vấn đề: so sánh hai lần tập duỗi gối cần một số đo thống nhất; đây là demo mô phỏng, không chẩn đoán, không dữ liệu bệnh nhân.
2. (0:15-0:35) Cho xem hai clip A và B (cùng người, cùng chân, camera cố định nhìn ngang) và nói rõ chúng là hai buổi mô phỏng.
3. (0:35-1:00) Chạy `python evaluate.py --manifest data/manifest.csv ...` hoặc giao diện; chỉ ra `valid_ratio`, góc P10 của A và B, và delta kèm dấu (dương = B duỗi thẳng hơn).
4. (1:00-1:20) Ca lỗi: clip `occluded` hoặc `bad_angle`; giải thích rằng `valid_ratio` không chứng minh phát hiện được camera lệch.
5. (1:20-1:30) Giới hạn: 2D, chưa có goniometer, không có % hồi phục; hướng phát triển.

**Phương án dự phòng**: dùng các clip quay thật đã chuẩn bị sẵn và kết quả đã chạy trước. Nếu YOLO không chạy được, chỉ trình bày `python evaluate.py --synthetic-test` và **nói rõ đây là keypoint synthetic, không phải suy luận trên video**; không ngụ ý rằng đó là kết quả mô hình.
