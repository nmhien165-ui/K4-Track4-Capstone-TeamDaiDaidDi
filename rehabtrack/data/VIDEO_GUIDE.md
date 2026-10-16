# Hướng dẫn quay video — VinMec RehabTrack

Tài liệu này dành cho người quay và người làm mẫu. Mục tiêu: có hai clip mô phỏng của **cùng một người, cùng một chân, cùng một camera**, để hệ thống so sánh góc gập gối. Đây là demo kỹ thuật, **không phải chẩn đoán y khoa** và **không phải dữ liệu bệnh nhân**.

## 1. Nguyên tắc bắt buộc

- Chỉ người làm mẫu là **thành viên nhóm, đã thành niên, tự nguyện đồng ý**. Không quay bệnh nhân, không quay người ngoài nhóm.
- Chỉ làm động tác **thoải mái, chậm, không ép khớp, không đau**. Dừng ngay nếu đau hoặc khó chịu.
- Không ngã, không giả bệnh nhân. Hai buổi là **mô phỏng**: buổi A = duỗi hạn chế, buổi B = duỗi nhiều hơn. Chỉ gắn nhãn đó khi người mẫu thực sự đã làm đúng như vậy.
- Không được nói hay viết rằng video là "trước mổ / sau mổ" thật, và không nêu kết quả phẫu thuật.
- Không tải video ngẫu nhiên trên mạng rồi gọi là hai buổi ghép cặp.

## 2. Quyền riêng tư và đồng ý

- Xử lý **trên máy cục bộ**; không tự động tải lên đâu cả.
- Không ghi tên thật, số điện thoại hay mã định danh trong manifest. Dùng `participant_01`.
- Không cần quay mặt. Mặc trang phục kín đáo, tương phản với nền.
- Video cá nhân **không đưa lên Git** theo mặc định (`.gitignore` do người phụ trách repo quản lý). Muốn chia sẻ thì dùng kênh riêng đã thống nhất và có sự đồng ý của người mẫu.
- Trong `data/manifest.csv`, cột `consent_confirmed` giữ là `no` cho tới khi người mẫu **xác nhận đồng ý**; chỉ khi đó mới đổi thành `yes`. Công cụ đánh giá sẽ từ chối chạy clip `staged_real` nếu chưa là `yes`.

## 3. Bố trí quay (giống hệt nhau cho mọi clip)

1. Ghế ổn định, không lăn, không lún.
2. Camera nhìn **ngang từ bên hông** (mặt phẳng dọc, side view) vuông góc với hướng duỗi chân. Thấy trọn **hông - gối - cổ chân** của chân được đo.
3. Camera cố định (chân máy hoặc dựa chắc chắn): cùng vị trí, cùng độ cao, cùng hướng, cùng độ phân giải, cùng zoom cho **cả hai clip**.
4. Đủ sáng; nền tương phản với quần áo; một người trong khung hình.
5. Không có vật che hông/gối/cổ chân (trừ clip lỗi cố ý).
6. **Khung hình**: hông, gối và bàn chân phải nằm **trọn trong khung hình suốt clip**, chừa khoảng trống phía dưới bàn chân và phía trên hông. Khớp nằm trong 3% kích thước khung hình tính từ mép bị hệ thống loại (mô hình ngoại suy keypoint ở mép). Cổng dữ liệu: cần ít nhất **10 frame hợp lệ** và **60%** tổng số frame hợp lệ, nếu không sẽ báo `insufficient_data` và không có số đo.
7. Chọn một chân (`left` hoặc `right`, theo giải phẫu của người mẫu) và dùng chân đó cho cả hai clip. Ghi đúng vào cột `side` trong manifest. Lưu ý: chân giải phẫu và phía màn hình có thể khác nhau, đừng đoán.

## 4. Nội dung mỗi clip

- Dài **6-10 giây**, **2-3 lần** duỗi chân **chậm**, trong biên độ thoải mái.
- Giữ chân ở điểm duỗi tối đa khoảng 1 giây mỗi lần, rồi hạ xuống.

## 5. File cần bỏ vào thư mục `data/`

| File | Bắt buộc | Nhãn (`session_label`) | Nội dung |
|---|---|---|---|
| `session_before.mp4` | Có | `A` | Buổi mô phỏng A: duỗi hạn chế |
| `session_after.mp4` | Có | `B` | Buổi mô phỏng B: duỗi nhiều hơn, cùng người/chân/camera |
| `occluded.mp4` | Tùy chọn | `failure_occluded` | Che một phần cổ chân (ca lỗi) |
| `bad_angle.mp4` | Tùy chọn | `failure_bad_angle` | Camera xoay lệch khỏi góc nhìn ngang (ca lỗi) |

Đặt tên **đúng như bảng** (chữ thường, đuôi `.mp4`). Nếu đặt tên khác, sửa `relative_path` trong `data/manifest.csv`. Đường dẫn trong manifest tính từ **thư mục gốc dự án** (thư mục cha của `data/`).

Sau khi quay xong, mở `data/manifest.csv`: xóa chữ `PLACEHOLDER - file not recorded yet` trong cột `notes`, và đổi `consent_confirmed` thành `yes` chỉ khi người mẫu đã đồng ý.

## 6. Hai clip lỗi (ca thất bại có chủ ý)

- **`occluded.mp4`**: lặp lại động tác nhưng che một phần cổ chân. Kỳ vọng: mất keypoint, `valid_ratio` thấp hoặc `insufficient_data`.
- **`bad_angle.mp4`**: xoay camera lệch (oblique) khỏi mặt phẳng dọc. **Lưu ý**: keypoint vẫn có thể có độ tin cậy cao dù góc 2D bị sai. `valid_ratio` cao **không** chứng minh góc camera đúng. Clip này để trình bày giới hạn, không phải để chứng minh hệ thống phát hiện được góc lệch.
- Hai clip này **không bao giờ** được ghép cặp làm buổi B.

## 7. Nếu camera bị xoay hoặc bị che giữa chừng

- Dừng và quay lại clip đó từ đầu; không cắt ghép hai đoạn khác góc.
- Nếu đã đổi vị trí camera giữa hai clip A và B, **quay lại cả hai** để chúng cùng điều kiện.
- Nếu không còn thời gian: ghi `camera_view` là `oblique` hoặc `unknown` trong manifest; công cụ sẽ cảnh báo (không chặn), nhưng so sánh có thể bị lệch và phải được nói rõ khi trình bày.

## 8. Checklist quay trong 10 phút

- [ ] Người mẫu đã đồng ý bằng lời, biết video chỉ xử lý trên máy.
- [ ] Ghế ổn định; camera cố định, nhìn ngang, thấy hông - gối - cổ chân.
- [ ] Ánh sáng đủ; không vật cản; chỉ một người trong khung hình.
- [ ] Khung hình: chừa khoảng trống dưới bàn chân và trên hông; hông, gối, bàn chân không chạm mép khung (cách mép hơn 3%) ở bất kỳ frame nào.
- [ ] Chọn chân (left/right) và ghi lại.
- [ ] Quay `session_before.mp4` (6-10 s, 2-3 lần duỗi chậm, hạn chế).
- [ ] Không chạm camera. Quay `session_after.mp4` (duỗi nhiều hơn, thoải mái).
- [ ] Xem lại hai clip: thấy rõ hông, gối, cổ chân suốt clip.
- [ ] (Tùy chọn) Quay `occluded.mp4` và `bad_angle.mp4`.
- [ ] Bỏ file vào `data/`; sửa `data/manifest.csv` (bỏ PLACEHOLDER, `consent_confirmed=yes`).
- [ ] Chạy: `python evaluate.py --manifest data/manifest.csv --output-dir outputs/evaluation --side right`
