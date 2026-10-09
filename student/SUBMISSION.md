# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

## Thông tin học viên

- Họ tên: Bùi Đức Thành
- MSSV: 2A202602364
- Email: 26ai.thanhbd@vinuni.edu.vn
- Link repo (fork): https://github.com/n4hhh/K4-L2L3-DAY23-BuiDucThanh-2A202602364-SensorFusion
- Commit hash nộp: Xem hash của commit CP6 trên VLearn sau khi hoàn thành bài.

## Tóm tắt kết quả

**Cấu hình chạy:**
- `fusion_mode`: compare
- `frames`: [0, 198], tổng 199 frame
- `seed`: 0
- `segment`: training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord

**Detection:**
- TP: 519
- FP: 16
- FN: 222
- Precision: 0.9700934579439252
- Recall: 0.7004048582995951

**Tracking LiDAR-only:**
- RMSE: 0.15032268781360134 m
- Matches: 502
- Sum squared error: 11.343649056695735 m²
- Ghost track frames: 0
- Missed GT frames: 239
- Mean confirmed tracks: 2.522613065326633

**Tracking LiDAR + Camera:**
- RMSE: 0.1358667883353908 m
- Matches: 502
- Sum squared error: 9.26681165463209 m²
- Ghost track frames: 0
- Missed GT frames: 239
- Mean confirmed tracks: 2.522613065326633

**Nhận xét:**

Chế độ fused giảm RMSE từ 0.15032 m xuống 0.13587 m, tương đương cải thiện khoảng 9.62%. Số matches của cả hai chế độ đều bằng 502, số ghost track frames bằng 0 và missed GT frames bằng 239.

Tracking precision bằng 502/(502+0) = 100%, trong khi coverage bằng 502/519 ≈ 96.72%. Cả hai mode đều đạt ngưỡng RMSE, precision và coverage theo rubric.

Việc RMSE giảm trong khi số matches và ghost/miss không thay đổi cho thấy phép cập nhật camera cải thiện độ chính xác vị trí của những track đã được ghép, thay vì cải thiện độ bao phủ hoặc thay đổi vòng đời track.

Các số liệu được lấy từ `student/artifacts/metrics.json` của lần chạy `--fusion compare --seed 0`, frame 0–198. Kết quả có thể đối chiếu bằng `student/artifacts/grade_run.log`, `metrics_lidar.json`, `metrics_fused.json` và các log riêng từng mode.

Giới hạn: Camera trong lab sử dụng tâm bounding box 2D ground-truth FRONT cộng nhiễu có seed, không chạy camera detector thực tế. Vì vậy kết quả fused không chứng minh hiệu quả của một hệ perception camera độc lập với ground truth.

## Giải thích ngắn (Parts E–H)

**1. Khác biệt đo LiDAR 3D và camera 2D trong EKF (`z`, `R`)?**

LiDAR cung cấp vị trí 3D với vector đo `z = [x, y, z]ᵀ`, có kích thước 3×1. Covariance `R` là ma trận 3×3 mô tả độ bất định của phép đo vị trí theo mét bình phương.

Camera cung cấp tọa độ pixel `z = [u, v]ᵀ`, kích thước 2×1. Covariance `R` là ma trận 2×2, biểu diễn phương sai nhiễu pixel. Trong cấu hình lab, độ lệch chuẩn là 5 pixel theo mỗi trục.

LiDAR sử dụng mô hình đo tuyến tính, còn camera sử dụng mô hình pinhole phi tuyến. EKF dùng Jacobian camera do platform cung cấp để cập nhật trạng thái 6D.

Code: `student/workspace/kalman.py` và `camera_fusion.py`.

**2. Vì sao cần gating Mahalanobis trước khi gán?**

Khoảng cách Euclidean không tính đến độ bất định của track và sensor. Mahalanobis sử dụng innovation `γ = z - h(x)` và innovation covariance `S = HPHᵀ + R` để tính `d² = γᵀS⁻¹γ`.

Gating chi-square loại bỏ những cặp track–measurement có sai khác quá lớn so với mức độ bất định dự kiến. Điều này giúp hạn chế gán nhầm giữa các xe và tránh cập nhật EKF bằng phép đo không phù hợp. Các cặp camera ngoài FOV phải bị loại trước khi tính khoảng cách.

Code: `student/workspace/association.py`.

**3. Pipeline là track-then-fuse hay fuse-then-track?**

Pipeline sử dụng **track-then-fuse**. Detector LiDAR tạo các detection 3D, sau đó tracker thực hiện EKF predict một lần mỗi frame, LiDAR association/update và camera association/update trên cùng danh sách track.

Không gộp dữ liệu thô LiDAR và camera trước detection. Camera bổ sung phép đo để tinh chỉnh state đã được LiDAR khởi tạo.

Bằng chứng: `student/workspace/association.py`, `kalman.py` và hai mode được ghi trong `student/artifacts/grade_run.log`. Kết quả cuối cho thấy cả hai mode có 502 matches, 0 ghost và 239 misses, trong khi RMSE khác nhau.

**4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?**

Calibration sai khiến tọa độ ảnh dự báo `h(x)` bị lệch so với measurement thực tế. Vì vậy innovation `γ = z - h(x)` có thể xuất hiện sai lệch có hệ thống.

Khoảng cách Mahalanobis có thể tăng, khiến nhiều cặp bị chi-square gating loại bỏ. Nếu phép đo sai vẫn qua gating, EKF có thể cập nhật state theo hướng sai và làm RMSE tăng.

Ảnh hưởng phụ thuộc mức độ lệch calibration và covariance `R`, không phải mọi sai lệch đều bị gating phát hiện.

**5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?**

Khi `meas_list` rỗng, không thể suy ra modality từ measurement. Tuy nhiên TrackManager vẫn cần biết lượt hiện tại thuộc LiDAR hay camera.

Ở lượt LiDAR, track không có measurement trong FOV phải được xử lý như miss và có thể bị giảm score hoặc xóa. Ở lượt camera, measurement chỉ dùng để cập nhật EKF; không làm thay đổi score, không tạo track mới và không xóa track.

Vì vậy `associate_and_update` luôn gọi `manager.manage_tracks(...)`, kể cả khi không có measurement.

Code: `student/workspace/association.py` và `platform/fusion_lab/tracking/manager.py`.

**6. Điều kiện xác nhận, giữ confirmed sau miss và xóa track?**

Track mới có score `1/window`, với `window = 6`. Mỗi LiDAR hit cộng `1/6`, tối đa score 1. Mỗi miss trong LiDAR FOV trừ `1/6`.

Track được xác nhận khi score vượt `confirmed_threshold = 0.8`. Khi đã confirmed, một miss không làm track trở lại tentative.

Track bị xóa nếu `P[0,0] > max_P` hoặc `P[1,1] > max_P`, với `max_P = 9`, hoặc khi confirmed có score dưới 0.6, hoặc khi chưa confirmed có score nhỏ hơn hoặc bằng 0.

Camera không tham gia các quyết định vòng đời này.

Code: `student/workspace/track_management.py`.

## Bonus

Không thực hiện bonus.

## Khai báo sử dụng AI

- Công cụ đã dùng: ChatGPT (OpenAI).
- Dùng cho phần nào: Hỗ trợ giải thích công thức EKF, Mahalanobis gating, camera projection, thiết kế và triển khai các hàm Part E–H; hướng dẫn debug môi trường Windows, pytest, cách chạy Waymo và diễn giải kết quả. Hỗ trợ soạn và rà soát báo cáo.
- Cách đã kiểm tra lại: Chạy `pytest student/tests -q` đạt 128 passed; chạy pipeline Waymo với `--fusion compare --seed 0` trên frame 0–198; đối chiếu RMSE, matches, ghost, miss và coverage từ metrics. Kiểm tra artifact và công cụ submission của repo trước khi nộp.

## Checklist nộp

- [x] Part E–H đã implement, 128 test passed.
- [x] Part A–D giữ nguyên.
- [x] Đã chạy compare, seed 0, frame 0–198.
- [x] Đã commit đầy đủ sáu artifacts.
- [x] Đã điền email và rà soát báo cáo, khai báo AI.
- [x] Đã kiểm tra không commit dataset, weights, paths.yaml hoặc API key.
- [x] `python tools/check_submission.py` báo sẵn sàng.
- [x] Đã push và nộp URL repo cùng commit hash trên VLearn.