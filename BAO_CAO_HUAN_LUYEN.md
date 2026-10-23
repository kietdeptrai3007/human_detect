# BÁO CÁO CÔNG VIỆC: HUẤN LUYỆN MÔ HÌNH NHẬN DIỆN NGƯỜI VÀ CỬ CHỈ CHO ROBOT AMR

> **Lưu ý (2026-10-23):** báo cáo này mô tả model DD-Net 7 lớp RoCoG cũ (`scratch_f16`). Model hiện tại có 5 lớp (MoveForward, Halt, MoveInReverse, Left, Right), train bằng RoCoG + dữ liệu robot (xem `DATASET_ROBOT.md`, `NHAT_KY.md` mục 21). Số liệu dưới đây chưa được cập nhật.


**Dự án:** Human Detection trên ROS 2 (điều khiển robot AMR bằng cử chỉ)
**Thời gian:** 16/10/2026 – 21/10/2026
**Người thực hiện:** Trang To

---

## 1. Tóm tắt

Mục tiêu của dự án là để robot AMR nhận ra người đứng trước camera và hiểu lệnh bằng cử chỉ tay, chạy thời gian thực trên CPU (robot không có GPU) trong ROS 2 Jazzy.

Hệ thống ghép **bốn thành phần**. Hai thành phần có huấn luyện riêng, hai thành phần dùng sẵn hoặc không có trọng số:

| Thành phần | Nhiệm vụ | Huấn luyện? | Kết quả chính |
|---|---|---|---|
| **YOLOv5n** (finetune) | Tìm người trong ảnh | Có – finetune 1 lớp `Person` | mAP50 = **0.826**, mAP50-95 = **0.514** |
| **MediaPipe Pose** (lite) | Lấy khung xương 33 điểm | Không – dùng model có sẵn | – |
| **DD-Net** (lite, 16 filters) | Phân loại cử chỉ từ chuỗi khung xương | Có – train từ đầu | Độ chính xác test = **95.0%**, Macro-F1 = **0.950** |
| **ByteTrack** | Gán ID ổn định cho từng người | Không – thuật toán, không có trọng số | – |

**Kết quả đạt được:** pipeline hoàn chỉnh chạy được end-to-end trong ROS 2 trên CPU (~10 FPS). Test với video, các cử chỉ `Halt` và `MoveForward` được nhận đúng 100% số frame. **Chưa kiểm thử với người thật trước camera robot.**

---

## 2. Kiến trúc tổng thể

```
Ảnh camera ─► YOLOv5n ─► ByteTrack ─► Cắt ảnh người ─► MediaPipe Pose ─► 15 khớp chuẩn hoá
 (/image_front)  (tìm người)  (gán ID)   (crop + làm mượt)  (33 điểm)

   ─► cửa sổ 32 frame @10 FPS (3,2 giây) ─► DD-Net ─► 7 cử chỉ ─► CommandFilter ─► /gesture
```

Bảy cử chỉ (bộ dữ liệu RoCoG-v2): `Advance`, `Attention`, `Rally`, `MoveForward`, `Halt`, `FollowMe`, `MoveInReverse`.

Lý do chọn hướng "phát hiện người → khung xương → phân loại chuỗi" thay vì phân loại trực tiếp từ video:
- Khung xương nhẹ, chạy được trên CPU.
- DD-Net chỉ có phần phân loại cuối, nên model rất nhỏ (file trọng số ~0.6 MB).
- Đặc trưng khoảng cách giữa các khớp không phụ thuộc vị trí hay kích thước người trong ảnh.

---

## 3. Mô hình 1: YOLOv5n – phát hiện người

### 3.1. Vai trò
Tìm người trong từng frame, cho khung bao (bbox) để cắt ảnh đưa vào bước lấy khung xương và để ByteTrack theo dõi.

### 3.2. Dữ liệu
Gộp 3 bộ dữ liệu người (định dạng Roboflow/YOLO, 1 lớp `Person`) thành `Merged_Human_Dataset`:

| Tập | Số ảnh |
|---|---|
| Train | 6,288 |
| Validation | 2,043 |
| Test | 1,044 |
| **Tổng** | **9,375** |

Nguồn: `Human Dataset` (2,188 ảnh), `person_1` (1,484 ảnh), `person_2` (5,703 ảnh).

### 3.3. Cấu hình huấn luyện
| Tham số | Giá trị |
|---|---|
| Model khởi tạo | `yolov5n.pt` (pretrained COCO) |
| Số epoch tối đa / patience | 60 / 20 (early stopping) |
| Kích thước ảnh | 640 |
| Batch | 16 |
| Optimizer / lr khởi đầu | auto / 0.01 |
| Số epoch thực tế chạy | 55 (thời gian ~32 phút) |
| Công cụ | Ultralytics (`finetune_human.py`) |

### 3.4. Kết quả (tập validation)

| Chỉ số | Epoch tốt nhất (35) | Epoch cuối (55) |
|---|---|---|
| Precision | **0.856** | 0.840 |
| Recall | 0.721 | 0.723 |
| mAP50 | **0.826** | 0.808 |
| mAP50-95 | **0.514** | 0.467 |

Nhận xét:
- Model phát hiện người khá tốt (mAP50 > 0.8, precision > 0.85).
- Recall ~0.72: còn bỏ sót khoảng 28% người, thường ở trường hợp người xa, bị che hoặc ngược sáng. Đây là điểm cần theo dõi khi thử trên robot thật.
- Từ epoch 35, mAP50-95 giảm nhẹ trong khi loss train vẫn giảm (0.87 so với 1.07), cho thấy dấu hiệu bắt đầu quá khớp. Ultralytics tự lưu `best.pt` theo epoch tốt nhất, nên model đang dùng là bản epoch 35.
- Model đang dùng trong ROS: `models/best.pt` (5.3 MB), ngưỡng tin cậy `conf = 0.4`.

> Lưu ý: file `NHAT_KY.md` ghi số liệu của epoch cuối (epoch 55: mAP50-95 0.47). Số liệu của `best.pt` là epoch 35 như bảng trên.

---

## 4. Mô hình 2: MediaPipe Pose – lấy khung xương

- Dùng bản `pose_landmarker_lite.task` có sẵn của Google, **không huấn luyện thêm**.
- Chạy trên ảnh đã cắt quanh người, trả về 33 điểm; chọn **15 khớp** liên quan đến thân và tay, chuẩn hoá theo độ rộng vai.
- Cùng bản model (`lite`) phải dùng cho cả lúc trích dữ liệu train và lúc chạy trên robot; tên model được ghi vào checkpoint để phát hiện lệch.

---

## 5. Mô hình 3: DD-Net – nhận diện cử chỉ

### 5.1. Vai trò
Nhận chuỗi 32 khung xương (3,2 giây ở 10 FPS) và phân loại thành 1 trong 7 cử chỉ.

### 5.2. Kiến trúc
DD-Net (Double-feature Double-motion Network, Yang et al. 2019), viết lại bằng PyTorch từ bản Keras gốc. Bản "lite" với 16 filters:
- 3 nhánh đặc trưng: khoảng cách giữa mọi cặp khớp (JCD), chuyển động chậm, chuyển động nhanh; mỗi nhánh dùng Conv1D.
- Ghép nhánh → 3 khối Conv1D → Global Max Pool → 2 lớp fully connected (128) → 7 lớp đầu ra.
- Đặc trưng JCD và chuyển động được tính **bên trong model**, nên lúc train, lúc chạy robot và khi xuất ONNX không thể lệch cách tính.
- Trọng số chỉ ~0.6 MB, phù hợp chạy CPU.

### 5.3. Dữ liệu (RoCoG-v2, `real_ground`)
Chia theo **người** để tránh rò rỉ dữ liệu giữa các tập:

| Tập | Số video | Người quay |
|---|---|---|
| Train | 162 | S01, S03, S05, S07, S08 |
| Validation | 42 | S02, S10 |
| Test | 100 | S00, S04, S04b, S06, S09 |

Mỗi video được chuyển thành khung xương (YOLO → MediaPipe) rồi lưu `.npz`; resample về 10 FPS, cắt cửa sổ 32 frame (bỏ 0,3 giây đầu/cuối vì nhãn nhiễu).

### 5.4. Cấu hình huấn luyện (`scratch_f16`)
| Tham số | Giá trị |
|---|---|
| Chế độ | Train từ đầu (scratch), không pretrain |
| Filters / Dropout / FC dropout | 16 / 0.1 / 0.5 |
| Batch / lr / weight decay | 64 / 1e-3 / 1e-4 |
| Epoch tối đa / patience | 300 / 40 (dừng sớm ở epoch 95, tốt nhất tại epoch 55) |
| Mất mát | Cross-entropy, label smoothing 0.1, class weight |
| Cửa sổ lấy mẫu | 8 cửa sổ ngẫu nhiên mỗi clip mỗi epoch |
| Tăng cường dữ liệu | Xoay ±10°, co giãn 0.9–1.1, đổi tốc độ 0.8–1.2, nhiễu Gauss, che ngẫu nhiên 1 khớp; **tắt** lật gương (vì cử chỉ gắn với tay cụ thể) |
| Thiết bị / seed | CUDA / 42 |

Mỗi epoch chỉ ~0.3 giây, nên toàn bộ huấn luyện xong trong vài chục giây.

### 5.5. Kết quả

**Tập test (100 video, 4 người chưa từng thấy khi train):**

| Chỉ số | Giá trị |
|---|---|
| Độ chính xác theo video | **95.0%** |
| Macro-F1 | **0.950** |
| Độ chính xác theo cửa sổ (531 cửa sổ) | 88.9% |
| Loss | 0.373 |

**Kết quả theo từng cử chỉ (test):**

| Cử chỉ | Precision | Recall | F1 | Số video |
|---|---|---|---|---|
| Advance | 1.00 | 1.00 | 1.00 | 14 |
| Attention | 0.82 | 1.00 | 0.90 | 14 |
| Rally | 1.00 | 0.80 | 0.89 | 15 |
| MoveForward | 0.93 | 1.00 | 0.97 | 14 |
| Halt | 1.00 | 0.93 | 0.96 | 14 |
| FollowMe | 0.94 | 1.00 | 0.97 | 15 |
| MoveInReverse | 1.00 | 0.93 | 0.96 | 14 |

**Độ chính xác theo từng người (test):** S00 100%, S04 94%, S06 92%, S09 95% – kết quả đồng đều, không phụ thuộc nhiều vào một người.

**Các nhầm lẫn còn lại (5/100 video):**
- `Rally` → `Attention`: 3 video. Cả hai đều là động tác vẫy tay nên giống nhau.
- `Halt` → `FollowMe`: 1 video.
- `MoveInReverse` → `MoveForward`: 1 video.

**Tập validation (42 video):** accuracy 92.9%, Macro-F1 0.924. Lỗi tập trung ở `MoveInReverse` (3/6 video bị nhầm thành `MoveForward`) – hai cử chỉ có động tác tay giống nhau nhưng ngược chiều.

---

## 6. Thành phần ByteTrack và khoá "người chủ"

- ByteTrack là thuật toán ghép detection giữa các frame (Kalman + IoU), **không có trọng số nên không huấn luyện**. Phần có trọng số là YOLO đã finetune ở mục 3.
- Mỗi người có ID ổn định; máy trạng thái `IDLE → SEARCHING → LOCKED → LOST` khoá một "người chủ" khi người đó làm cử chỉ `FollowMe`, chỉ nghe lệnh của người này.
- Dự đoán quỹ đạo từng người bằng bộ lọc Kalman vận tốc không đổi (`trajectory_node`), chuẩn bị cho phần tránh người sau này.

---

## 7. Kiểm thử tích hợp

| Nội dung kiểm thử | Kết quả |
|---|---|
| `colcon build` cả workspace (venv, ROS 2 Jazzy) | Thành công |
| Video `Halt` qua ROS | Ra lệnh **Halt** đúng 100% frame |
| Video `MoveForward` qua ROS | Ra lệnh **MoveForward** đúng 100% frame |
| Video `Attention` qua ROS | Phần lớn đúng, thỉnh thoảng nhầm `Rally` |
| Video `MoveInReverse` qua ROS | Ra **MoveInReverse** (229 lần), nhầm sang MoveForward 79 lần |
| Tốc độ xử lý trên CPU | ~10 FPS |
| Tracking: nhánh cử chỉ có/không có tracker | Kết quả giống hệt (150/150 frame) |
| Khoá người chủ trên video ghép 2 người | Khoá đúng người làm `FollowMe` và giữ khoá |

Cử chỉ cần ~3,2 giây để ra kết quả đầu tiên: đây là đặc điểm thiết kế (cửa sổ 32 frame), không phải lỗi.

---

## 8. Hạn chế và rủi ro

1. **Chưa thử với người thật trước camera robot.** Dữ liệu RoCoG-v2 quay trong điều kiện khác (nền, ánh sáng, góc camera); độ chính xác thực tế có thể thấp hơn 95%.
2. **Bộ test nhỏ** (100 video, 4 người): sai số thống kê còn lớn, mỗi video sai đổi ~1% độ chính xác.
3. **`MoveForward` ↔ `MoveInReverse` và `Attention` ↔ `Rally`** là hai cặp dễ nhầm; lệnh có thể nhảy qua lại giữa hai cử chỉ.
4. **Recall YOLO ~0.72:** có thể bỏ sót người ở xa hoặc bị che.
5. Dự đoán quỹ đạo chưa được kiểm thử với người đi thật (video RoCoG chỉ có người đứng); thông số camera trong `camera_model.yaml` mới là giá trị giả định.
6. Phần tracking mới chỉ được test logic trong conda, chưa chạy trong ROS trên máy huấn luyện hiện tại.

---

## 9. Công việc đang làm và kế hoạch

**Đã chuẩn bị:** `record_node` để quay dữ liệu cử chỉ ngay từ camera robot (có chế độ tự động và thủ công); cập nhật bảng lớp sang **7 lớp mới** (thêm `Left`, `Right`, `None`; gộp `Advance`/`Attention`/`Rally` thành `None`). Model 7 lớp mới phải huấn luyện lại, model hiện tại là 7 lớp RoCoG cũ.

**Kế hoạch:**
1. Kiểm thử `record_node` với camera thật; thu dataset mới từ nhiều người, nhiều khoảng cách (2–10 m).
2. Trích khung xương, chia tập theo người, huấn luyện lại DD-Net 7 lớp mới và đánh giá bằng `evaluate_ddnet.py`.
3. Triển khai lên robot trong workspace riêng (`~/gesture_ws`), đo CPU và tốc độ khi chạy chung với SLAM/Navigation.
4. Kiểm thử với người thật (một người, nhiều người, đi qua lại) rồi chỉnh tham số tracking.
5. Nếu YOLO bỏ sót người trên robot thật thì thu thêm ảnh từ camera robot để finetune bổ sung.
6. Làm phần bám người (`FollowMe`) và tránh người dựa trên quỹ đạo đã dự đoán.

---

## 10. Kết luận

Đã hoàn thành chuỗi mô hình nhận diện cử chỉ chạy được trên CPU: **YOLOv5n** tìm người (mAP50 0.83), **MediaPipe Pose** lấy khung xương, **DD-Net lite** phân loại 7 cử chỉ (accuracy test 95%, Macro-F1 0.95) và **ByteTrack** theo dõi người. Toàn bộ đã đóng gói thành package ROS 2 và test thành công trên video. Bước tiếp theo quan trọng nhất là thu dữ liệu và kiểm thử với camera và người thật trên robot, vì đây là điều kiện quyết định độ chính xác thực tế.

**Các tệp liên quan**
- YOLO: `Human Detection/runs/detect/runs/detect/robot_person_detect/` (`results.csv`, `args.yaml`)
- DD-Net: `Human Detection/Human Detection/checkpoints/scratch_f16/` (`args.json`, `log.csv`, `results_test.json`, `results_val.json`, `confusion_test.png`)
- Mã huấn luyện: `finetune_human.py`, `train_ddnet.py`, `evaluate_ddnet.py`, `ddnet_model.py`
- Package ROS: `ros2_amr_UI-main/src/human_detection/`
- Nhật ký chi tiết: `NHAT_KY.md`
