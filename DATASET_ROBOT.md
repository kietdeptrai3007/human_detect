# Dataset cử chỉ quay bằng camera robot (`real/robot`)

Bộ dữ liệu bổ sung quay bằng `record_node.py`, dùng chung với RoCoG-v2 để train DD-Net 5 lớp.
Video và keypoint **không nằm trong repo** (`.gitignore`); file này mô tả cách chúng được tạo và dùng.

## Tóm tắt

| Mục | Giá trị |
|---|---|
| Số clip | 229 (sau khi lọc annotation và loại 1 clip hỏng, xem "Làm sạch") |
| Người quay | 7 người: S11, S12, S13, S14, S15, S16, S17 |
| Khoảng cách | 2 m (tất cả) |
| Độ dài / FPS | 5 s / 30 FPS, ảnh 640x360 |
| Số lớp | 5: MoveForward, Halt, MoveInReverse, Left, Right |
| Không có | `None` và `FollowMe` (đã bỏ khỏi bộ lớp của model) |

Mỗi clip có nhịp: đứng yên khoảng 0,4 s, làm cử chỉ khoảng 3,5–4 s, rồi hạ tay.
Đo bằng chuyển động cổ tay trên 162 clip tốt: cử chỉ bắt đầu ở 0,4 s (median) và kết thúc ở 4,0–4,4 s.

| Lớp | Nhãn | Số clip |
|---|---|---|
| MoveForward | 0 | 44 |
| Halt | 1 | 63 |
| MoveInReverse | 2 | 39 |
| Left | 3 | 42 |
| Right | 4 | 41 |

## Hai phòng quay

- **S11–S13:** phòng có ghế và bàn, camera đặt sát sàn, người đứng nhỏ và xa trong khung. Đây là dữ liệu "xa".
- **S14–S17:** phòng sáng hơn, người to hơn trong khung.

Giữ cả hai phòng trong train để model quen với cả người gần và người xa.

## Cấu trúc thư mục

```
Human Detection/Dataset/real_ground/
├── real/robot/<Lớp>/S11_2m_Halt_take03.mp4     # <người>_<khoảng cách>_<lớp>_take<số>
└── annotations/
    ├── robot_all.txt                           # 230 dòng: <đường dẫn> <nhãn>
    ├── robot_train.txt  robot_val.txt  robot_test.txt
    └── all_train.txt  all_val.txt  all_test.txt   # RoCoG + robot, dùng để train
```

Dòng annotation: `real/robot/Left/S11_2m_Left_take01.mp4 3`. Số nhãn theo `CLASS_NAMES` trong `rocog_labels.py`
(khác số nhãn RoCoG gốc; code tự nhận ra video `real/robot/` để dùng bảng này).

## Bảng nhãn

```
0 MoveForward   1 Halt   2 MoveInReverse   3 Left   4 Right
```

Video RoCoG thuộc Advance, Attention, Rally, FollowMe bị bỏ qua khi đọc file chia (không train). Model không có lớp `None`;
lúc chạy robot dùng ngưỡng `conf` trong `command_filter.py` để bỏ dự đoán yếu.

## Làm sạch

1. **Annotation lệch file:** `robot_all.txt` ban đầu có 268 dòng, 38 dòng trỏ tới file đã xoá (17 FollowMe, 20 Right, 1 MoveForward).
   Đã lọc còn 230 dòng khớp 230 file. Bản gốc: `robot_all.txt.bak` (trong `~/robot_gesture_data/annotations/`).
2. **Đánh lại số nhãn** theo bảng 5 lớp ở trên (suy ra từ tên thư mục lớp).
3. **Không cắt clip 5 s.** Cửa sổ train dài 3,2 s và `TRIM_SEC = 0.3` đã bỏ 0,3 s đầu và cuối. Tính trên clip tốt,
   cửa sổ 3,2 s trung bình có 91–98% thời gian đang cử động. Muốn sạch hơn thì đổi `TRIM_SEC` (0,4–0,5), không cắt file.
4. **Lỗi chọn người (S11–S13):** `PersonCropper` chọn bbox to nhất, nên cái ghế bị YOLO nhận nhầm là người thắng người đứng xa,
   và MediaPipe vẽ khung xương lên cái ghế. Có 68 clip bị loại (`ok_ratio` < 0,5). Đã sửa `person_crop.py`: chọn theo
   `conf x độ gần giữa khung`. Chạy lại trên 68 clip: `ok_ratio` trung bình 0,23 → 0,99, 67/68 clip dùng được; clip tốt không bị ảnh hưởng.
   Đừng hạ ngưỡng tin cậy pose để "cứu" clip: khung xương khi đó vẫn bám cái ghế.
5. **Loại 1 clip:** `MoveInReverse/S14_2m_MoveInReverse_take03.mp4` (`ok_ratio` 0,027, pose gần như không ra). Đã chuyển sang
   `rejected/` (không xoá) và bỏ khỏi annotation. Sau khi trích keypoint, đây là clip duy nhất có `ok_ratio` dưới 0,5.

## Chia tập (theo người)

| Tập | Người robot | Clip robot |
|---|---|---|
| train | S11–S14 | 118 |
| val | S15 | 36 |
| test | S16, S17 | 75 |

Ghép với RoCoG (`real_ground_*_tsonly.txt`, `real_ground_test.txt`) thành `all_train/val/test.txt`.
Số video mỗi lớp sau khi bỏ lớp RoCoG không dùng:

| Lớp | train | val | test |
|---|---|---|---|
| MoveForward | 46 | 11 | 30 |
| Halt | 55 | 17 | 35 |
| MoveInReverse | 43 | 12 | 26 |
| Left | 20 | 7 | 15 |
| Right | 23 | 7 | 11 |

Left và Right chỉ có từ data robot nên ít ở train; đây là hai lớp cần theo dõi khi đánh giá.

## Cách tái tạo

```bash
cd "Human Detection/Dataset/real_ground"
cp -r ~/robot_gesture_data/real/robot/* real/robot/ && cp ~/robot_gesture_data/annotations/robot_all.txt annotations/
cd annotations
grep -E "/S1[1-4]_" robot_all.txt > robot_train.txt
grep -E "/S15_"     robot_all.txt > robot_val.txt
grep -E "/S1[67]_"  robot_all.txt > robot_test.txt
cat real_ground_train_tsonly.txt robot_train.txt > all_train.txt   # tương tự val, test

cd "Human Detection/Human Detection"
python rocog_labels.py --splits all_train all_val all_test        # kiểm tra không trùng người/video
cp -r keypoints keypoints_v2                                      # giữ keypoint RoCoG cũ, thư mục cũ không bị đè
python extract_keypoints.py --splits all_train all_val all_test --device 0 --out-dir keypoints_v2
python train_ddnet.py --mode scratch --train-splits all_train --val-split all_val \
    --kp-dir keypoints_v2 --name v2_5cls --device cuda
python evaluate_ddnet.py --ckpt checkpoints/v2_5cls/best.pt --split all_test --kp-dir keypoints_v2
```
