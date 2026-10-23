# Human Detection

Robot AMR nhận ra người trước camera và hiểu lệnh bằng cử chỉ tay. Chạy trên CPU, ROS 2 Jazzy.

## Cách hoạt động

```
Camera ─► YOLOv5n ─► ByteTrack ─► MediaPipe Pose ─► DD-Net ─► lệnh (/gesture)
          tìm người   gán ID      khung xương      phân loại
                                                   cử chỉ
```

| Bước | Mô hình | Kết quả |
|---|---|---|
| Tìm người | YOLOv5n (finetune) | mAP50 0,83 |
| Theo dõi người | ByteTrack | – |
| Khung xương | MediaPipe Pose lite (dùng sẵn) | – |
| Phân loại cử chỉ | DD-Net lite (train từ đầu, 5 lớp) | Accuracy test 95% (model 7 lớp cũ, chưa cập nhật cho 5 lớp) |

Báo cáo đầy đủ: [BAO_CAO_HUAN_LUYEN.md](BAO_CAO_HUAN_LUYEN.md)

## Cấu trúc thư mục

```
human_detect/
│
├── Human Detection/Human Detection/   HUẤN LUYỆN MÔ HÌNH
│   ├── extract_keypoints.py              video -> khung xương
│   ├── train_ddnet.py                    train DD-Net
│   ├── evaluate_ddnet.py                 đánh giá trên tập test
│   ├── finetune_human.py                 finetune YOLO
│   └── checkpoints/scratch_f16/          DD-Net đã train + kết quả
│
├── ros2_amr_UI-main/src/              ROS 2 WORKSPACE
│   ├── camera/                           đọc camera RealSense
│   └── human_detection/                  nhận diện cử chỉ (package chính)
│       ├── human_detection/                 code: gesture_node, trajectory_node, record_node
│       ├── config/                          tham số (yaml)
│       ├── launch/                          file launch
│       └── models/                          best.pt, ddnet_gesture.pt, pose_landmarker_lite.task
│
├── install_ros2_jazzy.sh              cài ROS 2
├── BAO_CAO_HUAN_LUYEN.md              báo cáo huấn luyện
└── NHAT_KY.md                         nhật ký làm việc
```

Dữ liệu train (`Dataset/`, `runs/`, `keypoints/`) không có trong repo.

## Cài đặt và chạy

```bash
./install_ros2_jazzy.sh
./ros2_amr_UI-main/src/human_detection/setup_venv.sh     # tạo venv ~/gesture_venv

source /opt/ros/jazzy/setup.bash
source ~/gesture_venv/bin/activate
cd ros2_amr_UI-main
python3 -m colcon build --symlink-install
source install/setup.bash

ros2 launch human_detection gesture.launch.py             # nhận diện cử chỉ
ros2 launch human_detection people_tracking.launch.py     # thêm theo dõi người + quỹ đạo
```

Cần camera publish `/image_front`. Chưa có camera thì dùng `video_publisher` để phát video.

## Topic chính

| Topic | Nội dung |
|---|---|
| `/gesture` | Lệnh đã lọc (số hiệu) |
| `/gesture/raw` | Xác suất từng cử chỉ (JSON) |
| `/people/tracks` | Mọi người trong khung, có ID |
| `/people/target_id` | ID của người chủ đã khoá |
| `/people/trajectories` | Quỹ đạo dự đoán |

Danh sách đầy đủ: [package README](ros2_amr_UI-main/src/human_detection/README.md).

## Huấn luyện lại

```bash
cd "Human Detection/Human Detection"
python extract_keypoints.py --splits train val test --device 0
python train_ddnet.py --mode scratch --name v2
python evaluate_ddnet.py --ckpt checkpoints/v2/best.pt --split test
```

## Hạn chế

- Chưa thử với người thật trước camera robot.
- `MoveForward` hay bị nhầm với `MoveInReverse`, `Attention` với `Rally`.
- Cử chỉ cần khoảng 3,2 giây để ra kết quả.
- Chưa có phần bám người và tránh người.
