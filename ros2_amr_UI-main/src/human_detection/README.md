# Gói ROS 2 Human Detection + Gesture Recognition

Node **`gesture_node`** nhận diện 7 cử chỉ RoCoG (Advance, Attention, Rally, MoveForward, Halt,
FollowMe, MoveInReverse) trên ảnh camera trước `/image_front` của gói `camera`.

```
/image_front ─► YOLO tìm người ─► MediaPipe Pose ─► 15 khớp ─► cửa sổ 32 frame @ 10 FPS ─► DD-Net ─► CommandFilter ─► /gesture
                (best.pt)         (pose_landmarker_      (3,2 giây)             (ddnet_gesture.pt)
                    │              lite.task)
                    └─► /gesture/person (khung người)
```

Code pipeline nằm trong `human_detection/gesture/`, copy từ `Human Detection/Human Detection/`.
Công cụ **`video_publisher`** phát video hoặc webcam lên `/image_front`, để test khi không có RealSense.

## Topic

| Topic | Kiểu | Nội dung |
|---|---|---|
| `/gesture` | `std_msgs/String` | Lệnh đã lọc ổn định: `Halt`, `MoveForward`, ... hoặc `Attention` (đứng yên) |
| `/gesture/raw` | `std_msgs/String` (JSON) | Dự đoán thô: `label`, `prob`, `person`, `skeleton`, `reason`, `probs` (7 lớp) |
| `/gesture/person` | `vision_msgs/Detection2DArray` | Khung người đang theo dõi (0 hoặc 1 người), đơn vị pixel; `score` là độ tin cậy YOLO (0 = đang dùng lại khung cũ) |
| `/gesture/debug_image` | `sensor_msgs/Image` | Ảnh có khung người và bảng xác suất |
| `/cmd_vel` | `geometry_msgs/Twist` | **Chỉ khi** `enable_cmd_vel: true` (mặc định TẮT) |

Lệnh chỉ đổi khi cùng một cử chỉ đạt xác suất ≥ `conf_threshold` (0.7) trong `confirm_count` (3) lần liên tiếp; riêng `Halt` được nhận ngay.
Mất người quá `lost_sec`, hoặc không thấy cử chỉ quá `hold_sec`, thì lệnh về `Attention`.
Bảng cử chỉ → vận tốc nằm ở `human_detection/gesture/command_filter.py`. Toàn bộ tham số nằm trong `config/gesture_params.yaml`.

## 1. Cài đặt (môi trường ảo, chạy CPU)

Đã test trên ROS 2 **Jazzy** / Ubuntu 24.04 / Python 3.12. Thư viện Python được cài vào **venv**, không cài vào Python hệ thống,
để không làm hỏng các gói ROS cài từ apt trên robot.

```bash
./setup_venv.sh                 # tạo <thư_mục_dự_án>/gesture_venv, vd ~/projects/human_detect/gesture_venv
                                #   (hỏi sudo một lần nếu thiếu python3-venv, ros-jazzy-vision-msgs)
```

Script tạo venv với `--system-site-packages`, nên venv vẫn thấy rclpy và numpy 1.26 của apt. Script cũng đặt
`PYTHONNOUSERSITE=1` để venv không đọc thư viện trong `~/.local`, rồi cài theo `requirements.txt`
(torch CPU, ultralytics, mediapipe, opencv; phiên bản đã ghim). Phải giữ `numpy<2`.

## 2. Build (trong venv)

```bash
source /opt/ros/jazzy/setup.bash
source <thư_mục_dự_án>/gesture_venv/bin/activate
cd <ros2_ws>
python3 -m colcon build --symlink-install
source install/setup.bash
```

Khi build trong venv, dòng đầu của file chạy node trỏ thẳng vào python của venv, nên node luôn dùng đúng venv.

## 3. Chạy

Mỗi terminal: `source /opt/ros/jazzy/setup.bash && source <thư_mục_dự_án>/gesture_venv/bin/activate && source <ros2_ws>/install/setup.bash`

```bash
# Trên robot
ros2 launch camera camera.launch.py
ros2 launch human_detection gesture.launch.py

# Test trên laptop bằng video (chỉ trong máy này, tránh lẫn camera robot trong LAN)
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
ros2 launch human_detection gesture.launch.py
ros2 run human_detection video_publisher --ros-args -p source:=/duong/dan/video.mp4   # hoặc source:=0 cho webcam
```

## 4. Kiểm tra

```bash
ros2 topic echo /gesture
ros2 topic echo /gesture/raw
ros2 topic hz /gesture/raw                # cần khoảng ≥ 10 Hz
ros2 run rqt_image_view rqt_image_view    # chọn /gesture/debug_image
```

## Ghi chú

- QoS subscriber mặc định là **Best Effort** để khớp với camera (`qos_reliability: sensor_data`).
- Node chỉ xử lý frame mới nhất; frame cũ bị bỏ khi model chạy chậm hơn camera.
- Cần khoảng 3,2 giây hình ảnh cử chỉ mới ra kết quả đầu tiên.



ip: 1