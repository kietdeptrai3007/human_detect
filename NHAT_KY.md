# Nhật ký dự án: Human Detection trên ROS 2

## Mục tiêu

**Nhận diện cử chỉ (gesture) của người trước camera trước để điều khiển robot AMR**, chạy trong ROS 2.

- Có 7 cử chỉ (bộ dữ liệu RoCoG-v2): Advance, Attention, Rally, MoveForward, Halt, FollowMe, MoveInReverse.
- Package `camera` (C++, có sẵn) đọc 2 camera Intel RealSense và publish ảnh lên `/image_front` và `/image_rear`.
- Package `human_detection` (Python, mới viết) subscribe `/image_front`, nhận diện cử chỉ, rồi publish lệnh.
- **Mở rộng (2026-10-19):** theo dõi mọi người bằng ByteTrack (ID ổn định), chỉ nghe lệnh của "người chủ" đã khoá bằng cử chỉ `FollowMe`,
  và dự đoán quỹ đạo từng người (`trajectory_node`) để sau này robot dừng/đổi hướng tránh người. **Chưa làm phần né người.**

```
Camera ─► camera node ─► /image_front ─► gesture_node ───────────────────────────────────────────► /gesture      (lệnh đã lọc)
          (C++)          (bgr8 640x360,  YOLO best.pt → MediaPipe Pose → 15 khớp → 32 frame @10FPS   /gesture/raw  (JSON xác suất)
                          30 FPS)        → DD-Net ddnet_gesture.pt → CommandFilter                   /gesture/person (khung người)
                                                                                                     /gesture/debug_image
                                                                                                     /cmd_vel      (mặc định TẮT)

   gesture_node cũng chạy ByteTrack (cùng một lần YOLO) rồi publish thêm:
        /people/tracks (mọi người, id) ─► trajectory_node ─► /people/trajectories (JSON), /people/markers
        /people/target_id, /people/target_state  (người chủ đã khoá)  ─► trajectory_node gắn is_commander
```

## Cấu trúc thư mục

```
human_detect/
├── Human Detection/                       ← phần train model (copy từ ~/Downloads/Human Detection, 2,8 GB)
│   ├── Human Detection/                   code gesture gốc: train_ddnet.py, gesture_recognizer.py, robot_control.py, ...
│   │   └── checkpoints/scratch_f16/best.pt   ← DD-Net đã train (test acc 95%)
│   ├── weights/pose_landmarker_{lite,full}.task   MediaPipe Pose
│   ├── Dataset/                           person_*, Merged_Human_Dataset, real_ground (video RoCoG)
│   └── runs/detect/runs/detect/robot_person_detect/weights/best.pt   ← YOLO tìm người
├── ros2_amr_UI-main/                      ← ROS 2 WORKSPACE (chạy colcon build ở đây)
│   └── src/
│       ├── camera/                        package C++ có sẵn
│       └── human_detection/               package Python mới viết
│           ├── human_detection/
│           │   ├── gesture_node.py        ← NODE CHÍNH: nhận diện cử chỉ + ByteTrack + khoá người chủ
│           │   ├── trajectory_node.py     ← NODE: dự đoán quỹ đạo từng người (nhận /people/tracks)
│           │   ├── tracking_demo.py       chạy thử tracking + quỹ đạo trên video, KHÔNG cần ROS
│           │   ├── gesture/               pipeline copy từ code gốc (sửa import cho package)
│           │   ├── tracking/              Python thuần, không import ROS: person_tracker (YOLO+ByteTrack), target_selector
│           │   │                          (khoá người chủ), kalman_cv, trajectory_predictor, camera_model, types
│           │   ├── image_utils.py         chuyển sensor_msgs/Image <-> numpy
│           │   └── video_publisher.py     công cụ test: phát video/webcam lên /image_front
│           ├── config/gesture_params.yaml, bytetrack_person.yaml, camera_model.yaml, trajectory_params.yaml
│           ├── launch/gesture.launch.py, people_tracking.launch.py
│           ├── models/best.pt, ddnet_gesture.pt, pose_landmarker_lite.task
│           ├── requirements.txt, setup_venv.sh   ← tạo venv để chạy trên robot
│           └── setup.py, setup.cfg, package.xml, resource/, README.md
├── install_ros2_jazzy.sh                  script cài ROS 2 + thư viện
└── NHAT_KY.md                             file này
```

## Đã làm

### 2026-10-16

1. **Phân tích code ban đầu**
   - `Human Detection` chỉ có model, dataset và kết quả train. Chưa có code inference, chưa có node ROS.
   - Model `best.pt` có 1 class `{0: 'Person'}`, train YOLOv5n với imgsz=640.
     Epoch 55: Precision 0.84, Recall 0.72, mAP50 0.81, mAP50-95 0.47.
   - Package `camera` publish `sensor_msgs/Image` dạng bgr8, 640x360, 30 FPS, QoS **Best Effort** (`SensorDataQoS`).
   - `camera/lib/libunilidar_sdk2.a` là thư viện LiDAR Unitree, không được dùng, có lẽ bị copy nhầm vào.

2. **Viết package `human_detection`**
   - Subscribe `/image_front` với QoS Best Effort cho khớp với camera. Dùng Reliable thì sẽ không nhận được ảnh.
   - Không dùng cv_bridge mà tự chuyển Image ↔ numpy, để tránh xung đột phiên bản numpy.
   - Chạy YOLO trên thread riêng và chỉ xử lý frame mới nhất, để không bị trễ dồn khi model chậm.
   - Publish `vision_msgs/Detection2DArray` lên `/image_front/detections`, và ảnh đã vẽ khung lên `/image_front/detections_image`.
   - Tham số: `conf_threshold=0.5`, `iou_threshold=0.45`, `imgsz=640`, `device=""` (tự chọn), `publish_annotated=true`.
   - Ban đầu node đọc cả `/image_front` và `/image_rear`. Sau đó đổi sang **chỉ `/image_front`**.
     Muốn thêm camera sau thì thêm `"/image_rear"` vào `image_topics` trong file yaml.

3. **Cài môi trường trên laptop** (Ubuntu 24.04, chạy `install_ros2_jazzy.sh`)
   - Đã cài ROS 2 **Jazzy**: `ros-jazzy-ros-base`, `ros-dev-tools`, `vision-msgs`, `rqt-image-view`, `librealsense2`.
   - Đã cài `ultralytics` 8.4.178 và torch bản CPU. numpy giữ ở 1.26.4: **không được lên numpy 2**, vì ROS cài từ apt được build với numpy 1.x.
   - Đã thêm `source /opt/ros/jazzy/setup.bash` vào `~/.bashrc`.
   - Đã kiểm tra: `ros2` chạy được, pub/sub test OK, `import rclpy, vision_msgs, ultralytics` OK.

### 2026-10-16 (chiều): chuyển sang nhận diện cử chỉ

4. **Nhận code gesture mới** từ `~/Downloads/Human Detection` và copy (rsync) vào `Human Detection/`.
   Bản cũ là tập con của bản mới, nên không mất gì; YOLO `best.pt` giống hệt bản cũ.
   - Pipeline: YOLO tìm người → crop + làm mượt bbox → MediaPipe Pose (33 điểm) → 15 khớp chuẩn hoá
     → cửa sổ T=32 frame @ 10 FPS → DD-Net (16 filters) → 7 cử chỉ → CommandFilter.
   - DD-Net `checkpoints/scratch_f16/best.pt`: test accuracy 95%, macro F1 0.95.
5. **Cài thêm `mediapipe` 1.1.1** bằng pip (`--user --break-system-packages "numpy<2"`), numpy vẫn ở 1.26.4.
   Chạy thử `demo_camera.py` trên video Halt: ra **Halt 93%**.
6. **Đưa pipeline vào package ROS**:
   - Copy `config, ddnet_model, person_crop, pose_estimate, skeleton_adapter, gesture_recognizer` vào
     `human_detection/gesture/`, đổi sang import tương đối. `GestureRecognizer` thêm tham số `pose_model_path`,
     thuộc tính `last_crop` và hàm `close()`.
   - `command_filter.py`: tách `CommandFilter` và `GESTURE_TO_MOTION` ra từ `robot_control.py`.
   - `gesture_node.py`: subscribe `/image_front` (Best Effort), dùng thread riêng và chỉ lấy frame mới nhất,
     dùng `header.stamp` làm thời gian. Publish `/gesture`, `/gesture/raw` (JSON), `/gesture/debug_image`,
     và `/cmd_vel` khi bật `enable_cmd_vel` (mặc định TẮT).
   - `video_publisher.py`: phát video/webcam lên `/image_front` (giữ tỉ lệ, thêm viền đen) để test không cần RealSense.
   - Copy model vào `models/`: `ddnet_gesture.pt`, `pose_landmarker_lite.task`.
7. **`colcon build` cả workspace thành công** (cả `camera` lẫn `human_detection`).
8. **Test end-to-end qua ROS** (video tập test, người S00):
   Halt → lệnh **Halt** (100% frame), MoveForward → **MoveForward** (100%),
   Attention → phần lớn Attention, thỉnh thoảng nhầm Rally. Node xử lý được khoảng 10 FPS trên CPU.
9. **Gộp còn 1 node**: xoá `human_detection_node`, vì `gesture_node` đã chạy YOLO bên trong, chạy 2 node sẽ phải chạy YOLO 2 lần.
   - Hai hàm chuyển ảnh chuyển sang `image_utils.py`.
   - `gesture_node` publish thêm `/gesture/person` (`vision_msgs/Detection2DArray`), là khung người đang theo dõi lấy
     từ YOLO bên trong pipeline. `score` là độ tin cậy YOLO (`CropResult` có thêm trường `conf`).
   - Đã build sạch lại và test với video Halt: ra lệnh Halt, `/gesture/person` có bbox đúng.
10. **Chuẩn bị môi trường ảo (venv) để chạy trên robot** (robot dùng ROS 2 **Jazzy**, **không có GPU** → torch CPU).
    - Lý do: hiện thư viện được cài bằng `pip --user --break-system-packages` vào Python hệ thống. Trên robot, cách này có thể
      làm hỏng các gói ROS từ apt (ví dụ khi numpy bị nâng lên 2.x), mà robot còn đang chạy nhiều node khác. Docker thì quá nặng cho một node Python.
    - `requirements.txt`: ghim phiên bản đã test là torch 2.14.1+cpu, torchvision 0.29.1+cpu, ultralytics 8.4.178,
      mediapipe 1.1.1, opencv(-contrib)-python 4.11.0.86, `numpy<2`.
    - `setup_venv.sh [đường_dẫn]` (mặc định `~/gesture_venv`):
      1. cài `python3-venv` và `ros-jazzy-vision-msgs` nếu thiếu (cần sudo);
      2. tạo venv với `--system-site-packages` (để dùng numpy 1.26 của apt và yaml mà ROS cần), thêm `COLCON_IGNORE`;
      3. thêm `PYTHONNOUSERSITE=1` vào `activate` để venv không đọc thư viện trong `~/.local`;
      4. `pip install -r requirements.txt`, rồi kiểm tra import và numpy < 2.
    - `setup.cfg` thêm `[build_scripts] executable = /usr/bin/env python3`. Đã kiểm tra: khi build **không** có `--symlink-install`,
      dòng đầu của file `install/.../gesture_node` là `#!/usr/bin/env python3`, nên node chạy bằng python của venv đang kích hoạt.
      Khi build **có** `--symlink-install`, dòng này vẫn là `#!/usr/bin/python3`; muốn dùng venv thì build bằng
      `python3 -m colcon build` khi đang kích hoạt venv. **Chưa test.**
    - **Đang chờ:** máy thiếu `python3-venv` (cần sudo), nên bạn cần tự chạy `setup_venv.sh`.
11. **Đã tạo venv `~/gesture_venv` trên laptop và test thành công.**
    - `setup_venv.sh` chạy xong: đã cài `python3-venv`. torch, ultralytics, cv2, mediapipe nằm trong venv; numpy 1.26.4 lấy từ apt.
    - Build bằng `python3 -m colcon build --symlink-install` khi đang kích hoạt venv. Dòng đầu file chạy node là `#!/usr/bin/env python3`
      (với setuptools mới trong venv thì `--symlink-install` cũng nhận `[build_scripts]`).
    - Kiểm tra tiến trình node đang chạy: torch và cv2 nạp từ `~/gesture_venv`, **không có file nào từ `~/.local`**, có `PYTHONNOUSERSITE=1`.
      (`exe` hiện là `/usr/bin/python3.12`; bình thường, vì python của venv là symlink tới python hệ thống.)
    - Test video MoveInReverse (S00): lệnh **MoveInReverse** (229 lần dự đoán), nhưng có 79 lần nhầm sang MoveForward, nên lệnh
      nhảy qua lại giữa hai cử chỉ. Đây là lỗi nhầm của model (hai cử chỉ có động tác tay giống nhau, ngược chiều), không phải lỗi venv.
      Clip ngắn chỉ 3–4 giây và phát lặp lại, nên mỗi lần quay về đầu clip cũng làm chuyển động bị đứt đoạn.
12. **Dọn `~/.local` trên laptop**: gỡ 32 gói đã cài bằng `pip --user` trong buổi này (torch, ultralytics, mediapipe, opencv...),
    giải phóng 1,6 GB. Tất cả các gói đều được cài trong ngày 2026-10-16 khoảng 08:43–09:19, không có gói nào có từ trước.
    Đã kiểm tra lại: Python hệ thống quay về dùng `setuptools`, `typing_extensions`, numpy 1.26.4 của apt; ROS (`rclpy`, colcon) vẫn chạy;
    venv vẫn đủ thư viện; `gesture_node` trong venv vẫn nhận đúng Halt.
    Từ giờ **mọi thư viện Python của gesture chỉ nằm trong `~/gesture_venv`**, nên chạy node phải kích hoạt venv trước.
    (pip cache `~/.cache/pip` 451 MB vẫn giữ lại, dùng khi cài lại venv cho nhanh. Có thể xoá bằng `pip cache purge`.)
13. **Chốt cách triển khai lên robot: WORKSPACE RIÊNG** (chưa làm, đang chờ robot rảnh vì có người khác đang SSH vào).
    - Bối cảnh: robot dùng chung, có nhiều chức năng và node của nhóm (SLAM, Path planning...). Mình chỉ phụ trách mảng gesture,
      nên **không được đụng vào workspace hay package của người khác**.
    - Chỉ chép package `human_detection` (12 MB, đã gồm 3 model) vào **`~/gesture_ws/src/`** trên robot. Không chép:
      `Human Detection/` (2,8 GB dữ liệu train), `build/ install/ log/`, venv (venv không chép sang máy khác được, phải tạo lại bằng `setup_venv.sh`),
      `camera/` (robot đã có sẵn).
    - Làm vậy được vì package không phụ thuộc package nào của nhóm khi build (chỉ cần `/opt/ros/jazzy`). Khi chạy,
      node nhận `/image_front` từ camera node của nhóm qua DDS.
    - Lợi ích: build không ảnh hưởng người khác; gỡ chỉ cần `rm -rf ~/gesture_ws ~/gesture_venv`; thư viện Python nằm riêng trong venv.
    - Code nên quản lý bằng **repo git riêng** cho package (không push vào repo của nhóm); trên robot chỉ cần `git pull` rồi build lại.
14. **Thêm `.gitignore` ở thư mục gốc `human_detect/`** (thư mục này chưa phải git repo).
    Bỏ qua: `build/ install/ log/`, cache Python, venv, `Human Detection/{Dataset,runs,weights}`, `keypoints/`, video `*.mp4`, rosbag, IDE.
    Giữ lại: code, config, nhật ký, model trong `src/human_detection/models/` và `checkpoints/scratch_f16/best.pt`.
    Đã thử bằng một git tạm đặt ngoài dự án: còn **63 file / 13 MB** (thay vì hơn 2,8 GB).

### 2026-10-19

15. **Tracking người (ByteTrack) + dự đoán quỹ đạo — giai đoạn A** (tracking và quỹ đạo chạy song song; phần chọn người cho nhánh cử chỉ làm ở mục 16):
    - Thư mục mới `human_detection/tracking/` (Python thuần, không import ROS): `person_tracker.py` (YOLO + ByteTrack của ultralytics),
      `kalman_cv.py`, `trajectory_predictor.py` (Kalman vận tốc không đổi cho mỗi track_id), `camera_model.py` (pixel → mặt đất), `types.py`.
    - Node mới `trajectory_node` (subscribe `/people/tracks`, publish `/people/trajectories` JSON và `/people/markers`),
      `tracking_demo.py` (chạy video không cần ROS, in số ID và sai số dự đoán), launch `people_tracking.launch.py`,
      config `bytetrack_person.yaml`, `camera_model.yaml`, `trajectory_params.yaml`.
    - Code cũ đã sửa: `person_crop.py` (`YoloPersonDetector` nhận `tracker_cfg`, vẫn trả `Box` như cũ, thêm `last_tracks`),
      `gesture_recognizer.py` (`tracker_cfg`, `last_tracks`, `reset()` xoá ID), `gesture/config.py`, `gesture_node.py`
      (publish `/people/tracks` mọi người, `enable_tracking`), `setup.py` (thêm package `tracking`, entry `trajectory_node`),
      `package.xml`, `requirements.txt` (thêm `lapx`).
    - Đã test trên laptop (conda, không có ROS): nhánh cử chỉ có tracker cho kết quả giống hệt không tracker (150/150 frame);
      Kalman bám đúng người đi thẳng giả lập; `camera_model` ground↔pixel khớp 2 chiều.
    - **Chưa test:** `gesture_node` và `trajectory_node` trong ROS (máy này không có ROS); dự đoán với người ĐI THẬT
      (video RoCoG chỉ có người đứng, nên sai số dự đoán chưa tốt hơn mốc "đứng yên"; tham số `accel_std`/`pixel_std` chưa chỉnh);
      `camera_model.yaml` là giá trị giả định, phải đo chiều cao, góc cúi và FOV thật.
    - Lưu ý: ByteTrack kích hoạt người mới sau 1 frame, nên người mới xuất hiện chậm 1 frame so với YOLO thuần. Điểm chân lấy từ tâm đáy bbox
      nên khi người giơ tay, x của điểm chân bị lệch theo tay.

16. **Tích hợp tracking vào nhánh cử chỉ — giai đoạn B (khoá "người chủ")**:
    - File mới `tracking/target_selector.py`: máy trạng thái IDLE → SEARCHING → LOCKED → LOST. Ứng viên là người gần nhất, nhiều người thì
      xoay ứng viên mỗi `candidate_dwell_sec` (5 s); ứng viên làm `FollowMe` thì bị khoá; mất người chủ thì chỉ nối lại với track MỚI (id lớn hơn
      mọi id đã thấy trước lúc mất) gần chỗ cũ, nên người đứng cạnh không bị khoá nhầm.
    - Code cũ đã sửa: `person_crop.py` (`PersonCropper.process(frame, target_fn)`, `_choose(boxes, target_id)`, `CropResult.track_id`),
      `gesture_recognizer.py` (đổi người thì xoá buffer 32 frame, adapter, xác suất; `window_epoch`), `command_filter.py` (thêm `reset()`),
      `gesture_node.py` (gọi selector, chặn lệnh khi chưa khoá trừ `Halt`, publish `/people/target_id` và `/people/target_state`),
      `trajectory_node.py` (gắn `is_commander`), `gesture_params.yaml`.
    - Quyết định mặc định (chưa hỏi nhóm, đều là tham số): khoá bằng `FollowMe`; `Halt` của ứng viên chưa khoá vẫn có hiệu lực (`halt_from_any`);
      người chủ mất thì về `Attention` sau `lost_sec`, đổi người chủ thì bộ lọc lệnh reset về `Attention`.
    - Đã test trên laptop (không có ROS): 5 kịch bản của selector (nối lại ID, người đứng cạnh không bị khoá, ID cũ quay lại, xoay ứng viên);
      video ghép 2 người (Halt trái + FollowMe phải, lặp): Halt đi qua khi chưa khoá, sau khi xoay ứng viên thì khoá đúng người FollowMe và giữ
      khoá; đổi track_id giữa chừng thì buffer về 1 frame.
    - **Chưa test:** trong ROS (`gesture_node` nay có thêm publisher/tham số); người thật đi qua lại trước camera; khi chỉ có ứng viên xoay thì người
      chủ phải chờ tới `candidate_dwell_sec` x (số người - 1) mới tới lượt, có thể thấy lệnh `FollowMe` bị bỏ lỡ nếu làm cử chỉ ngắn hơn cửa sổ 3,2 s.

17. **Quyết định thiết kế và môi trường (2026-10-19)**:
    - **Không finetune ByteTrack**: nó là thuật toán ghép detection giữa các frame (Kalman + IoU), không có trọng số. Phần có trọng số là YOLO `best.pt`,
      đã finetune cho người. Chỉ finetune thêm YOLO nếu thực tế detector trượt người (xa, ngược sáng, bị che). Chỉnh bằng `config/bytetrack_person.yaml`.
    - **Tracking đặt trong `gesture_node`**, không tách node riêng, vì YOLO chỉ nên chạy một lần mỗi frame (robot không có GPU). Tắt được bằng
      `enable_tracking: false` (về hành vi cũ) hoặc `enable_target_lock: false` (có tracking, không khoá người chủ).
    - `PersonCropper.reset()` KHÔNG xoá ID tracker (nó tự gọi khi mất người 5 frame, xoá ID lúc đó sẽ làm người quay lại bị đổi ID).
      Chỉ `GestureRecognizer.reset()` xoá ID.
    - `requirements.txt`: thêm `lapx` (ByteTrack cần; thiếu thì ultralytics tự pip install lúc chạy) và `pyyaml`; `matplotlib`, `pyrealsense2` để dạng comment
      (chỉ cho phần train). Conda env `human_detect` (Python 3.11): cần thêm `pip install matplotlib`.
    - `install_ros2_jazzy.sh` chưa liệt kê `ros-jazzy-visualization-msgs` và `python3-yaml` (trajectory_node và camera_model cần); cài thêm bằng apt.
    - Máy `trangto` (`~/Trang/human_detect`, Ubuntu 24.04) **chưa cài ROS** (`/opt/ros` không có), nên toàn bộ phần tracking mới chỉ test được trong conda,
      chưa chạy trong ROS. Các mục "đã test" ở trên (colcon build, chạy trong venv) là của máy `~/kiet/human_detect`.

### 2026-10-21

18. **Cài môi trường train trên máy `trangto`** (conda env `human_detect`, Python 3.12, đã có ROS, GPU NVIDIA, chọn CUDA 12.1):
    ```bash
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # nếu không có bản phù hợp: torch==2.5.1 torchvision==0.20.1
    pip install ultralytics lapx mediapipe matplotlib
    pip install "numpy<2"                                   # chạy lại cuối cùng, vì ultralytics/mediapipe đôi khi kéo numpy 2.x
    pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
    pip install opencv-python==4.11.0.86 opencv-contrib-python==4.11.0.86   # env đang có opencv 5.0.0, headless + full trùng nhau
    ```
    - Không cài từ `requirements.txt` trên máy train, vì file đó ghim torch **CPU** cho robot (sẽ mất GPU).
    - `pyrealsense2` chỉ cần cho `camera_source.py`/`demo_camera.py` (đọc RealSense trực tiếp). `record_node` nhận ảnh qua topic `/image_front` nên không cần.
    - Kiểm tra: `python -c "import torch, cv2, numpy; print(torch.__version__, torch.cuda.is_available(), numpy.__version__, cv2.__version__)"`
      (mong muốn: `True`, numpy `1.26.x`).

19. **`record_node.py` (thu dataset cử chỉ từ camera robot)**: file mới, nằm ở `src/human_detection/human_detection/`, đã đăng ký trong `setup.py`.
    - Chạy: `ros2 run human_detection record_node --ros-args -p subject:=S11 -p out_dir:=/đường/dẫn`. Cần camera node publish `/image_front` và cần màn hình
      (`$DISPLAY`) hoặc `ssh -X`; cửa sổ OpenCV hiện ảnh camera, trạng thái, đếm ngược / thời gian quay.
    - Kết quả: `<out_dir>/real/robot/<Lớp>/S11_5m_Left_take01.mp4` và `<out_dir>/annotations/robot_all.txt` (số nhãn theo `CLASS_NAMES` trong `rocog_labels.py`).
    - Tham số: `countdown` (3 s), `duration` (5 s), `distances`, `manual` (false), `image_topic`, `out_dir`, `subject`.
    - **Thêm nút bấm thủ công** (cùng ngày):
      - `m`: đổi chế độ TỰ ĐỘNG (đếm ngược rồi tự quay `duration` giây) và THỦ CÔNG (chọn nhãn, bấm `SPACE` mới quay, `SPACE` lần nữa để dừng).
      - `SPACE` khi đang đếm ngược = TẠM DUNG (người chưa vào kịp); bấm lại để đếm ngược lại từ đầu.
      - `SPACE` khi đang quay chỉ có tác dụng ở chế độ thủ công; chế độ tự động muốn bỏ clip thì bấm `x`. Clip thủ công ngắn hơn
        `MIN_CLIP_SECS` = 3,5 s thì bỏ, không lưu (cửa sổ train ~3,2 s nên clip ngắn vô dụng). Quên dừng thì tự dừng sau 30 s.
    - **Chưa test với camera thật**, mới kiểm tra cú pháp.

20. **Quy trình thêm dữ liệu mới vào dataset rồi train lại** (chưa chạy thử; đường dẫn và tên file chia suy ra từ code):
    1. Quay bằng `record_node` (nhiều người, nhiều khoảng cách; đổi người bằng `p` để chia tập theo người).
    2. Chép vào `Human Detection/Dataset/real_ground/real/robot/` và `annotations/robot_all.txt`, rồi chia file `robot_train/val/test.txt` **theo người** (grep theo `S1x`).
       Gộp với RoCoG bằng `cat real_ground_train_tsonly.txt robot_train.txt > all_train.txt` (tương tự val, test).
    3. Trong `Human Detection/Human Detection/`: `python extract_keypoints.py --splits all_train all_val all_test --device 0`
       (video đã có `.npz` thì bỏ qua; xem `keypoints/extract_report.csv`).
    4. `python train_ddnet.py --mode scratch --train-splits all_train --val-split all_val --name v2`, rồi
       `python evaluate_ddnet.py --ckpt checkpoints/v2/best.pt --split all_test`.
    - 7 lớp (có Left/Right/None) nên model cũ 5 lớp phải train lại. **(Đã đổi ngày 2026-10-23: còn 5 lớp, không có None/FollowMe, xem mục 21.)**

### 2026-10-23

21. **Làm sạch dataset robot quay thêm** (`~/robot_gesture_data`, mô tả đầy đủ trong [DATASET_ROBOT.md](DATASET_ROBOT.md)).
    - 230 clip, 7 người (S11–S17), 2 m, 5 s, 30 FPS. `robot_all.txt` có 268 dòng nhưng 38 dòng trỏ tới file đã xoá
      (17 FollowMe, 20 Right, 1 MoveForward) → lọc còn 230 dòng khớp file (bản gốc: `robot_all.txt.bak`).
    - **Bỏ lớp `None` và `FollowMe`** khỏi bộ lớp của model. Còn 5 lớp: MoveForward 0, Halt 1, MoveInReverse 2, Left 3, Right 4
      (sửa `rocog_labels.py`: bỏ `None`/`MERGED_TO_NONE`; video RoCoG Advance/Attention/Rally/FollowMe bị bỏ qua khi đọc file chia;
      sửa `record_node.py`: 5 phím `1 2 3 l r`). Đã đánh lại số nhãn trong `robot_all.txt` theo tên thư mục.
      Video RoCoG còn dùng được: train 69, val 18, test 42 (đếm tay từ file annotation).
      **Chưa sửa phía robot:** `lock_gesture: "FollowMe"` (`gesture_params.yaml`, `target_selector.py`) và `GESTURE_IDS` trong
      `command_filter.py` (`"None"` ở đó là lệnh đứng yên, không phải lớp model). Model mới không bao giờ ra FollowMe nên khoá người chủ sẽ không chạy.
    - **Không cắt clip 5 s.** Đo bằng chuyển động cổ tay (162 clip tốt): cử chỉ bắt đầu ~0,4 s, kết thúc ~4,0–4,4 s. Cửa sổ 3,2 s (sau `TRIM_SEC`)
      có 91–98% thời gian đang cử động. Muốn sạch hơn thì tăng `TRIM_SEC` lên 0,4–0,5 thay vì cắt file.
    - **Lỗi chọn người:** trích keypoint, 68 clip (S11–S13 và 1 clip S14) có `ok_ratio` < 0,5. Nguyên nhân không phải pose kém:
      phòng S11–S13 camera sát sàn, người nhỏ và xa, YOLO nhận cái ghế là người, `PersonCropper._choose` chọn bbox **to nhất**
      nên chọn ghế, MediaPipe vẽ khung xương lên ghế. Hạ ngưỡng tin cậy pose chỉ làm khung xương bám ghế nhiều hơn (không dùng được).
    - **Sửa `person_crop.py`** (`_choose`): khi chưa có bbox trước, chọn theo `conf x độ gần giữa khung` thay vì diện tích.
      Chạy lại 68 clip hỏng: `ok_ratio` 0,23 → 0,99, 67/68 dùng được; 14 clip tốt kiểm tra lại không đổi (1,00). Xem khung hình: khung xương bám đúng người.
      Bản copy trong ROS (`gesture/person_crop.py`) **chưa sửa** (robot dùng ByteTrack chọn người chủ nên ít bị).
      Giữ cả 230 clip, kể cả clip "xa", để model học được cả người nhỏ.
    - **Chia theo người:** train S11–S14, val S15, test S16–S17; ghép RoCoG thành `all_train/val/test.txt`.
      Số video mỗi lớp (train/val/test): MoveForward 46/11/30, Halt 55/17/35, MoveInReverse 43/12/26, Left 20/7/15, Right 23/7/11.
      Left và Right ít ở train (chỉ có từ data robot), cần theo dõi khi đánh giá.
    - **Loại clip hỏng:** sau khi trích keypoint (người dùng chạy), chỉ còn `MoveInReverse/S14_2m_MoveInReverse_take03.mp4` có `ok_ratio` < 0,5 (0,027).
      Đã chuyển video sang `rejected/` (không xoá; cả `~/robot_gesture_data/rejected/` và `Dataset/real_ground/rejected/`), bỏ khỏi `robot_all/robot_train/all_train.txt`
      và chuyển `.npz` sang `keypoints_v2/rejected/`. Còn 229 clip robot (MoveInReverse 39); các lớp khác không đổi.
    - **Keypoint lưu ở `keypoints_v2/`** (chép keypoint RoCoG cũ sang, thư mục `keypoints/` cũ không bị đè; đã thêm vào `.gitignore`).
      Trích keypoint do người dùng tự chạy: `python extract_keypoints.py --splits all_train all_val all_test --device 0 --out-dir keypoints_v2`.
      (Mình có chạy thử và dừng sau 28/230 clip; `.npz` đã trích còn nguyên, lần sau tự bỏ qua.)
    - **Train (người dùng tự chạy):**
      `python train_ddnet.py --mode scratch --train-splits all_train --val-split all_val --kp-dir keypoints_v2 --name v2_5cls --device cuda`
      rồi `python evaluate_ddnet.py --ckpt checkpoints/v2_5cls/best.pt --split all_test --kp-dir keypoints_v2`.
      Không bật `--use-mirror` (lật ngang lẫn Left với Right). Model cũ (`scratch_f16`, 7 lớp) không dùng được với bảng 5 lớp.

22. **Sửa phía robot cho model 5 lớp (ai cũng điều khiển được).** Nhận diện không cần sửa code vì `gesture_node` lấy `class_names` từ file `.pt`
    và `command_filter.py` tra theo tên lớp. Lỗi cần sửa: `lock_gesture: "FollowMe"` không còn trong danh sách lớp nên `gesture_node` sẽ báo lỗi và dừng
    ngay khi khởi động (`gesture_node.py`, kiểm tra `lock_gesture not in self.names`).
    - `gesture_params.yaml`: `enable_target_lock: false` (không khoá người chủ, ai cũng điều khiển được); `lock_gesture: "Right"` (chỉ dùng khi bật khoá).
    - Đổi giá trị mặc định `lock_gesture` từ `FollowMe` sang `Right` trong `gesture_node.py` và `tracking/target_selector.py` (tránh lỗi khi ai đó bật lại khoá).
      Chọn `Right`/`Left` vì chưa gắn chuyển động nên khoá nhầm cũng không làm robot chạy; không dùng `Halt`, `MoveForward`, `MoveInReverse`.
    - Cập nhật chú thích trong `gesture_params.yaml`, docstring `gesture_node.py`, `README.md` của package. `GESTURE_IDS` giữ nguyên (bên nhận `/gesture` dùng bảng số này).
    - **Giới hạn khi tắt khoá:** mỗi lúc chỉ nhận diện 1 người (chọn theo IoU với khung trước; chưa có thì người to nhất), nên người đứng trước camera
      làm cử chỉ vẫn có thể bị người khác chen vào. Bản `gesture/person_crop.py` trong ROS vẫn chọn bbox to nhất (lỗi ghế thành người có thể lặp lại).
    - Chưa test trên robot. Cần chép `best.pt` thành `models/ddnet_gesture.pt` và chỉnh `conf_threshold`/`confirm_count` vì không còn lớp `None`.

23. **Đổi output `/gesture` và giao diện debug theo 5 lớp mới.**
    - **Bảng số hiệu `/gesture` (`GESTURE_IDS` trong `command_filter.py`) đã ĐỔI** theo thứ tự lớp của model:
      MoveForward 0, Halt 1, MoveInReverse 2, Left 3, Right 4, None 5 (không có lệnh). Trước đây là 0 MoveForward, 1 Halt, 2 FollowMe, 3 MoveInReverse,
      4 Left, 5 Right, 6 None. **Phần Navigation của nhóm đọc `/gesture` phải đổi theo bảng mới** (`/gesture/name` vẫn là tên chữ, không đổi cách dùng).
    - Bỏ `FollowMe` và trường `follow` khỏi `GESTURE_TO_MOTION`, bỏ `IGNORED_GESTURES` (Advance/Attention/Rally) khỏi `CommandFilter`.
    - **`draw_panel` (ảnh `/gesture/debug_image`) rút gọn:** chỉ còn một dòng ở góc ảnh `<cử chỉ> <xx>%` (độ tin cậy của dự đoán); chưa có dự đoán thì hiện lý do
      (`collecting frames...`, `no person`, ...). Bỏ thanh xác suất từng lớp và dòng `CMD`. Test cục bộ bằng ảnh giả (không chạy trong ROS).
      `CommandFilter` test: 3 lần `Left` 0,9 → `Left`, `Halt` đi qua ngay, mất người quá `lost_sec` → `None`.

24. **Rà soát toàn bộ code sau khi đổi sang 5 lớp** (đọc lại mọi file `.py`/`.yaml`/`.md` có nhắc tới tên lớp). Kết quả:
    - **Đã sửa:** `Human Detection/Human Detection/robot_control.py` (bản chạy tay trên máy train): bảng cử chỉ còn 5 lớp + `None`, trạng thái nghỉ đổi từ
      `Attention` sang `None`, bỏ `follow`; trước đó model 5 lớp ra `Left`/`Right` thì bảng không có, và trạng thái nghỉ `Attention` không còn tồn tại.
    - **Đã sửa chú thích "7 lớp"** ở `demo_camera.py`, `train_ddnet.py`, `ddnet_model.py` (hai bản), `gesture_node.py`, `camera_source.py`.
    - **Đã kiểm tra, không cần sửa:** `train_ddnet.py`/`evaluate_ddnet.py`/`gesture_dataset.py` (lấy `CLASS_NAMES`/`class_names` động; `mirror_label_map` vẫn đúng với Left/Right),
      `extract_keypoints.py`, `gesture_recognizer.py` (hai bản, lấy tên lớp từ checkpoint), `demo_camera.py` (vẽ theo `class_names`), `trajectory_node.py`
      (`/people/target_id` mặc định -1 khi tắt khoá), hai file launch, `test_inference.py` (chỉ test YOLO).
    - `rocog_labels.py` chạy thật bằng env `human_detect`: `all_train` 187, `all_val` 54, `all_test` 117 video, đủ 5 lớp mỗi tập (khớp mục 21).
    - `BAO_CAO_HUAN_LUYEN.md` và bảng "Accuracy test 95%" trong `README.md` là của model 7 lớp cũ; sẽ cập nhật sau khi train xong `v2_5cls`.

## Trạng thái hiện tại

- [x] ROS 2 Jazzy cài xong; thư viện Python nằm trong venv `~/gesture_venv`
- [x] `colcon build` thành công (build trong venv)
- [x] `gesture_node` chạy được qua ROS **trong venv** với video test và nhận đúng cử chỉ
- [x] Đã viết `requirements.txt` và `setup_venv.sh`, đã test trên laptop
- [x] Đã gỡ các gói cũ cài kiểu `pip --user` trong `~/.local` (laptop gọn, chỉ còn venv)
- [x] Đã chốt cách triển khai: workspace riêng `~/gesture_ws` trên robot (mục 13)
- [ ] Chép package lên robot, chạy `setup_venv.sh` và build (chờ robot rảnh)
- [ ] Thống nhất với nhóm: `/cmd_vel`, tên topic, mức dùng CPU (xem Ghi chú)
- [ ] MoveForward và MoveInReverse đôi khi nhầm lẫn nhau, cần thử với người thật
- [ ] Chưa test với người thật đứng trước camera robot
- [ ] `/cmd_vel` giữ TẮT. Robot nhận lệnh qua `/gesture`, do phần Navigation của nhóm xử lý
- [ ] FollowMe chưa có bộ bám người (hiện chỉ đứng yên). Đã có ID và vị trí người chủ (`/people/target_id`), chưa có phần điều khiển bám theo
- [x] Tracking ByteTrack + khoá người chủ + dự đoán quỹ đạo: đã viết, đã test logic trong conda (mục 15, 16)
- [ ] Chạy `colcon build` và thử `people_tracking.launch.py` trong ROS (gesture_node đã sửa nhiều, trajectory_node mới)
- [ ] Thử với người đi thật (video RoCoG chỉ có người đứng): chỉnh `accel_std`, `pixel_std`, `track_buffer`, `candidate_dwell_sec`
- [ ] Đo camera thật (chiều cao, góc cúi, FOV) và điền `config/camera_model.yaml` trước khi bật `use_ground: true`
- [ ] Chưa làm phần né người (dừng/đổi hướng); `trajectory_node` chỉ dự đoán, không publish `/cmd_vel`
- [x] Cài thư viện train trên máy `trangto` (mục 18) và viết `record_node` có nút tạm dừng / thủ công (mục 19)
- [x] Làm sạch dataset robot (230 clip), bỏ lớp None/FollowMe, sửa `person_crop.py`, chia theo người (mục 21)
- [x] Trích keypoint vào `keypoints_v2/`; loại clip `S14 MoveInReverse take03` (ok=0,027) sang `rejected/` (mục 21)
- [ ] Train `v2_5cls` và đánh giá trên `all_test` (mục 21); chép `best.pt` thành `ddnet_gesture.pt` trong `models/`
- [x] Phía robot: tắt khoá người chủ, `lock_gesture` mặc định `Right` (mục 22)
- [ ] Thử ngưỡng `conf_threshold`/`confirm_count` trên robot vì không còn lớp None
- [ ] Sửa bản copy `person_crop.py` trong package ROS cho giống bản train (nếu muốn)
- [ ] Test `record_node` với camera thật

## Việc tiếp theo

0. **Triển khai lên robot bằng workspace riêng** (khi robot rảnh; báo trước cho người quản lý robot vì có bước dùng sudo):
   ```bash
   # Laptop → robot: chỉ chép package (thay user@IP_robot cho đúng)
   ssh user@IP_robot "mkdir -p ~/gesture_ws/src"
   rsync -av --exclude='__pycache__' \
     ~/kiet/human_detect/ros2_amr_UI-main/src/human_detection/ \
     user@IP_robot:~/gesture_ws/src/human_detection/

   # Trên robot
   ~/gesture_ws/src/human_detection/setup_venv.sh      # sudo 1 lần; chỉ cài thêm python3-venv / vision-msgs nếu thiếu
   source /opt/ros/jazzy/setup.bash
   source ~/gesture_venv/bin/activate
   cd ~/gesture_ws && python3 -m colcon build --symlink-install
   head -1 install/human_detection/lib/human_detection/gesture_node     # phải là: #!/usr/bin/env python3
   source install/setup.bash
   ros2 launch human_detection gesture.launch.py       # camera node của nhóm đã chạy sẵn
   ```
   Mỗi lần chạy: source ROS → kích hoạt venv → `source ~/gesture_ws/install/setup.bash` → `ros2 launch`.
   **Không** build trong workspace của nhóm, **không** sửa package `camera`.
   Trên laptop vẫn build như cũ trong `~/kiet/human_detect/ros2_amr_UI-main`.

1. **Test với camera robot thật** qua mạng LAN. Robot đang publish `/image_front` trên domain 0, nên chỉ cần **không** đặt
   `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`:
   ```bash
   source ~/gesture_venv/bin/activate
   cd ~/kiet/human_detect/ros2_amr_UI-main && source install/setup.bash
   ros2 launch human_detection gesture.launch.py
   ros2 topic echo /gesture
   ros2 run rqt_image_view rqt_image_view      # chọn /gesture/debug_image
   ```
   Đứng trước robot, cách 2–10 m, thấy cả thân người, rồi làm từng cử chỉ khoảng 3 giây.
2. **Kiểm tra tốc độ và CPU trên robot**: `ros2 topic hz /gesture/raw` (cần khoảng ≥10 Hz), và `htop` để xem node có làm SLAM/Nav
   của nhóm chậm đi không. Robot không có GPU; nếu nặng quá thì tắt `publish_debug_image` hoặc giảm `imgsz` của YOLO.
3. **Thống nhất với nhóm cách dùng `/gesture`**: phần Navigation của nhóm subscribe `/gesture` và quyết định hành vi
   (ví dụ `Halt` → dừng). **Không** tự bật `enable_cmd_vel`, trừ khi nhóm đồng ý (có thể ghép bằng `twist_mux` với mức ưu tiên).
   Bảng cử chỉ → vận tốc trong `command_filter.py` chỉ dùng khi test riêng.
4. Làm FollowMe (bám người theo bbox, có thể dùng thêm depth của RealSense).
5. **Kiểm tra tracking trong ROS và với người thật** (xem mục 15, 16, 17):
   ```bash
   ros2 launch human_detection people_tracking.launch.py
   ros2 topic echo /people/target_state      # SEARCHING -> LOCKED sau khi làm FollowMe
   ros2 topic echo /people/trajectories
   ros2 topic hz /people/tracks              # tốc độ xử lý thật, dùng để chỉnh track_buffer
   ```
   Thử: một người, hai người (người gần không phải người chủ), người đi chéo nhau (có đổi ID không), người chủ ra khỏi khung rồi quay lại.
   Chạy `python -m human_detection.tracking_demo --video ... --stride 3` (từ `src/human_detection`) để mô phỏng CPU xử lý chậm.
6. Thống nhất với nhóm: có publish thêm depth của RealSense từ package `camera` không (để ước lượng khoảng cách chính xác hơn), có `/odom` để bù
   chuyển động robot không, và ai quyết định dừng/đổi hướng khi người sắp vào đường đi.

## Ghi chú và lỗi đã gặp

- `ros2: command not found` → terminal mở trước khi cài xong. Chạy `source ~/.bashrc` hoặc mở terminal mới.
- `BrokenPipeError` khi chạy `ros2 pkg list | head` → vô hại, do `head` đóng pipe sớm.
- `ros2 doctor` bị treo lâu → bỏ qua, kiểm tra bằng pub/sub thay thế.
- Package Python **không có thư mục `src/`**. Code nằm trong `human_detection/human_detection/`, đúng chuẩn ament_python. Không đổi tên thư mục này.
- `build/`, `install/`, `log/` do `colcon build` tự sinh ra và đã nằm trong `.gitignore`.
- Laptop có GPU RTX 3050 Ti nhưng chưa cài driver NVIDIA, nên hiện YOLO chạy bằng CPU.
- Cần ROS 2 Humble trở lên, vì `vision_msgs` ≥ 4 dùng `bbox.center.position.x`.
- **Trong mạng LAN có máy khác (robot) đang publish `/image_front`** (ảnh văn phòng) trên domain 0. Khi test bằng
  `video_publisher`, ảnh hai nguồn bị trộn lẫn, khiến khung xương nhận được xen kẽ True/False và node báo "quá ít frame dùng được".
  Muốn test riêng trên laptop: `export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` ở **mọi** terminal (và chạy `ros2 daemon stop` một lần).
- Video RoCoG có kích thước vuông 1148×1148. `video_publisher` giữ tỉ lệ và thêm viền đen; nếu ép thẳng về 640×360, người sẽ bị méo.
- Cử chỉ cần khoảng 3,2 giây (32 frame @ 10 FPS) mới ra kết quả đầu tiên. Đây là thiết kế của model, không phải lỗi.
- Attention và Rally đều là cử chỉ vẫy tay nên đôi khi bị nhầm với nhau.
- ByteTrack kích hoạt người mới sau 1 frame, nên người mới xuất hiện chậm 1 frame so với YOLO thuần. `new_track_thresh` phải >= `YOLO_CONF` (0.4),
  nếu cao hơn thì người có conf 0.4–0.5 không có ID và bị mất khỏi nhánh cử chỉ.
- Điểm chân lấy từ tâm đáy bbox nên khi người giơ tay thì toạ độ ngang bị lệch theo tay. Khoảng cách từ ảnh màu kém chính xác, nhất là khi chân bị cắt
  khỏi ảnh (người đứng gần, ảnh 640x360).
- Chưa khoá người chủ thì cử chỉ chỉ chạy cho một ứng viên (người gần nhất, xoay mỗi `candidate_dwell_sec`) và chỉ `Halt` đi qua. Người chủ phải chờ
  tới lượt, nên cử chỉ `FollowMe` ngắn hơn cửa sổ 3,2 s có thể bị bỏ lỡ khi có nhiều người.
- Người chủ mất khỏi khung: chỉ nối lại với track MỚI (id lớn hơn mọi id đã thấy trước lúc mất) gần chỗ cũ; nối lại thì bộ lọc lệnh reset về `Attention`
  và người đó phải làm lại cử chỉ.
- Lỗi `PoseLandmarker.__del__ ... NoneType` khi tắt `demo_camera.py` là vô hại. Trong node đã gọi `close()` để tránh lỗi này.
- **Venv + ROS 2**: phải tạo venv với `--system-site-packages`, và node phải chạy bằng python của venv (xem mục 10). Venv thường sẽ không thấy được thư viện của ROS.
  Nếu venv nằm trong workspace thì cần có file `COLCON_IGNORE` (script đã tự tạo).
- Venv tạo bằng `--system-site-packages` vẫn đọc thư viện trong `~/.local`, và `~/.local` còn được ưu tiên trước. Vì vậy script đặt `PYTHONNOUSERSITE=1`.
- **Robot dùng chung, nhiều node của nhóm**:
  - `/cmd_vel` do Navigation/Path planning của nhóm publish. Nếu gesture_node cũng publish `/cmd_vel`, hai bên sẽ **giành quyền điều khiển**
    robot, vì vậy `enable_cmd_vel` phải giữ `false`.
  - Hỏi nhóm xem `/gesture`, `/gesture/raw`, `/gesture/person`, `/gesture/debug_image` có trùng topic nào không.
  - YOLO + MediaPipe + DD-Net chạy CPU khá nặng, có thể làm chậm SLAM; cần đo khi chạy chung.
  - Laptop và robot ở cùng domain 0 nên tự thấy nhau (DDS multicast). Nếu đang test trên laptop mà không muốn ảnh hưởng robot,
    đặt `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` hoặc `ROS_DOMAIN_ID` khác.
