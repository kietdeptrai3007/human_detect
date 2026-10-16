# Nhật ký dự án: Human Detection trên ROS 2

## Mục tiêu

**Nhận diện cử chỉ (gesture) của người trước camera trước để điều khiển robot AMR**, chạy trong ROS 2.

- Có 7 cử chỉ (bộ dữ liệu RoCoG-v2): Advance, Attention, Rally, MoveForward, Halt, FollowMe, MoveInReverse.
- Package `camera` (C++, có sẵn) đọc 2 camera Intel RealSense và publish ảnh lên `/image_front` và `/image_rear`.
- Package `human_detection` (Python, mới viết) subscribe `/image_front`, nhận diện cử chỉ, rồi publish lệnh.

```
Camera ─► camera node ─► /image_front ─► gesture_node ───────────────────────────────────────────► /gesture      (lệnh đã lọc)
          (C++)          (bgr8 640x360,  YOLO best.pt → MediaPipe Pose → 15 khớp → 32 frame @10FPS   /gesture/raw  (JSON xác suất)
                          30 FPS)        → DD-Net ddnet_gesture.pt → CommandFilter                   /gesture/person (khung người)
                                                                                                     /gesture/debug_image
                                                                                                     /cmd_vel      (mặc định TẮT)
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
│           │   ├── gesture_node.py        ← NODE CHÍNH: nhận diện cử chỉ
│           │   ├── gesture/               pipeline copy từ code gốc (sửa import cho package)
│           │   ├── image_utils.py         chuyển sensor_msgs/Image <-> numpy
│           │   └── video_publisher.py     công cụ test: phát video/webcam lên /image_front
│           ├── config/gesture_params.yaml
│           ├── launch/gesture.launch.py
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
11. **Đã tạo venv `~/gesture_venv` trên laptop và test thành công.** (venv này đã chuyển vào thư mục dự án, xem mục 15)
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
13. **Chốt cách triển khai lên robot: WORKSPACE RIÊNG** (**đã thay bằng mục 15**: robot dùng git trong `/home/vistar/projects/human_detect`).
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
15. **Chuyển venv vào trong thư mục dự án** (thay cho mục 13 và `~/gesture_venv`).
    - Trên robot, mọi thứ nằm trong `/home/vistar/projects/human_detect/` (lấy code về bằng **git**, không dùng rsync/scp),
      nên venv cũng phải nằm trong đó, không đặt ở `~`.
    - `setup_venv.sh`: mặc định venv = `<thư_mục_dự_án>/gesture_venv`, tính bằng 3 cấp thư mục trên package
      (`<dự_án>/ros2_amr_UI-main/src/human_detection`). Laptop: `/home/kietdz/kiet/human_detect/gesture_venv`;
      robot: `/home/vistar/projects/human_detect/gesture_venv`. `.gitignore` chặn bằng quy tắc `*_venv/`.
    - Sửa lỗi script: `set -u` làm script thoát khi `source /opt/ros/jazzy/setup.bash`, nên bước 4 (kiểm tra) chưa từng chạy.
      Giờ tạm tắt `set -u` khi source.
    - `requirements.txt` ghim `setuptools>=77.0.3,<80` (torch cần ≥77, colcon của ROS cần <80); `pip check` không còn xung đột.
    - Laptop: tạo lại venv ở vị trí mới (venv không chuyển chỗ được), **đã xoá `~/gesture_venv` cũ**. Build lại trong venv:
      dòng đầu file chạy node giờ là đường dẫn tuyệt đối tới python của venv (`#!.../human_detect/gesture_venv/bin/python3`).
      Test video Halt ra lệnh Halt.
    - Robot: đã `git pull`, build và chạy `gesture_node` thành công (log khởi tạo đủ 3 model). Trên robot đã build **bên trong**
      thư mục package (`src/human_detection/install/...`); vẫn chạy được vì colcon tự đặt `COLCON_IGNORE`.
      **Cần chạy lại `setup_venv.sh` trên robot** để tạo venv trong `projects/human_detect/`, rồi xoá `~/gesture_venv` cũ trên robot.
    - Vấn đề đang mở: trên robot, `rqt_image_view` mở `/gesture/debug_image` nhưng chỉ thấy màu xám. Nghi ngờ QoS không khớp
      (node publish Best Effort, rqt subscribe Reliable) hoặc node không nhận được `/image_front`. Chưa kiểm tra xong.
16. **Chạy thử trên laptop với webcam: hoạt động** (`video_publisher source:=0` → `gesture_node` → `rqt_image_view` hiện khung người).
    - Sửa `video_publisher`: `-p source:=0` bị ROS hiểu là số nguyên, gây lỗi `InvalidParameterTypeException`; giờ tham số nhận cả số lẫn chuỗi.
    - `rqt_image_view` bị tắt ngay khi mở (`GLib-GIO-ERROR ... xsettings ... 'antialiasing'`): do **VSCode bản Snap**, terminal trong VSCode
      thừa hưởng `GSETTINGS_SCHEMA_DIR`, `GTK_PATH`... của Snap. Cách xử lý: chạy rqt (và mọi chương trình có giao diện) từ terminal
      thường (`Ctrl+Alt+T`). Đặt `QT_QPA_PLATFORMTHEME` không có tác dụng.
    - rqt **hiện được** `/gesture/debug_image` (Best Effort), nên **không có lỗi QoS**. Màu xám trên robot khả năng cao là do node
      không nhận được `/image_front` lúc đó (cần kiểm tra lại bằng `ros2 topic hz /image_front` trên robot).
    - Giao diện ảnh debug: bảng xác suất giờ nằm **cạnh video** (mặc định bên phải), không che hình. Thêm tham số
      `debug_panel_position` (right/left/bottom/overlay), `debug_panel_scale`, `debug_box_thickness`; đổi được khi node đang chạy
      bằng `ros2 param set /gesture_node ...`.

## Trạng thái hiện tại

- [x] ROS 2 Jazzy cài xong; thư viện Python nằm trong venv `<dự_án>/gesture_venv` (mục 15)
- [x] `colcon build` thành công (build trong venv)
- [x] `gesture_node` chạy được qua ROS **trong venv** với video test và nhận đúng cử chỉ
- [x] Đã viết `requirements.txt` và `setup_venv.sh`, đã test trên laptop
- [x] Đã gỡ các gói cũ cài kiểu `pip --user` trong `~/.local` (laptop gọn, chỉ còn venv)
- [x] Robot: code lấy bằng git vào `/home/vistar/projects/human_detect/`, đã build và chạy được `gesture_node`
- [ ] Robot: chạy lại `setup_venv.sh` để venv nằm trong `projects/human_detect/`
- [ ] Robot: `rqt_image_view` chỉ hiện màu xám; trên laptop rqt chạy bình thường nên cần kiểm tra `/image_front` trên robot (mục 16)
- [ ] Thống nhất với nhóm: `/cmd_vel`, tên topic, mức dùng CPU (xem Ghi chú)
- [ ] MoveForward và MoveInReverse đôi khi nhầm lẫn nhau, cần thử với người thật
- [ ] Chưa test với người thật đứng trước camera robot
- [ ] `/cmd_vel` giữ TẮT. Robot nhận lệnh qua `/gesture`, do phần Navigation của nhóm xử lý
- [ ] FollowMe chưa có bộ bám người (hiện chỉ đứng yên)

## Việc tiếp theo

0. **Trên robot** (code ở `/home/vistar/projects/human_detect`, cập nhật bằng `git pull`):
   ```bash
   cd ~/projects/human_detect && git pull
   ros2_amr_UI-main/src/human_detection/setup_venv.sh    # tạo ~/projects/human_detect/gesture_venv (chỉ cần chạy lại khi requirements.txt đổi)
   rm -rf ~/gesture_venv                                  # venv cũ ngoài thư mục dự án
   source /opt/ros/jazzy/setup.bash
   source ~/projects/human_detect/gesture_venv/bin/activate
   cd ~/projects/human_detect/ros2_amr_UI-main/src/human_detection
   rm -rf build install log && python3 -m colcon build --symlink-install
   source install/setup.bash
   ros2 launch human_detection gesture.launch.py         # mở terminal khác để chạy rqt_image_view / topic echo
   ```
   **Không** sửa package `camera` và các package khác của nhóm.

1. **Test với camera robot thật** qua mạng LAN. Robot đang publish `/image_front` trên domain 0, nên chỉ cần **không** đặt
   `ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`:
   ```bash
   source ~/kiet/human_detect/gesture_venv/bin/activate
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
