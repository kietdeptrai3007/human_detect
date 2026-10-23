"""
Node ROS 2 nhận diện 5 cử chỉ (MoveForward, Halt, MoveInReverse, Left, Right) trên ảnh camera trước.

Pipeline (giống Human Detection/Human Detection/gesture_recognizer.py):
    ảnh -> YOLO tìm người (best.pt) -> MediaPipe Pose -> khung xương 15 khớp
        -> cửa sổ 32 frame @ 10 FPS (3,2 giây) -> DD-Net -> cử chỉ

Subscribe : /image_front              (sensor_msgs/Image, bgr8)
Publish   : /gesture                  (std_msgs/Int32) số hiệu lệnh đã lọc: 0 MoveForward, 1 Halt, 2 MoveInReverse,
                                        3 Left, 4 Right, 5 None (không có lệnh) (bảng GESTURE_IDS trong command_filter.py)
            /gesture/name             (std_msgs/String) tên của lệnh trên, để đọc/debug
            /gesture/raw              (std_msgs/String, JSON) dự đoán thô của model + xác suất 5 lớp
            /gesture/person           (vision_msgs/Detection2DArray) khung người đang được theo dõi (0 hoặc 1 người)
            /people/tracks            (vision_msgs/Detection2DArray) MỌI người trong khung, det.id = track_id
                                      (ByteTrack); chỉ khi enable_tracking = true
            /people/target_id         (std_msgs/Int32) track_id của người chủ đang khoá, -1 = chưa khoá
            /people/target_state      (std_msgs/String) IDLE | SEARCHING | LOCKED | LOST (khi enable_target_lock)
            /gesture/debug_image      (sensor_msgs/Image) ảnh có khung người + bảng xác suất
            /cmd_vel                  (geometry_msgs/Twist) CHỈ khi enable_cmd_vel = true
"""

import json
import os
import threading
import time

import cv2
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .image_utils import bgr_to_imgmsg, imgmsg_to_bgr

# cv2.putText không vẽ được chữ có dấu -> đổi lý do sang ASCII khi vẽ ảnh debug
REASON_ASCII = {
    'đang thu thập frame': 'collecting frames...',
    'quá ít frame dùng được': 'too few skeleton frames',
    'mất khung xương': 'skeleton lost',
}


def draw_panel(frame, res) -> None:
    """Một dòng ở góc ảnh: cử chỉ dự đoán và độ tin cậy (%). Chưa có dự đoán thì hiện lý do."""
    reason = REASON_ASCII.get(res.reason, res.reason) if res.person else 'no person'
    text = f"{res.label} {res.prob * 100:.0f}%" if res.label else reason or "..."
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
    cv2.putText(frame, text, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


class GestureNode(Node):

    def __init__(self):
        super().__init__('gesture_node')

        # 1. Declare & read parameters
        self.declare_parameter('image_topic', '/image_front')
        self.declare_parameter('yolo_model_path', '')        # '' -> models/best.pt
        self.declare_parameter('gesture_model_path', '')     # '' -> models/ddnet_gesture.pt
        self.declare_parameter('pose_model_path', '')        # '' -> models/pose_landmarker_lite.task
        self.declare_parameter('use_yolo', True)
        self.declare_parameter('enable_tracking', True)      # ByteTrack + publish /people/tracks
        self.declare_parameter('tracker_config', '')         # '' -> config/bytetrack_person.yaml
        # Khoá "người chủ": chỉ cử chỉ của người đã khoá bằng lock_gesture mới điều khiển robot
        self.declare_parameter('enable_target_lock', True)   # cần enable_tracking
        self.declare_parameter('lock_gesture', 'Right')
        self.declare_parameter('halt_from_any', True)        # Halt của ứng viên chưa khoá vẫn có hiệu lực
        self.declare_parameter('lost_grace_sec', 1.5)        # chờ nối lại người chủ bị đổi ID (>= lost_sec)
        self.declare_parameter('miss_tolerance_sec', 0.3)    # người chủ trượt vài frame: chưa coi là mất
        self.declare_parameter('candidate_dwell_sec', 5.0)   # nhiều người: cứ bấy nhiêu giây đổi ứng viên (0 = tắt)
        self.declare_parameter('reassoc_dx_ratio', 1.5)
        self.declare_parameter('reassoc_size_ratio', 1.6)
        self.declare_parameter('device', '')                 # '' = tự chọn, 'cpu', 'cuda'
        self.declare_parameter('qos_reliability', 'sensor_data')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('conf_threshold', 0.8)
        self.declare_parameter('confirm_count', 3)
        self.declare_parameter('hold_sec', 3.0)
        self.declare_parameter('lost_sec', 1.0)
        self.declare_parameter('enable_cmd_vel', False)
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')

        models_dir = os.path.join(get_package_share_directory('human_detection'), 'models')

        def model_param(name, default_file):
            path = self.get_parameter(name).value or os.path.join(models_dir, default_file)
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Khong tim thay model ({name}): {path}')
            return path

        yolo_path = model_param('yolo_model_path', 'best.pt')
        gesture_path = model_param('gesture_model_path', 'ddnet_gesture.pt')
        pose_path = model_param('pose_model_path', 'pose_landmarker_lite.task')
        self.image_topic = self.get_parameter('image_topic').value
        use_yolo = bool(self.get_parameter('use_yolo').value)
        self.enable_tracking = use_yolo and bool(self.get_parameter('enable_tracking').value)
        tracker_cfg = None
        if self.enable_tracking:
            tracker_cfg = self.get_parameter('tracker_config').value or os.path.join(
                get_package_share_directory('human_detection'), 'config', 'bytetrack_person.yaml')
            if not os.path.isfile(tracker_cfg):
                raise FileNotFoundError(f'Khong tim thay tracker_config: {tracker_cfg}')
        device = self.get_parameter('device').value
        qos_type = self.get_parameter('qos_reliability').value
        self.publish_debug = bool(self.get_parameter('publish_debug_image').value)
        self.enable_cmd_vel = bool(self.get_parameter('enable_cmd_vel').value)

        # 2. Load pipeline (import muộn để lỗi thiếu thư viện hiện rõ qua log)
        import torch
        from .gesture.command_filter import CommandFilter, GESTURE_IDS, GESTURE_TO_MOTION, IDLE, STOP
        from .gesture.gesture_recognizer import GestureRecognizer
        self._motions, self._stop = GESTURE_TO_MOTION, STOP
        self._ids, self._idle = GESTURE_IDS, IDLE

        if not device:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.recognizer = GestureRecognizer(
            gesture_path, yolo_weights=yolo_path if use_yolo else None,
            device=device, pose_model_path=pose_path, tracker_cfg=tracker_cfg)
        self.names = self.recognizer.class_names
        self.selector = None
        if self.enable_tracking and bool(self.get_parameter('enable_target_lock').value):
            from .tracking.target_selector import TargetSelector
            lock_gesture = self.get_parameter('lock_gesture').value
            if lock_gesture not in self.names:
                raise ValueError(f'lock_gesture "{lock_gesture}" khong co trong {self.names}')
            self.halt_from_any = bool(self.get_parameter('halt_from_any').value)
            self.selector = TargetSelector(
                lock_gesture=lock_gesture,
                lost_grace_sec=float(self.get_parameter('lost_grace_sec').value),
                miss_tolerance_sec=float(self.get_parameter('miss_tolerance_sec').value),
                candidate_dwell_sec=float(self.get_parameter('candidate_dwell_sec').value),
                reassoc_dx_ratio=float(self.get_parameter('reassoc_dx_ratio').value),
                reassoc_size_ratio=float(self.get_parameter('reassoc_size_ratio').value))
        self._epoch = self.recognizer.window_epoch
        self.filter = CommandFilter(
            conf=float(self.get_parameter('conf_threshold').value),
            confirm=int(self.get_parameter('confirm_count').value),
            hold_sec=float(self.get_parameter('hold_sec').value),
            lost_sec=float(self.get_parameter('lost_sec').value))
        if self.selector is not None and self.selector.lost_grace_sec < self.filter.lost_sec:
            self.get_logger().warn('lost_grace_sec nho hon lost_sec: lenh cu co the con sot lai khi nguoi chu bi mat')

        # 3. QoS subscriber phải tương thích với camera (mặc định Best Effort)
        reliability = (ReliabilityPolicy.RELIABLE if qos_type == 'reliable'
                       else ReliabilityPolicy.BEST_EFFORT)
        sub_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                             reliability=reliability, durability=DurabilityPolicy.VOLATILE)

        self.create_subscription(Image, self.image_topic, self.image_callback, sub_qos)
        self.gesture_pub = self.create_publisher(Int32, '/gesture', 10)
        self.gesture_name_pub = self.create_publisher(String, '/gesture/name', 10)
        self.raw_pub = self.create_publisher(String, '/gesture/raw', 10)
        self.person_pub = self.create_publisher(Detection2DArray, '/gesture/person', 10)
        if self.enable_tracking:
            self.tracks_pub = self.create_publisher(Detection2DArray, '/people/tracks', 10)
        if self.selector is not None:
            self.target_id_pub = self.create_publisher(Int32, '/people/target_id', 10)
            self.target_state_pub = self.create_publisher(String, '/people/target_state', 10)
        if self.publish_debug:
            self.debug_pub = self.create_publisher(Image, '/gesture/debug_image', sub_qos)
        if self.enable_cmd_vel:
            self.cmd_pub = self.create_publisher(
                Twist, self.get_parameter('cmd_vel_topic').value, 10)

        # 4. Thread suy luận riêng: callback chỉ giữ frame mới nhất
        self._lock = threading.Lock()
        self._pending = None
        self._new_frame = threading.Event()
        self._running = True
        self._last_cmd = None
        self._worker = threading.Thread(target=self.inference_loop, daemon=True)
        self._worker.start()

        self.get_logger().info(
            'GestureNode initialized.\n'
            f'  - Image topic: {self.image_topic} (qos={qos_type})\n'
            f'  - YOLO: {yolo_path if use_yolo else "TAT (dung ca khung hinh)"}\n'
            f'  - Tracking: {tracker_cfg if self.enable_tracking else "tat"}\n'
            f'  - Khoa nguoi chu: {("BAT, lock_gesture=" + self.selector.lock_gesture) if self.selector else "tat"}\n'
            f'  - DD-Net: {gesture_path} (T={self.recognizer.T}, fps={self.recognizer.fps:g})\n'
            f'  - Pose: {pose_path}\n'
            f'  - Classes: {self.names}\n'
            f'  - device={device} cmd_vel={"BAT" if self.enable_cmd_vel else "tat"}')

    def image_callback(self, msg: Image):
        with self._lock:
            self._pending = msg
        self._new_frame.set()

    def inference_loop(self):
        while self._running and rclpy.ok():
            got = self._new_frame.wait(timeout=0.2)
            with self._lock:
                msg, self._pending = self._pending, None
                self._new_frame.clear()

            if not got or msg is None:
                # Không có ảnh mới (camera dừng/mất kết nối): vẫn cập nhật bộ lọc để tự về None
                self.publish_command(self.gate_command(self.filter.step(None, 0.0, time.time())))
                continue

            try:
                frame = imgmsg_to_bgr(msg)
            except ValueError as e:
                self.get_logger().error(str(e), throttle_duration_sec=5.0)
                continue

            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            target_fn = None
            if self.selector is not None:
                target_fn = lambda tracks: self.selector.update(tracks, time.time())  # noqa: E731
            try:
                res = self.recognizer.update(frame, stamp, target_fn)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f'Loi pipeline gesture: {e}', throttle_duration_sec=5.0)
                continue

            if self.recognizer.window_epoch != self._epoch:      # đã đổi người -> bỏ lệnh đang xác nhận dở
                self._epoch = self.recognizer.window_epoch
                self.filter.reset()
            now = time.time()
            cmd = self.filter.step(res.label, res.prob, now)
            if self.selector is not None:
                self.selector.observe_command(cmd, now)          # có thể khoá người chủ ở đây
            cmd = self.gate_command(cmd)
            self.publish_command(cmd)
            self.publish_target()
            self.publish_raw(res)
            self.publish_person(msg.header)
            if self.enable_tracking:
                self.publish_tracks(msg.header)
            if self.publish_debug:
                self.publish_debug_image(frame, res, cmd, msg.header)

    def gate_command(self, cmd: str) -> str:
        """Chưa khoá người chủ thì chỉ Halt (nếu halt_from_any) được đi qua, còn lại giữ None."""
        if self.selector is None or self.selector.state == 'LOCKED':
            return cmd
        if self.selector.state == 'SEARCHING':
            return cmd if (cmd == 'Halt' and self.halt_from_any) else 'None'
        return cmd      # IDLE / LOST: bộ lọc tự về None sau lost_sec

    def publish_target(self):
        if self.selector is None:
            return
        cid = self.selector.commander_id
        self.target_id_pub.publish(Int32(data=-1 if cid is None else int(cid)))
        self.target_state_pub.publish(String(data=self.selector.state))

    def publish_command(self, cmd: str):
        self.gesture_pub.publish(Int32(data=self._ids.get(cmd, self._ids[self._idle])))
        self.gesture_name_pub.publish(String(data=cmd))
        if self.enable_cmd_vel:
            m = self._motions.get(cmd, self._stop)
            twist = Twist()
            twist.linear.x, twist.angular.z = m.vx, m.wz
            self.cmd_pub.publish(twist)
        if cmd != self._last_cmd:
            self.get_logger().info(f'Lenh: {self._last_cmd} -> {cmd}')
            self._last_cmd = cmd

    def publish_raw(self, res):
        data = {
            'label': res.label,
            'prob': round(res.prob, 4),
            'person': res.person,
            'skeleton': bool(self.recognizer._frames and self.recognizer._frames[-1][1]),
            'reason': res.reason,
            'probs': (None if res.probs is None
                      else {n: round(float(p), 4) for n, p in zip(self.names, res.probs)}),
        }
        self.raw_pub.publish(String(data=json.dumps(data, ensure_ascii=False)))

    def publish_person(self, header):
        """Khung người đang theo dõi (lấy từ YOLO bên trong pipeline, không chạy YOLO thêm lần nữa)."""
        out = Detection2DArray()
        out.header = header
        crop = self.recognizer.last_crop
        if crop is not None:
            x1, y1, x2, y2 = (float(v) for v in crop.person_box)
            det = Detection2D()
            det.header = header
            det.bbox.center.position.x = (x1 + x2) / 2
            det.bbox.center.position.y = (y1 + y2) / 2
            det.bbox.size_x = x2 - x1
            det.bbox.size_y = y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = 'Person'
            hyp.hypothesis.score = crop.conf   # 0 = YOLO trượt frame này, đang dùng lại bbox cũ
            det.results.append(hyp)
            out.detections.append(det)
        self.person_pub.publish(out)

    def publish_tracks(self, header):
        """Mọi người ByteTrack đang theo dõi (kể cả người không phải người đang nhận diện cử chỉ).
        Luôn publish, kể cả danh sách rỗng, để trajectory_node biết frame này không thấy ai."""
        out = Detection2DArray()
        out.header = header
        for p in self.recognizer.last_tracks:
            x1, y1, x2, y2 = p.bbox
            det = Detection2D()
            det.header = header
            det.id = str(p.track_id)
            det.bbox.center.position.x = (x1 + x2) / 2
            det.bbox.center.position.y = (y1 + y2) / 2
            det.bbox.size_x = x2 - x1
            det.bbox.size_y = y2 - y1
            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = 'Person'
            hyp.hypothesis.score = p.conf
            det.results.append(hyp)
            out.detections.append(det)
        self.tracks_pub.publish(out)

    def publish_debug_image(self, frame, res, cmd, header):
        img = frame.copy()
        crop = self.recognizer.last_crop
        if crop is not None:
            x1, y1, x2, y2 = crop.person_box.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2),
                          (0, 255, 0) if crop.detected else (0, 165, 255), 2)
        draw_panel(img, res)
        if self.selector is not None:
            cid = self.selector.commander_id
            cv2.putText(img, f"{self.selector.state}" + ("" if cid is None else f" id{cid}"),
                        (8, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        self.debug_pub.publish(bgr_to_imgmsg(img, header))

    def destroy_node(self):
        self._running = False
        self._new_frame.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)
        if self.enable_cmd_vel:
            self.cmd_pub.publish(Twist())   # dừng robot trước khi thoát
        self.recognizer.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GestureNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001
        rclpy.logging.get_logger('main').fatal(f'Loi khoi chay gesture_node: {e}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
