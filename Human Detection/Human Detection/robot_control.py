"""
robot_control.py - Chạy camera -> DD-Net -> lệnh di chuyển cho robot.

    python robot_control.py --ckpt checkpoints/scratch_f16/best.pt --backend print --show
    python robot_control.py --ckpt checkpoints/scratch_f16/best.pt --backend ros2 --camera 0

Luôn chạy thử bằng --backend print (robot không động) trước khi nối robot thật.

An toàn:
  - Lệnh chỉ đổi khi cùng một cử chỉ có xác suất >= --conf trong --confirm lần dự đoán liên tiếp.
  - Mất người / mất khung xương quá --lost-sec -> dừng.
  - Gửi lại lệnh hiện tại liên tục (watchdog của robot cần); không có cử chỉ mới thì giữ lệnh
    tối đa --hold-sec rồi dừng.
  - Halt luôn dừng ngay (--confirm = 1 cho Halt).
  - Nhấn q (khi --show) hoặc Ctrl+C -> dừng robot rồi thoát.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Dict, Optional

import cv2

import config
from camera_source import open_source
from gesture_recognizer import GestureRecognizer


# ---------------------------------------------------------------------------
# Cử chỉ -> chuyển động. SỬA BẢNG NÀY theo ý nghĩa bạn muốn.  (vx m/s, wz rad/s)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Motion:
    vx: float = 0.0
    wz: float = 0.0


GESTURE_TO_MOTION: Dict[str, Motion] = {
    "MoveForward":   Motion(vx=0.20),
    "MoveInReverse": Motion(vx=-0.15),
    "Halt":          Motion(),
    "Left":          Motion(),          # chưa gắn chuyển động: bạn tự gắn hành vi riêng
    "Right":         Motion(),
    "None":          Motion(),          # không có lệnh: đứng yên
}
IDLE = "None"
STOP = Motion()


# ---------------------------------------------------------------------------
# Backend nối với robot
# ---------------------------------------------------------------------------
class PrintBackend:
    """Chỉ in lệnh ra màn hình. Dùng để thử."""
    def send(self, m: Motion, gesture: str = "") -> None:
        pass          # in ở vòng lặp chính khi lệnh đổi

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


class Ros2Backend:
    """
    Publish lên ROS2 (cần ROS2 đã source):
      - geometry_msgs/Twist  -> topic cmd_vel (mặc định /cmd_vel)
      - std_msgs/String      -> /gesture_cmd: tên cử chỉ đang được chấp nhận (để node khác dùng/in ra)
    """
    def __init__(self, topic: str = "/cmd_vel", gesture_topic: str = "/gesture_cmd"):
        try:
            import rclpy
            from geometry_msgs.msg import Twist
            from std_msgs.msg import String
        except ImportError as e:
            raise SystemExit("Chưa có ROS2 trong Python này. Chạy: source /opt/ros/<distro>/setup.bash "
                             "(Ubuntu 22.04 = humble) rồi chạy lại.") from e
        self._rclpy, self._Twist, self._String = rclpy, Twist, String
        rclpy.init()
        self.node = rclpy.create_node("gesture_teleop")
        self.pub = self.node.create_publisher(Twist, topic, 10)
        self.pub_gesture = self.node.create_publisher(String, gesture_topic, 10)

    def send(self, m: Motion, gesture: str = "") -> None:
        msg = self._Twist()
        msg.linear.x, msg.angular.z = m.vx, m.wz
        self.pub.publish(msg)
        g = self._String()
        g.data = gesture
        self.pub_gesture.publish(g)

    def stop(self) -> None:
        self.send(STOP, "Stop")

    def close(self) -> None:
        self.stop()
        self.node.destroy_node()
        self._rclpy.shutdown()


def make_backend(name: str, topic: str):
    if name == "print":
        return PrintBackend()
    if name == "ros2":
        return Ros2Backend(topic)
    raise SystemExit(f"backend không hỗ trợ: {name}")


# ---------------------------------------------------------------------------
# Quyết định lệnh từ chuỗi dự đoán
# ---------------------------------------------------------------------------
class CommandFilter:
    def __init__(self, conf: float, confirm: int, hold_sec: float, lost_sec: float):
        self.conf, self.confirm, self.hold_sec, self.lost_sec = conf, confirm, hold_sec, lost_sec
        self.current = IDLE
        self._cand, self._count = None, 0
        self._last_set = time.time()
        self._last_seen = time.time()

    def step(self, label: Optional[str], prob: float, now: float) -> str:
        if label is None:                                     # chưa có dự đoán / mất người
            self._cand, self._count = None, 0
            if now - self._last_seen > self.lost_sec:
                self.current = IDLE
            return self.current
        self._last_seen = now

        if prob >= self.conf and label != self.current:
            self._count = self._count + 1 if label == self._cand else 1
            self._cand = label
            need = 1 if label == "Halt" else self.confirm
            if self._count >= need:
                self.current, self._last_set = label, now
                self._cand, self._count = None, 0
        elif label == self.current and prob >= self.conf:
            self._last_set = now                              # vẫn đang làm cử chỉ đó -> giữ lệnh
            self._cand, self._count = None, 0
        else:
            self._cand, self._count = None, 0

        if now - self._last_set > self.hold_sec:              # lâu không thấy cử chỉ -> dừng
            self.current = IDLE
        return self.current


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Điều khiển robot bằng cử chỉ (DD-Net)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--camera", default="0", help="realsense | chỉ số webcam | đường dẫn video/RTSP")
    ap.add_argument("--backend", choices=["print", "ros2"], default="print")
    ap.add_argument("--topic", default="/cmd_vel")
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--confirm", type=int, default=3, help="số lần dự đoán liên tiếp để đổi lệnh")
    ap.add_argument("--hold-sec", type=float, default=3.0)
    ap.add_argument("--lost-sec", type=float, default=1.0)
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--show", action="store_true", help="hiện cửa sổ camera + kết quả")
    args = ap.parse_args()

    rec = GestureRecognizer(args.ckpt, yolo_weights=None if args.no_yolo else config.YOLO_WEIGHTS)
    print(f"Checkpoint: T={rec.T}, fps={rec.fps:g}, lớp={rec.class_names}")
    flt = CommandFilter(args.conf, args.confirm, args.hold_sec, args.lost_sec)
    robot = make_backend(args.backend, args.topic)
    cap, _ = open_source(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được camera {args.camera}")

    last_cmd, t_pred = None, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Mất tín hiệu camera -> dừng")
                break
            now = time.time()
            res = rec.update(frame, now)
            cmd = flt.step(res.label, res.prob, now)
            motion = GESTURE_TO_MOTION.get(cmd, STOP)
            robot.send(motion, cmd)
            if now - t_pred >= 0.5:                               # dự đoán thô của model, để so với lệnh
                t_pred = now
                print(f"   model: {res.label or res.reason} {res.prob * 100:3.0f}%   lệnh hiện tại: {cmd}", flush=True)
            if cmd != last_cmd:
                print(f"[{time.strftime('%H:%M:%S')}] {cmd}  ->  vx={motion.vx:+.2f} wz={motion.wz:+.2f}",
                      flush=True)
                last_cmd = cmd
            if args.show:
                txt = f"{res.label or res.reason} {res.prob:.2f} | CMD: {cmd}"
                cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("gesture", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        robot.stop()
        robot.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Đã dừng robot.")


if __name__ == "__main__":
    main()
