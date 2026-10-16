"""
Công cụ test: phát video (hoặc webcam) lên topic ảnh, giả lập package camera.

    ros2 run human_detection video_publisher --ros-args -p source:=/duong/dan/video.mp4
    ros2 run human_detection video_publisher --ros-args -p source:=0          # webcam

Ảnh được resize giữ tỉ lệ + viền đen về 640x360 bgr8, QoS Best Effort, giống intel_camera_node.
"""

import cv2
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from .image_utils import bgr_to_imgmsg


class VideoPublisher(Node):

    def __init__(self):
        super().__init__('video_publisher')
        # dynamic_typing: nhận cả "source:=0" (ROS hiểu là số) lẫn đường dẫn video (chuỗi)
        self.declare_parameter('source', '0', ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('topic', '/image_front')
        self.declare_parameter('frame_id', 'camera_front_link')
        self.declare_parameter('width', 640)
        self.declare_parameter('height', 360)
        self.declare_parameter('loop', True)

        src = str(self.get_parameter('source').value)
        self.frame_id = self.get_parameter('frame_id').value
        self.size = (int(self.get_parameter('width').value), int(self.get_parameter('height').value))
        self.loop = bool(self.get_parameter('loop').value)
        self.is_file = not src.isdigit()

        self.cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if not self.cap.isOpened():
            raise RuntimeError(f'Khong mo duoc nguon video: {src}')
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.pub = self.create_publisher(Image, self.get_parameter('topic').value,
                                         qos_profile_sensor_data)
        self.timer = self.create_timer(1.0 / fps, self.tick)
        self.get_logger().info(f'Phat {src} -> {self.get_parameter("topic").value} @ {fps:.1f} FPS')

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            if self.is_file and self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                self.get_logger().info('Het video.')
                self.timer.cancel()
                return
        frame = self.letterbox(frame)
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id=self.frame_id)
        self.pub.publish(bgr_to_imgmsg(frame, header))

    def letterbox(self, frame):
        """Resize giữ nguyên tỉ lệ rồi thêm viền đen (không bóp méo người trong ảnh)."""
        W, H = self.size
        h, w = frame.shape[:2]
        s = min(W / w, H / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        resized = cv2.resize(frame, (nw, nh))
        x0, y0 = (W - nw) // 2, (H - nh) // 2
        return cv2.copyMakeBorder(resized, y0, H - nh - y0, x0, W - nw - x0,
                                  cv2.BORDER_CONSTANT, value=(0, 0, 0))

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = VideoPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
