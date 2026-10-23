"""
Node ROS 2 dự đoán quỹ đạo của mọi người đang được theo dõi.

Subscribe : /people/tracks         (vision_msgs/Detection2DArray) từ gesture_node; det.id = track_id
Publish   : /people/trajectories   (std_msgs/String, JSON) vị trí, vận tốc, các điểm dự đoán + sigma
            /people/markers        (visualization_msgs/MarkerArray) quỹ đạo cho rviz,
                                   CHỈ khi use_ground = true
Subscribe thêm /people/target_id (std_msgs/Int32, từ gesture_node): người có id này được gắn
"is_commander": true trong JSON.

Hai chế độ (tham số use_ground):
  false: toạ độ là pixel của ảnh (điểm chân bbox), không cần đo camera
  true : toạ độ là mét trên mặt đất, hệ robot (x trước, y trái); cần config/camera_model.yaml đã đo
"""

import json
import os

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from rclpy.node import Node
from std_msgs.msg import Int32, String
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection2DArray

from .tracking.camera_model import CameraModel
from .tracking.trajectory_predictor import TrajectoryPredictor
from .tracking.types import Observation, TrackedPerson


class TrajectoryNode(Node):

    def __init__(self):
        super().__init__('trajectory_node')

        self.declare_parameter('tracks_topic', '/people/tracks')
        self.declare_parameter('use_ground', False)
        self.declare_parameter('camera_model_path', '')
        self.declare_parameter('horizon_sec', 2.0)
        self.declare_parameter('step_sec', 0.25)
        self.declare_parameter('min_hits', 5)
        self.declare_parameter('lost_sec', 1.0)
        self.declare_parameter('accel_std_image', 50.0)
        self.declare_parameter('accel_std_ground', 1.0)
        self.declare_parameter('pixel_std', 6.0)
        self.declare_parameter('publish_markers', True)
        self.declare_parameter('marker_frame_id', 'base_link')

        self.use_ground = bool(self.get_parameter('use_ground').value)
        self.pixel_std = float(self.get_parameter('pixel_std').value)
        self.marker_frame = self.get_parameter('marker_frame_id').value
        self.publish_markers = self.use_ground and bool(self.get_parameter('publish_markers').value)
        self.step_sec = float(self.get_parameter('step_sec').value)

        self.camera = None
        if self.use_ground:
            path = self.get_parameter('camera_model_path').value or os.path.join(
                get_package_share_directory('human_detection'), 'config', 'camera_model.yaml')
            if not os.path.isfile(path):
                raise FileNotFoundError(f'Khong tim thay camera model: {path}')
            self.camera = CameraModel.from_yaml(path)

        accel = float(self.get_parameter('accel_std_ground' if self.use_ground
                                         else 'accel_std_image').value)
        self.predictor = TrajectoryPredictor(
            horizon_sec=float(self.get_parameter('horizon_sec').value),
            step_sec=self.step_sec,
            min_hits=int(self.get_parameter('min_hits').value),
            lost_sec=float(self.get_parameter('lost_sec').value),
            accel_std=accel,
            init_vel_std=2.0 if self.use_ground else 200.0)

        self.create_subscription(Detection2DArray, self.get_parameter('tracks_topic').value,
                                 self.tracks_callback, 10)
        self.create_subscription(Int32, '/people/target_id', self.target_callback, 10)
        self._commander_id = -1
        self.traj_pub = self.create_publisher(String, '/people/trajectories', 10)
        if self.publish_markers:
            self.marker_pub = self.create_publisher(MarkerArray, '/people/markers', 10)
        self._last_marker_ids = set()

        self.get_logger().info(
            'TrajectoryNode initialized.\n'
            f'  - Che do: {"mat dat (met)" if self.use_ground else "anh (pixel)"}\n'
            f'  - horizon={self.predictor.horizon_sec}s step={self.step_sec}s '
            f'min_hits={self.predictor.min_hits} lost={self.predictor.lost_sec}s accel_std={accel}')

    @staticmethod
    def _to_tracked(det) -> TrackedPerson:
        cx, cy = det.bbox.center.position.x, det.bbox.center.position.y
        w, h = det.bbox.size_x, det.bbox.size_y
        score = det.results[0].hypothesis.score if det.results else 0.0
        return TrackedPerson(int(det.id), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), float(score))

    def target_callback(self, msg: Int32):
        self._commander_id = msg.data

    def tracks_callback(self, msg: Detection2DArray):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        observations = []
        for det in msg.detections:
            try:
                person = self._to_tracked(det)
            except ValueError:          # det.id rong / khong phai so
                continue
            if self.camera is not None:
                ob = self.camera.locate(person)
            else:
                fx, fy = person.foot
                ob = Observation(person.track_id, fx, fy, self.pixel_std)
            if ob is not None:
                observations.append(ob)

        trajectories = self.predictor.update(stamp, observations)

        payload = {
            'stamp': stamp,
            'frame': 'ground' if self.use_ground else 'image',
            'unit': 'm' if self.use_ground else 'px',
            'commander_id': self._commander_id,
            'people': [dict(t.to_dict(), is_commander=(t.track_id == self._commander_id))
                       for t in trajectories],
        }
        self.traj_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        if self.publish_markers:
            self.publish_marker_array(msg.header.stamp, trajectories)

    def publish_marker_array(self, stamp, trajectories):
        arr = MarkerArray()
        ids = set()
        for t in trajectories:
            line = Marker()
            line.header.frame_id = self.marker_frame
            line.header.stamp = stamp
            line.ns, line.id = 'people_path', t.track_id
            line.type, line.action = Marker.LINE_STRIP, Marker.ADD
            line.scale.x = 0.05
            line.color.r, line.color.g, line.color.b, line.color.a = 1.0, 0.6, 0.0, 1.0
            for x, y in [(t.x, t.y)] + [(p.x, p.y) for p in t.future]:
                line.points.append(Point(x=float(x), y=float(y), z=0.05))
            arr.markers.append(line)
            ids.add(t.track_id)
        for gone in self._last_marker_ids - ids:       # xoa marker cua nguoi da het track
            m = Marker()
            m.header.frame_id = self.marker_frame
            m.ns, m.id, m.action = 'people_path', gone, Marker.DELETE
            arr.markers.append(m)
        self._last_marker_ids = ids
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TrajectoryNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:  # noqa: BLE001
        rclpy.logging.get_logger('main').fatal(f'Loi khoi chay trajectory_node: {e}')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
