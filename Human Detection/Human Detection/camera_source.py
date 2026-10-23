from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class RealSenseCapture:
    """
    Đọc luồng MÀU của RealSense. Chỉ cần ảnh màu: pipeline YOLO + MediaPipe chạy trên RGB.

    Dùng 640x480 @ 30 FPS: đủ cho cử chỉ, nhẹ cho USB, và gần với video RoCoG (độ phân giải
    thấp, người nhỏ). Nếu muốn lấy thêm khoảng cách tới người (ví dụ để bám theo người), đặt
    with_depth=True; depth được căn theo ảnh màu, đọc bằng .depth_at(x, y).
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 serial: Optional[str] = None, with_depth: bool = False,
                 warmup_frames: int = 30, timeout_ms: int = 5000):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise ImportError("Chưa cài pyrealsense2. Cài bằng: pip install pyrealsense2") from e
        self._rs = rs
        self._timeout = timeout_ms
        self._opened = False
        self._depth_frame = None
        self.with_depth = with_depth
        self.width, self.height, self.fps = width, height, fps

        ctx = rs.context()
        if len(ctx.query_devices()) == 0:
            raise RuntimeError("Không thấy camera RealSense nào. Kiểm tra cáp USB 3 và udev rules.")

        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if with_depth:
            cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.pipeline = rs.pipeline()
        profile = self.pipeline.start(cfg)
        self._align = rs.align(rs.stream.color) if with_depth else None
        if with_depth:
            self._depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self._opened = True

        # bỏ vài chục frame đầu: tự động phơi sáng/cân bằng trắng chưa ổn định
        for _ in range(warmup_frames):
            self.pipeline.wait_for_frames(timeout_ms)

    def isOpened(self) -> bool:
        return self._opened

    def read(self):
        try:
            frames = self.pipeline.wait_for_frames(self._timeout)
        except RuntimeError:          # quá thời gian chờ / rút cáp
            return False, None
        if self._align is not None:
            frames = self._align.process(frames)
            self._depth_frame = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not color:
            return False, None
        return True, np.asanyarray(color.get_data())

    def depth_at(self, x: int, y: int) -> Optional[float]:
        """Khoảng cách (mét) tại pixel (x, y) của ảnh màu; None nếu không có depth/điểm hỏng."""
        if self._depth_frame is None:
            return None
        d = self._depth_frame.get_distance(int(x), int(y))
        return d if d > 0 else None

    def get(self, prop):
        return float(self.fps) if prop == cv2.CAP_PROP_FPS else 0.0

    def release(self) -> None:
        if self._opened:
            self._opened = False
            try:
                self.pipeline.stop()
            except Exception:
                pass


def open_source(spec: str, **rs_kwargs):
    """Trả (cap, is_file). spec: 'realsense', chỉ số webcam, hoặc đường dẫn video/RTSP."""
    if spec.lower() in ("realsense", "rs", "d435i"):
        return RealSenseCapture(**rs_kwargs), False
    if spec.isdigit():
        return cv2.VideoCapture(int(spec)), False
    return cv2.VideoCapture(spec), True
