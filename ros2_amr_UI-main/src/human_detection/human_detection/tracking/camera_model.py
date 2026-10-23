"""
Mô hình camera pinhole: đổi pixel thành vị trí trên mặt đất trong hệ toạ độ robot.

Hệ toạ độ robot: x = phía trước, y = bên TRÁI (mét), mặt đất phẳng.
Camera đặt ở độ cao camera_height_m, cúi xuống pitch_deg độ (dương = cúi), không nghiêng ngang.

Hai cách ước lượng, chọn tự động:
  foot   : giao tia nhìn qua điểm chân với mặt đất (tốt hơn, cần thấy chân người).
  height : khi chân bị cắt khỏi ảnh, suy khoảng cách từ chiều cao bbox giả định cao person_height_m
           (kém chính xác hơn nhiều, nên std lớn).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .types import Observation, TrackedPerson


@dataclass
class CameraModel:
    image_width: int = 640
    image_height: int = 360
    hfov_deg: float = 69.0
    camera_height_m: float = 0.5
    pitch_deg: float = 0.0
    person_height_m: float = 1.7
    pixel_err: float = 4.0            # sai số pixel của đáy bbox, dùng ước lượng std
    feet_margin_px: float = 6.0       # đáy bbox cách mép dưới ảnh ít hơn mức này = chân bị cắt

    def __post_init__(self):
        self.fx = (self.image_width / 2) / math.tan(math.radians(self.hfov_deg) / 2)
        self.fy = self.fx                                        # pixel vuông
        self.cx = self.image_width / 2
        self.cy = self.image_height / 2
        self._cos = math.cos(math.radians(self.pitch_deg))
        self._sin = math.sin(math.radians(self.pitch_deg))

    @classmethod
    def from_yaml(cls, path: str) -> "CameraModel":
        import yaml
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        data = data.get("camera_model", data)
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in fields})

    # ------------------------------------------------------------------
    def pixel_to_ground(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """Pixel -> (x, y) trên mặt đất; None nếu tia nhìn không chạm đất (ở trên đường chân trời)."""
        r = (u - self.cx) / self.fx
        d = (v - self.cy) / self.fy
        forward = self._cos - d * self._sin
        down = self._sin + d * self._cos
        if down <= 1e-6:
            return None
        t = self.camera_height_m / down
        return t * forward, -t * r

    def ground_to_pixel(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """(x, y) trên mặt đất -> pixel; None nếu nằm sau camera."""
        h = self.camera_height_m
        f = x * self._cos + h * self._sin
        d = -x * self._sin + h * self._cos
        if f <= 1e-6:
            return None
        return self.cx + self.fx * (-y) / f, self.cy + self.fy * d / f

    # ------------------------------------------------------------------
    def locate(self, person: TrackedPerson) -> Optional[Observation]:
        """TrackedPerson -> Observation trên mặt đất, None nếu không định vị được."""
        feet_visible = person.bbox[3] < self.image_height - self.feet_margin_px
        if feet_visible:
            pt = self.pixel_to_ground(*person.foot)
            if pt is not None and pt[0] > 0:
                x, y = pt
                rng = math.hypot(x, y)
                std_range = max(0.1, x * x * self.pixel_err / (self.fy * self.camera_height_m))
                std_lat = max(0.05, rng * self.pixel_err / self.fx)
                return Observation(person.track_id, x, y, math.hypot(std_range, std_lat), True, "foot")
        if person.height < 1.0:
            return None
        depth = self.fy * self.person_height_m / person.height      # khoảng cách dọc trục quang học
        uc, vc = person.center
        r = (uc - self.cx) / self.fx
        d = (vc - self.cy) / self.fy
        x = depth * (self._cos - d * self._sin)
        y = -depth * r
        if x <= 0:
            return None
        return Observation(person.track_id, x, y, max(0.3, 0.25 * depth), feet_visible, "height")
