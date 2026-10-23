"""Kiểu dữ liệu dùng chung giữa các module tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class TrackedPerson:
    """Một người do ByteTrack theo dõi ở một frame. bbox tính bằng pixel."""
    track_id: int
    bbox: Tuple[float, float, float, float]   # x1, y1, x2, y2
    conf: float

    @property
    def center(self) -> Tuple[float, float]:
        return (self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2

    @property
    def foot(self) -> Tuple[float, float]:
        """Điểm chân: giữa cạnh đáy bbox."""
        return (self.bbox[0] + self.bbox[2]) / 2, self.bbox[3]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass(frozen=True)
class Observation:
    """Một phép đo vị trí đưa vào bộ dự đoán. (x, y) cùng hệ toạ độ với kết quả dự đoán."""
    track_id: int
    x: float
    y: float
    std: float                 # độ lệch chuẩn của phép đo, cùng đơn vị với x, y
    feet_visible: bool = True
    method: str = "image"      # "image" | "foot" | "height" (xem camera_model.py)


@dataclass(frozen=True)
class PredictedPoint:
    t: float                   # giây kể từ thời điểm hiện tại
    x: float
    y: float
    sigma: float               # độ bất định vị trí (1 sigma), tăng dần theo t


@dataclass
class PersonTrajectory:
    track_id: int
    stamp: float               # thời điểm của trạng thái này (giây)
    hits: int                  # số phép đo đã nhận
    coasting: bool             # True = frame này không thấy người, đang ngoại suy
    x: float
    y: float
    vx: float
    vy: float
    feet_visible: bool = True
    method: str = "image"
    future: List[PredictedPoint] = field(default_factory=list)

    @property
    def speed(self) -> float:
        return float((self.vx ** 2 + self.vy ** 2) ** 0.5)

    def to_dict(self, ndigits: int = 3) -> dict:
        """Dạng dict thuần để đưa vào JSON."""
        import math
        r = lambda v: round(float(v), ndigits)  # noqa: E731
        return {
            "id": self.track_id,
            "hits": self.hits,
            "coasting": self.coasting,
            "x": r(self.x), "y": r(self.y),
            "vx": r(self.vx), "vy": r(self.vy),
            "speed": r(self.speed),
            "heading_deg": r(math.degrees(math.atan2(self.vy, self.vx))),
            "feet_visible": self.feet_visible,
            "method": self.method,
            "future": [{"t": r(p.t), "x": r(p.x), "y": r(p.y), "sigma": r(p.sigma)} for p in self.future],
        }
