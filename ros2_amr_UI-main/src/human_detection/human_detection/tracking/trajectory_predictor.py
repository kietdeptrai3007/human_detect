"""Dự đoán quỹ đạo nhiều người: mỗi track_id có một bộ lọc Kalman vận tốc không đổi."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .kalman_cv import ConstantVelocityKalman
from .types import Observation, PersonTrajectory, PredictedPoint


@dataclass
class _Track:
    kf: ConstantVelocityKalman
    last_stamp: float          # thời điểm bộ lọc đang ở
    last_seen: float           # lần cuối có phép đo
    hits: int = 1
    feet_visible: bool = True
    method: str = "image"


class TrajectoryPredictor:
    """
    horizon_sec : dự đoán xa bao nhiêu giây
    step_sec    : khoảng cách giữa các điểm dự đoán
    min_hits    : số phép đo tối thiểu trước khi xuất quỹ đạo (vận tốc ít phép đo thì rất nhiễu)
    lost_sec    : track không có phép đo quá lâu thì xoá (trong lúc đó vẫn ngoại suy)
    accel_std   : độ lệch chuẩn gia tốc (đơn vị toạ độ / s^2); lớn = tin vào vận tốc cũ ít hơn
    init_vel_std: bất định vận tốc ban đầu
    max_dt      : khoảng cách thời gian giữa 2 lần update lớn hơn mức này thì xoá hết (mất luồng ảnh)
    """

    def __init__(self, horizon_sec: float = 2.0, step_sec: float = 0.25, min_hits: int = 5,
                 lost_sec: float = 1.0, accel_std: float = 1.0, init_vel_std: float = 2.0,
                 max_dt: float = 2.0):
        self.horizon_sec = horizon_sec
        self.step_sec = step_sec
        self.min_hits = min_hits
        self.lost_sec = lost_sec
        self.accel_std = accel_std
        self.init_vel_std = init_vel_std
        self.max_dt = max_dt
        self._tracks: Dict[int, _Track] = {}
        self._now = None

    def reset(self) -> None:
        self._tracks.clear()
        self._now = None

    def update(self, stamp: float, observations: List[Observation]) -> List[PersonTrajectory]:
        """Gọi mỗi frame với MỌI người đang thấy (danh sách rỗng cũng phải gọi để track tự hết hạn)."""
        if self._now is not None and stamp - self._now > self.max_dt:
            self._tracks.clear()
        self._now = stamp

        seen = set()
        for ob in observations:
            seen.add(ob.track_id)
            tr = self._tracks.get(ob.track_id)
            if tr is None:
                kf = ConstantVelocityKalman(ob.x, ob.y, ob.std, self.init_vel_std, self.accel_std)
                self._tracks[ob.track_id] = _Track(kf, stamp, stamp, 1, ob.feet_visible, ob.method)
                continue
            tr.kf.predict(stamp - tr.last_stamp)
            tr.kf.update(ob.x, ob.y, ob.std)
            tr.last_stamp = stamp
            tr.last_seen = stamp
            tr.hits += 1
            tr.feet_visible = ob.feet_visible
            tr.method = ob.method

        out: List[PersonTrajectory] = []
        for tid in list(self._tracks):
            tr = self._tracks[tid]
            if stamp - tr.last_seen > self.lost_sec:
                del self._tracks[tid]
                continue
            if tid not in seen:
                tr.kf.predict(stamp - tr.last_stamp)       # ngoại suy qua frame mất người
                tr.last_stamp = stamp
            if tr.hits < self.min_hits:
                continue
            out.append(self._snapshot(tid, tr, stamp, coasting=tid not in seen))
        return out

    def _snapshot(self, tid: int, tr: _Track, stamp: float, coasting: bool) -> PersonTrajectory:
        x, y, vx, vy = (float(v) for v in tr.kf.x)
        future: List[PredictedPoint] = []
        n = int(round(self.horizon_sec / self.step_sec))
        for k in range(1, n + 1):
            t = k * self.step_sec
            s, sigma = tr.kf.predict_ahead(t)
            future.append(PredictedPoint(t=t, x=float(s[0]), y=float(s[1]), sigma=sigma))
        return PersonTrajectory(track_id=tid, stamp=stamp, hits=tr.hits, coasting=coasting,
                                x=x, y=y, vx=vx, vy=vy, feet_visible=tr.feet_visible,
                                method=tr.method, future=future)
