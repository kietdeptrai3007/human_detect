"""
command_filter.py - Chọn lệnh ổn định từ chuỗi dự đoán của DD-Net (lấy từ robot_control.py).

An toàn:
  - Lệnh chỉ đổi khi cùng một cử chỉ có xác suất >= conf trong `confirm` lần dự đoán liên tiếp.
  - Mất người / mất khung xương quá lost_sec -> về "None" (đứng yên).
  - Không có cử chỉ mới thì giữ lệnh tối đa hold_sec rồi về "None".
  - Dự đoán có xác suất < conf (mặc định 0.8) coi như "None".
  - Halt luôn được chấp nhận ngay (confirm = 1 cho Halt).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional


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
    "Left":          Motion(),          # chưa gắn chuyển động: sang trái/phải do Navigation của nhóm quyết định
    "Right":         Motion(),
    "None":          Motion(),          # không có lệnh: đứng yên
}
STOP = Motion()

# Số hiệu lệnh publish lên /gesture (std_msgs/Int32). BẢNG CỐ ĐỊNH, bên nhận dùng đúng bảng này;
# trùng thứ tự CLASS_NAMES lúc train (model 5 lớp, đã bỏ FollowMe). "None" (không có lệnh) là số cuối.
# Thêm cử chỉ mới thì thêm vào CUỐI model và bảng này.
GESTURE_IDS: Dict[str, int] = {
    "MoveForward": 0, "Halt": 1, "MoveInReverse": 2, "Left": 3, "Right": 4, "None": 5,
}
IDLE = "None"                           # lệnh khi không có cử chỉ


class CommandFilter:
    def __init__(self, conf: float, confirm: int, hold_sec: float, lost_sec: float):
        self.conf, self.confirm, self.hold_sec, self.lost_sec = conf, confirm, hold_sec, lost_sec
        self.current = IDLE
        self._cand, self._count = None, 0
        self._last_set = time.time()
        self._last_seen = time.time()

    def reset(self) -> None:
        """Về None và xoá bộ đếm xác nhận (khi đổi người chủ, tránh lệnh của người cũ còn sót lại)."""
        self.current = IDLE
        self._cand, self._count = None, 0
        self._last_set = self._last_seen = time.time()

    def step(self, label: Optional[str], prob: float, now: float) -> str:
        if label is None:                                     # chưa có dự đoán / mất người
            self._cand, self._count = None, 0
            if now - self._last_seen > self.lost_sec:
                self.current = IDLE
            return self.current
        self._last_seen = now
        if prob < self.conf:                                   # không đủ tin cậy -> None
            label, prob = IDLE, 1.0

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
