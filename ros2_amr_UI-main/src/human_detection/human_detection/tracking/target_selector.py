"""
Chọn và khoá "người chủ" (commander): người duy nhất mà nhánh cử chỉ nhận lệnh.

Máy trạng thái:
    IDLE      không có người nào trong khung
    SEARCHING chưa khoá ai: cử chỉ chạy cho ỨNG VIÊN (ban đầu là người có bbox to nhất = gần nhất);
              có từ 2 người trở lên thì cứ candidate_dwell_sec đổi ứng viên sang người kế tiếp, để người
              đứng xa hơn cũng có lượt được nhận diện (nếu không, người gần nhất không phải người điều
              khiển sẽ chặn mãi). Ứng viên làm cử chỉ `lock_gesture` (đã được CommandFilter xác nhận)
              thì được khoá
    LOCKED    chỉ người chủ (track_id đã khoá) được xử lý cử chỉ, người khác bị bỏ qua
    LOST      người chủ mất khỏi khung quá miss_tolerance_sec. Không xử lý cử chỉ của ai.
              Nếu trong lost_grace_sec xuất hiện track MỚI (id lớn hơn mọi id đã thấy trước lúc mất)
              có vị trí và kích thước gần chỗ cũ thì coi là cùng người (ByteTrack đổi ID) và kế thừa
              khoá; hết thời gian thì về SEARCHING/IDLE.

Python thuần, không import ROS. Thời gian `now` do nơi gọi truyền vào (giây).
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from .types import TrackedPerson

NO_TARGET = -1     # update() trả về giá trị này = "đừng xử lý cử chỉ của ai ở frame này"

IDLE, SEARCHING, LOCKED, LOST = "IDLE", "SEARCHING", "LOCKED", "LOST"


class TargetSelector:
    def __init__(self, lock_gesture: str = "Right", lost_grace_sec: float = 1.5,
                 miss_tolerance_sec: float = 0.3, reassoc_dx_ratio: float = 1.5,
                 reassoc_size_ratio: float = 1.6, min_conf: float = 0.4,
                 candidate_dwell_sec: float = 5.0):
        self.lock_gesture = lock_gesture
        self.lost_grace_sec = lost_grace_sec
        self.miss_tolerance_sec = miss_tolerance_sec
        self.reassoc_dx_ratio = reassoc_dx_ratio
        self.reassoc_size_ratio = reassoc_size_ratio
        self.candidate_dwell_sec = candidate_dwell_sec   # 0 = không đổi ứng viên; nên >= cửa sổ cử chỉ (3,2 s) + dư
        self.min_conf = min_conf     # chỉ track có conf >= mức này mới được xét (khớp YOLO_CONF của nhánh cử chỉ)
        self.reset()

    def reset(self) -> None:
        self.state = IDLE
        self.locked_id: Optional[int] = None
        self.candidate_id: Optional[int] = None
        self._cand_since = 0.0
        self._last_bbox = None                    # bbox cuối cùng của người chủ
        self._last_seen = 0.0
        self._missing_since: Optional[float] = None
        self._lost_since = 0.0
        self._max_id = 0                          # id lớn nhất đã thấy
        self._id_floor = 0                        # id lớn nhất đã thấy TRƯỚC khi người chủ mất

    @property
    def commander_id(self) -> Optional[int]:
        """track_id của người chủ khi đang LOCKED, ngược lại None."""
        return self.locked_id if self.state == LOCKED else None

    # ------------------------------------------------------------------
    def update(self, tracks: Sequence[TrackedPerson], now: float) -> Optional[int]:
        """
        Gọi mỗi frame ngay sau khi tracker chạy. Trả về:
            track_id : chạy cử chỉ cho người này
            NO_TARGET: không chạy cử chỉ cho ai (LOST)
            None     : không có ý kiến (không còn ai trong khung)
        """
        people: List[TrackedPerson] = [t for t in tracks if t.conf >= self.min_conf]
        ids = {t.track_id for t in people}
        prev_max = self._max_id
        if tracks:
            self._max_id = max(self._max_id, max(t.track_id for t in tracks))

        if self.state in (LOCKED, LOST):
            return self._update_locked(people, ids, prev_max, now)

        # IDLE / SEARCHING: chọn ứng viên, giữ ứng viên cũ nếu còn
        if not people:
            self.state, self.candidate_id = IDLE, None
            return None
        by_size = sorted(people, key=lambda t: t.width * t.height, reverse=True)
        if self.candidate_id not in ids:
            self.candidate_id, self._cand_since = by_size[0].track_id, now
        elif (self.candidate_dwell_sec > 0 and len(by_size) > 1
              and now - self._cand_since >= self.candidate_dwell_sec):
            order = [t.track_id for t in by_size]
            self.candidate_id = order[(order.index(self.candidate_id) + 1) % len(order)]
            self._cand_since = now
        self.state = SEARCHING
        return self.candidate_id

    def _update_locked(self, people, ids, prev_max, now) -> Optional[int]:
        if self.locked_id in ids:                              # kể cả LOST: ByteTrack tìm lại đúng ID cũ
            me = next(t for t in people if t.track_id == self.locked_id)
            self.state = LOCKED
            self._last_bbox, self._last_seen, self._missing_since = me.bbox, now, None
            return self.locked_id

        if self._missing_since is None:                        # vừa mất người chủ
            self._missing_since = now
            self._id_floor = prev_max
            self._lost_since = now

        match = self._reassociate(people)
        if match is not None:
            self.locked_id, self.state, self._missing_since = match.track_id, LOCKED, None
            self._last_bbox, self._last_seen = match.bbox, now
            return self.locked_id

        if self.state == LOCKED:
            if now - self._missing_since <= self.miss_tolerance_sec:
                return self.locked_id                           # có thể chỉ trượt vài frame
            self.state = LOST
        if now - self._lost_since > self.lost_grace_sec:       # hết thời gian chờ -> bỏ khoá
            self.locked_id, self._missing_since, self._last_bbox = None, None, None
            self.candidate_id = None
            self.state = SEARCHING if people else IDLE
            return self.update(people, now) if people else None
        return NO_TARGET

    def _reassociate(self, people) -> Optional[TrackedPerson]:
        """Track MỚI (id > mọi id đã thấy trước lúc mất) gần vị trí và kích thước cũ nhất."""
        if self._last_bbox is None:
            return None
        lx1, ly1, lx2, ly2 = self._last_bbox
        lw, lh = max(lx2 - lx1, 1.0), max(ly2 - ly1, 1.0)
        lcx = (lx1 + lx2) / 2
        best, best_score = None, None
        for t in people:
            if t.track_id <= self._id_floor:
                continue
            dx = abs((t.bbox[0] + t.bbox[2]) / 2 - lcx) / lw
            ratio = max(t.height, 1.0) / lh
            if dx > self.reassoc_dx_ratio or not (1 / self.reassoc_size_ratio <= ratio <= self.reassoc_size_ratio):
                continue
            score = dx + abs(math.log(ratio))
            if best_score is None or score < best_score:
                best, best_score = t, score
        return best

    # ------------------------------------------------------------------
    def observe_command(self, cmd: str, now: float) -> None:
        """Báo lệnh đã lọc ổn định của frame trước. Ứng viên làm lock_gesture -> khoá."""
        if self.state == SEARCHING and cmd == self.lock_gesture and self.candidate_id is not None:
            self.locked_id, self.state = self.candidate_id, LOCKED
            self._missing_since, self._last_seen = None, now
