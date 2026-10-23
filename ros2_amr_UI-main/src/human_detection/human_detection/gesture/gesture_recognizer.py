"""
gesture_recognizer.py - Nhận diện cử chỉ realtime bằng DD-Net đã train.

Dùng ĐÚNG các khối như lúc trích dữ liệu: PersonCropper (YOLO) -> PoseEstimator -> SkeletonAdapter,
rồi resample theo thời gian thật về fps của checkpoint và lấy cửa sổ T frame gần nhất.

    recognizer = GestureRecognizer("checkpoints/scratch_f16/best.pt")
    result = recognizer.update(frame_bgr, time.time())   # gọi mỗi frame camera
    result.label, result.prob, result.probs
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch

from . import config
from .ddnet_model import load_checkpoint
from .person_crop import FullFrameDetector, PersonCropper, YoloPersonDetector
from .skeleton_adapter import SkeletonAdapter


@dataclass
class GestureResult:
    label: Optional[str]          # None = chưa đủ dữ liệu / không thấy người
    prob: float                   # xác suất (đã làm mượt) của label
    probs: Optional[np.ndarray]   # (C,) xác suất đã làm mượt
    person: bool                  # có người trong khung hình không
    reason: str = ""


class GestureRecognizer:
    def __init__(self, ckpt_path: str, yolo_weights: Optional[str] = config.YOLO_WEIGHTS,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 min_ok_ratio: float = 0.6, max_gap_sec: float = 0.6, ema: float = 0.5,
                 pose_model_path: Optional[str] = None, tracker_cfg: Optional[str] = None):
        self.device = torch.device(device)
        self.model, self.meta = load_checkpoint(ckpt_path)
        self.model.to(self.device).eval()
        self.class_names: List[str] = self.meta["class_names"]
        self.T = int(self.meta["T"])
        self.fps = float(self.meta["fps"])
        self.window_sec = (self.T - 1) / self.fps
        self.min_ok_ratio = min_ok_ratio
        self.max_gap_sec = max_gap_sec
        self.ema = ema

        detector = FullFrameDetector() if yolo_weights is None else \
            YoloPersonDetector(yolo_weights, device=None if device == "cpu" else "0",
                               tracker_cfg=tracker_cfg)
        self.cropper = PersonCropper(detector)
        from .pose_estimate import PoseEstimator
        self.pose = PoseEstimator(model_complexity=0 if "lite" in str(self.meta.get("pose_model", "lite")) else 1,
                                  model_path=pose_model_path)
        self.adapter = SkeletonAdapter()
        self.last_crop = None    # CropResult của frame gần nhất (để vẽ ảnh debug), None = không thấy người

        self._buf: Deque[Tuple[float, np.ndarray]] = deque()   # (t, joints) chỉ frame ok
        self._frames: Deque[Tuple[float, bool]] = deque()      # (t, ok) mọi frame, để tính ok_ratio
        self._probs: Optional[np.ndarray] = None
        self._active_id: Optional[int] = None   # track_id người cuối cùng được xử lý (không xoá khi mất người)
        self.window_epoch = 0                   # tăng mỗi lần cửa sổ bị xoá vì đổi người (để node reset bộ lọc lệnh)

    @property
    def last_tracks(self) -> list:
        """Mọi người (TrackedPerson, có track_id) ở frame gần nhất; rỗng nếu không bật tracker."""
        return getattr(self.cropper.detector, "last_tracks", [])

    def reset(self) -> None:
        self.cropper.reset()
        self.cropper.detector.reset()     # xoá cả bộ nhớ ID của tracker
        self.adapter.reset()
        self._buf.clear()
        self._frames.clear()
        self._probs = None
        self._active_id = None

    def _new_person(self) -> None:
        """Đổi sang người khác: bỏ khung xương, xác suất và thước đo của người cũ."""
        self.adapter.reset()
        self._buf.clear()
        self._frames.clear()
        self._probs = None
        self.window_epoch += 1

    def close(self) -> None:
        """Giải phóng MediaPipe (tránh lỗi PoseLandmarker.__del__ khi thoát chương trình)."""
        self.pose.close()

    # ------------------------------------------------------------------
    def _process_frame(self, frame_bgr: np.ndarray, target_fn=None) -> Tuple[bool, Optional[np.ndarray]]:
        crop = self.cropper.process(frame_bgr, target_fn)
        self.last_crop = crop
        if crop is None or crop.crop.size == 0:
            self.adapter.reset()          # mất người -> quên vị trí/thước đo cũ (giống lúc trích dữ liệu)
            return False, None
        if crop.track_id is not None:
            if self._active_id is not None and crop.track_id != self._active_id:
                self._new_person()        # cửa sổ 32 frame không được trộn khung xương của hai người
            self._active_id = crop.track_id
        kps = self.pose.get_keypoints(crop.crop)
        if kps is None:
            return True, None
        full = [(x + crop.offset[0], y + crop.offset[1], v) for x, y, v in kps]
        res = self.adapter.convert(full)
        return True, (None if res is None else np.asarray(res.joints, np.float32))

    def _make_window(self, now: float) -> Tuple[Optional[np.ndarray], str]:
        """Resample các frame ok trong cửa sổ gần nhất về lưới đều `fps` (giống resample_by_time)."""
        t0 = now - self.window_sec
        while self._buf and self._buf[0][0] < t0 - 1.0:
            self._buf.popleft()
        while self._frames and self._frames[0][0] < t0:
            self._frames.popleft()
        if len(self._buf) < 2 or self._buf[0][0] > t0 + self.max_gap_sec:
            return None, "đang thu thập frame"
        in_win = [ok for _, ok in self._frames]
        if not in_win or np.mean(in_win) < self.min_ok_ratio:
            return None, "quá ít frame dùng được"
        if now - self._buf[-1][0] > self.max_gap_sec:
            return None, "mất khung xương"
        t = np.array([b[0] for b in self._buf])
        j = np.stack([b[1] for b in self._buf]).reshape(len(t), -1)
        grid = t0 + np.arange(self.T) / self.fps
        out = np.stack([np.interp(grid, t, j[:, c]) for c in range(j.shape[1])], 1)
        return out.reshape(self.T, -1, 2).astype(np.float32), ""

    # ------------------------------------------------------------------
    @torch.no_grad()
    def update(self, frame_bgr: np.ndarray, t: float, target_fn=None) -> GestureResult:
        """target_fn: xem PersonCropper.process (chọn người theo track_id); None = chọn theo IoU như cũ."""
        person, joints = self._process_frame(frame_bgr, target_fn)
        self._frames.append((t, joints is not None))
        if joints is not None:
            self._buf.append((t, joints))

        window, reason = self._make_window(t)
        if window is None:
            self._probs = None
            return GestureResult(None, 0.0, None, person, reason)

        x = torch.from_numpy(window).unsqueeze(0).to(self.device)
        p = torch.softmax(self.model(x), dim=1)[0].cpu().numpy()
        self._probs = p if self._probs is None else self.ema * p + (1 - self.ema) * self._probs
        k = int(self._probs.argmax())
        return GestureResult(self.class_names[k], float(self._probs[k]), self._probs.copy(), person)
