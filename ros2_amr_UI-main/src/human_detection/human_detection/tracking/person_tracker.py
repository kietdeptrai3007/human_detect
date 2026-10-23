"""YOLO + ByteTrack (ultralytics): trả về những người kèm track_id ổn định giữa các frame."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .types import TrackedPerson


class PersonTracker:
    """
    weights     : trọng số YOLO của người (best.pt)
    tracker_cfg : đường dẫn yaml ByteTrack (config/bytetrack_person.yaml)
    conf        : ngưỡng conf KHI GỌI YOLO. Để thấp (~0.1) cho ByteTrack dùng lượt ghép thứ hai
                  với detection điểm thấp; lọc ngưỡng cao hơn là việc của nơi dùng kết quả.
    """

    def __init__(self, weights: str, tracker_cfg: str, device: Optional[str] = None,
                 conf: float = 0.1, imgsz: int = 640, person_class_id: int = 0):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.tracker_cfg = str(tracker_cfg)
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.classes = [person_class_id]

    def track(self, frame_bgr: np.ndarray) -> List[TrackedPerson]:
        kw = dict(persist=True, tracker=self.tracker_cfg, conf=self.conf, imgsz=self.imgsz,
                  classes=self.classes, verbose=False)
        if self.device is not None:
            kw["device"] = self.device
        result = self.model.track(frame_bgr, **kw)[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:       # chưa có track nào được xác nhận
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        return [TrackedPerson(int(i), (float(b[0]), float(b[1]), float(b[2]), float(b[3])), float(c))
                for i, b, c in zip(ids, xyxy, conf)]

    def reset(self) -> None:
        """Xoá bộ nhớ ID (đổi video, camera kết nối lại). ID bắt đầu lại từ 1."""
        predictor = getattr(self.model, "predictor", None)
        for t in getattr(predictor, "trackers", None) or []:
            t.reset()
