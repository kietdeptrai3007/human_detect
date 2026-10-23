"""
person_crop.py - Tìm người trong khung hình (YOLO), làm mượt khung bao, cắt ảnh để đưa vào pose.

Dùng CHUNG cho lúc trích dữ liệu (extract_keypoints.py) và lúc chạy trên robot, để hai nơi
cho ra ảnh crop giống nhau.

Luồng cho mỗi frame:
    detector.detect(frame)            -> các bbox người (x1, y1, x2, y2, conf)
    chọn MỘT người                    -> người gần bbox của frame trước nhất (hoặc to nhất)
    làm mượt bbox theo thời gian      -> crop không rung, keypoint không rung theo
    nới lề CROP_MARGIN                -> tay duỗi ra ngoài thân vẫn nằm trong crop
    cắt ảnh                           -> CropResult

Hợp đồng với extract_keypoints.py:
    PersonCropper(detector).reset()
    PersonCropper(detector).process(frame) -> None nếu mất người, ngược lại CropResult có
        .crop    ảnh BGR đã cắt
        .box     np.float32 [x1, y1, x2, y2] của vùng crop (đã nới lề), toạ độ ẢNH GỐC
        .offset  (x0, y0) góc trên trái của crop trong ảnh gốc; keypoint trong crop + offset
                 = keypoint trong ảnh gốc

Chạy thử trên một video:
    python person_crop.py --video đường/dẫn/video.mp4 --out preview.mp4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from . import config
from ..tracking.target_selector import NO_TARGET

Box = Tuple[float, float, float, float, float]   # x1, y1, x2, y2, conf


# ---------------------------------------------------------------------------
# Bộ phát hiện người
# ---------------------------------------------------------------------------
class YoloPersonDetector:
    """
    YOLO (ultralytics). Trả các bbox người, sắp xếp giảm dần theo diện tích.

    tracker_cfg: đường dẫn yaml ByteTrack. Có thì detect() chạy YOLO + ByteTrack (một lần), vẫn trả
    các Box như cũ (chỉ người conf >= conf), đồng thời ghi MỌI người kèm track_id vào last_tracks.
    Không có thì chạy YOLO thuần như trước và last_tracks luôn rỗng.
    """

    def __init__(self, weights: str = config.YOLO_WEIGHTS, device: Optional[str] = None,
                 conf: float = config.YOLO_CONF, imgsz: int = config.YOLO_IMGSZ,
                 person_class_id: int = 0, tracker_cfg: Optional[str] = None,
                 track_conf: float = config.TRACK_CONF):
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        self.classes = [person_class_id]
        self.last_tracks: list = []        # List[TrackedPerson] của frame gần nhất
        self.tracker = None
        self.model = None
        if tracker_cfg:
            from ..tracking.person_tracker import PersonTracker
            self.tracker = PersonTracker(weights, tracker_cfg, device=device, conf=track_conf,
                                         imgsz=imgsz, person_class_id=person_class_id)
        else:
            from ultralytics import YOLO
            self.model = YOLO(str(weights))

    def reset(self) -> None:
        """Xoá bộ nhớ ID của tracker (đổi video, camera kết nối lại)."""
        self.last_tracks = []
        if self.tracker is not None:
            self.tracker.reset()

    def detect(self, frame_bgr: np.ndarray) -> List[Box]:
        if self.tracker is not None:
            self.last_tracks = self.tracker.track(frame_bgr)
            boxes = [(*t.bbox, t.conf) for t in self.last_tracks if t.conf >= self.conf]
            boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
            return boxes
        kw = dict(conf=self.conf, imgsz=self.imgsz, classes=self.classes, verbose=False)
        if self.device is not None:
            kw["device"] = self.device
        result = self.model(frame_bgr, **kw)[0]
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        boxes = [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c))
                 for b, c in zip(xyxy, conf)]
        boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
        return boxes


class FullFrameDetector:
    """Không dùng YOLO: coi cả khung hình là 'người' (chạy nhanh, nhưng khác lúc chạy thật)."""

    last_tracks: list = []      # không có tracking

    def reset(self) -> None:
        pass

    def detect(self, frame_bgr: np.ndarray) -> List[Box]:
        h, w = frame_bgr.shape[:2]
        return [(0.0, 0.0, float(w), float(h), 1.0)]


# ---------------------------------------------------------------------------
# Kết quả
# ---------------------------------------------------------------------------
@dataclass
class CropResult:
    crop: np.ndarray        # ảnh BGR đã cắt
    box: np.ndarray         # float32 [x1, y1, x2, y2], vùng crop (đã nới lề) trong ảnh gốc
    offset: Tuple[int, int]  # (x0, y0): góc trên trái của crop trong ảnh gốc
    person_box: np.ndarray  # float32 [x1, y1, x2, y2], bbox người đã làm mượt (chưa nới lề)
    detected: bool          # True = YOLO thấy người ở frame này, False = dùng lại bbox cũ
    conf: float = 0.0       # độ tin cậy YOLO của người được chọn (0 khi dùng lại bbox cũ)
    track_id: Optional[int] = None   # track_id của người được chọn (None khi chọn theo IoU, không dùng tracker)


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Cắt ảnh
# ---------------------------------------------------------------------------
class PersonCropper:
    """
    Có trạng thái: nhớ bbox đã làm mượt của frame trước. Gọi reset() khi bắt đầu video mới.

    margin       : nới bbox thêm bấy nhiêu phần của chiều rộng/cao về MỖI phía
    smoothing    : EMA của bbox, 1.0 = không làm mượt, nhỏ hơn = mượt hơn
    max_missing  : YOLO trượt tối đa bấy nhiêu frame liền thì vẫn dùng bbox cũ
    new_iou      : bbox mới có IoU với bbox cũ thấp hơn mức này thì coi là một người/cảnh khác
                   -> không làm mượt, nhảy thẳng tới bbox mới
    """

    MIN_CROP_PX = 8

    def __init__(self, detector, margin: float = config.CROP_MARGIN,
                 smoothing: float = config.BBOX_SMOOTHING,
                 max_missing: int = config.MAX_MISSING_FRAMES, new_iou: float = 0.1):
        self.detector = detector
        self.margin = margin
        self.smoothing = smoothing
        self.max_missing = max_missing
        self.new_iou = new_iou
        self.reset()

    def reset(self) -> None:
        self._box: Optional[np.ndarray] = None   # bbox người đã làm mượt (4,)
        self._missing = 0
        self._conf = 0.0
        self._track_id: Optional[int] = None     # track_id của người đang theo (None = chọn theo IoU)

    def _choose(self, boxes: List[Box], target_id: Optional[int] = None) -> Optional[np.ndarray]:
        """
        target_id có giá trị: chọn ĐÚNG track đó, không có thì trả None (không chuyển sang người khác).
        target_id None: chọn người gần bbox trước nhất; chưa có bbox trước thì lấy người to nhất.
        """
        self._conf = 0.0
        if target_id is not None:
            for t in getattr(self.detector, "last_tracks", ()):
                if t.track_id != target_id:
                    continue
                for b in boxes:
                    if tuple(b[:4]) == tuple(t.bbox):
                        self._conf, self._track_id = b[4], target_id
                        return np.array(b[:4], np.float32)
            return None
        self._track_id = None
        if not boxes:
            return None
        if self._box is not None:
            best = max(boxes, key=lambda b: iou(self._box, b[:4]))
            if iou(self._box, best[:4]) >= self.new_iou:
                self._conf = best[4]
                return np.array(best[:4], np.float32)
        self._conf = boxes[0][4]
        return np.array(boxes[0][:4], np.float32)      # boxes đã sắp xếp theo diện tích

    def process(self, frame_bgr: np.ndarray, target_fn=None) -> Optional[CropResult]:
        """
        target_fn: tuỳ chọn, gọi với detector.last_tracks sau khi detect, trả về
            track_id  -> chỉ xử lý người đó
            NO_TARGET -> không xử lý ai (trả None ngay)
            None      -> không có ý kiến, chọn theo IoU như cũ
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        boxes = self.detector.detect(frame_bgr)
        target_id = target_fn(getattr(self.detector, "last_tracks", [])) if target_fn is not None else None
        if target_id == NO_TARGET:
            self.reset()
            return None
        det = self._choose(boxes, target_id)
        detected = det is not None
        if detected:
            self._missing = 0
            if self._box is None or iou(self._box, det) < self.new_iou:
                self._box = det                                           # người mới / cảnh mới
            else:
                a = self.smoothing
                self._box = (a * det + (1.0 - a) * self._box).astype(np.float32)
        else:
            self._missing += 1
            if self._box is None or self._missing > self.max_missing:
                self.reset()                                              # mất người thật sự
                return None

        return self._make_crop(frame_bgr, self._box, detected)

    def _make_crop(self, frame: np.ndarray, person: np.ndarray, detected: bool) -> Optional[CropResult]:
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = person
        mw, mh = self.margin * (x2 - x1), self.margin * (y2 - y1)
        x0 = int(max(0, np.floor(x1 - mw)))
        y0 = int(max(0, np.floor(y1 - mh)))
        x3 = int(min(W, np.ceil(x2 + mw)))
        y3 = int(min(H, np.ceil(y2 + mh)))
        if x3 - x0 < self.MIN_CROP_PX or y3 - y0 < self.MIN_CROP_PX:
            return None
        return CropResult(
            crop=frame[y0:y3, x0:x3],
            box=np.array([x0, y0, x3, y3], np.float32),
            offset=(x0, y0),
            person_box=person.astype(np.float32).copy(),
            detected=detected,
            conf=float(self._conf) if detected else 0.0,
            track_id=self._track_id,
        )


# ---------------------------------------------------------------------------
# Chạy thử: python person_crop.py --video x.mp4 --out preview.mp4
# ---------------------------------------------------------------------------
def _demo() -> None:
    import cv2

    ap = argparse.ArgumentParser(description="Xem thử bbox và vùng crop trên một video")
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="person_crop_preview.mp4")
    ap.add_argument("--yolo", default=config.YOLO_WEIGHTS)
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    detector = FullFrameDetector() if args.no_yolo else YoloPersonDetector(args.yolo, device=args.device)
    cropper = PersonCropper(detector)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = found = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        res = cropper.process(frame)
        if res is not None:
            found += 1
            px1, py1, px2, py2 = res.person_box.astype(int)
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0) if res.detected else (0, 165, 255), 2)
            cx1, cy1, cx2, cy2 = res.box.astype(int)
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 200, 0), 2)
        writer.write(frame)
    cap.release()
    writer.release()
    print(f"{found}/{n} frame có crop. Đã ghi {args.out} "
          f"(xanh lá = bbox người, cam = dùng lại bbox cũ, xanh dương = vùng crop)")


if __name__ == "__main__":
    _demo()
