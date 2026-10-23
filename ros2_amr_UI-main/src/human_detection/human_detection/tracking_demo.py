"""
tracking_demo.py - Chạy thử tracking + dự đoán quỹ đạo trên một video, không cần ROS.

    cd src/human_detection
    python -m human_detection.tracking_demo --video đường/dẫn/video.mp4 --out preview.mp4
    # mô phỏng CPU yếu chỉ xử lý 1/3 số frame:   --stride 3
    # dự đoán trên mặt đất (mét), cần đo camera:  --camera-model config/camera_model.yaml

Vẽ khung + ID + quỹ đạo dự đoán lên video, rồi in:
  - số ID đã dùng và thời gian sống của từng ID (video 1 người mà ra nhiều ID = tracking bị đứt)
  - sai số dự đoán: điểm dự đoán tại t+h so với vị trí đo được tại t+h, kèm mốc "đứng yên"
    (giả sử người không di chuyển). Dự đoán tốt phải thấp hơn mốc này khi người đang đi.
    Đơn vị: pixel (chế độ ảnh) hoặc mét (chế độ mặt đất).
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .tracking.camera_model import CameraModel
from .tracking.person_tracker import PersonTracker
from .tracking.trajectory_predictor import TrajectoryPredictor
from .tracking.types import Observation

PKG_DIR = Path(__file__).resolve().parents[1]
EVAL_HORIZONS = (0.5, 1.0, 2.0)


def _color(track_id: int):
    rng = np.random.RandomState(track_id * 9973 % (2 ** 31))
    return tuple(int(c) for c in rng.randint(60, 255, 3))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="tracking_preview.mp4")
    ap.add_argument("--yolo", default=str(PKG_DIR / "models" / "best.pt"))
    ap.add_argument("--tracker", default=str(PKG_DIR / "config" / "bytetrack_person.yaml"))
    ap.add_argument("--camera-model", default=None, help="yaml camera -> chế độ mặt đất (mét)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--conf", type=float, default=0.1, help="ngưỡng conf khi gọi YOLO (để thấp cho ByteTrack)")
    ap.add_argument("--stride", type=int, default=1, help="chỉ xử lý 1 trong N frame")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--horizon", type=float, default=2.0)
    ap.add_argument("--step", type=float, default=0.25)
    ap.add_argument("--min-hits", type=int, default=5)
    ap.add_argument("--pixel-std", type=float, default=6.0)
    ap.add_argument("--accel-std", type=float, default=None,
                    help="mặc định 50 (pixel/s^2) ở chế độ ảnh, 1.0 (m/s^2) ở chế độ mặt đất")
    args = ap.parse_args()

    cam = CameraModel.from_yaml(args.camera_model) if args.camera_model else None
    ground = cam is not None
    unit = "m" if ground else "px"
    accel = args.accel_std if args.accel_std is not None else (1.0 if ground else 50.0)

    tracker = PersonTracker(args.yolo, args.tracker, device=args.device, conf=args.conf)
    predictor = TrajectoryPredictor(horizon_sec=args.horizon, step_sec=args.step,
                                    min_hits=args.min_hits, accel_std=accel,
                                    init_vel_std=2.0 if ground else 200.0)

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if ground and (w, h) != (cam.image_width, cam.image_height):
        print(f"CẢNH BÁO: video {w}x{h} khác image_width/height trong camera model "
              f"({cam.image_width}x{cam.image_height}); toạ độ mặt đất sẽ sai.")
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps / args.stride, (w, h))

    obs_hist = defaultdict(list)     # id -> [(t, x, y)]
    preds = []                       # (id, t0, horizon, px, py, x0, y0)
    life = defaultdict(int)          # id -> số frame được thấy
    n = used = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if (n - 1) % args.stride:
            continue
        if args.max_frames and used >= args.max_frames:
            break
        used += 1
        t = (n - 1) / fps

        people = tracker.track(frame)
        observations = []
        for p in people:
            life[p.track_id] += 1
            if ground:
                ob = cam.locate(p)
            else:
                fx, fy = p.foot
                ob = Observation(p.track_id, fx, fy, args.pixel_std)
            if ob is not None:
                observations.append(ob)
                obs_hist[p.track_id].append((t, ob.x, ob.y))
        trajs = predictor.update(t, observations)

        for p in people:
            x1, y1, x2, y2 = (int(v) for v in p.bbox)
            col = _color(p.track_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, f"id{p.track_id} {p.conf:.2f}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        for tr in trajs:
            col = _color(tr.track_id)
            pts = [(tr.x, tr.y)] + [(f.x, f.y) for f in tr.future]
            if ground:
                pix = [cam.ground_to_pixel(*q) for q in pts]
                pix = [q for q in pix if q is not None]
            else:
                pix = pts
            if len(pix) >= 2:
                cv2.polylines(frame, [np.array(pix, np.int32).reshape(-1, 1, 2)], False, col, 2)
                cv2.circle(frame, (int(pix[-1][0]), int(pix[-1][1])), 4, col, -1)
            for f in tr.future:
                if abs(f.t - round(f.t / 0.5) * 0.5) < 1e-6:
                    preds.append((tr.track_id, t, f.t, f.x, f.y, tr.x, tr.y))
        cv2.putText(frame, f"frame {n}  people {len(people)}  traj {len(trajs)}  [{unit}]",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        writer.write(frame)

    cap.release()
    writer.release()

    print(f"\nĐã xử lý {used} frame (stride={args.stride}), ghi {args.out}")
    print(f"Số ID đã dùng: {len(life)}  (video chỉ có 1 người mà ra nhiều ID = tracking bị đứt)")
    for tid, cnt in sorted(life.items()):
        print(f"  id{tid}: {cnt} frame")

    tol = max(0.5 * args.stride / fps, 0.05)
    print(f"\nSai số dự đoán ({unit}), so với vị trí đo được tại t+h (dung sai {tol:.2f}s):")
    print(f"  {'h(s)':>5} {'mẫu':>6} {'dự đoán TB':>11} {'trung vị':>9} {'đứng yên TB':>12}")
    for hz in EVAL_HORIZONS:
        err, base = [], []
        for tid, t0, hh, px, py, x0, y0 in preds:
            if abs(hh - hz) > 1e-6:
                continue
            hist = obs_hist.get(tid)
            if not hist:
                continue
            tt = np.array([q[0] for q in hist])
            k = int(np.argmin(np.abs(tt - (t0 + hh))))
            if abs(tt[k] - (t0 + hh)) > tol:
                continue
            _, ox, oy = hist[k]
            err.append(math.hypot(px - ox, py - oy))
            base.append(math.hypot(x0 - ox, y0 - oy))
        if err:
            print(f"  {hz:>5} {len(err):>6} {np.mean(err):>11.2f} {np.median(err):>9.2f} {np.mean(base):>12.2f}")
        else:
            print(f"  {hz:>5} {0:>6}  (không đủ dữ liệu)")


if __name__ == "__main__":
    main()
