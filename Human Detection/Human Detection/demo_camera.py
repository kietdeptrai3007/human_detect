"""
demo_camera.py - Thử nhận diện cử chỉ từ camera (không điều khiển robot).

Đứng trước camera, làm một trong 5 cử chỉ (MoveForward, Halt, MoveInReverse, Left, Right); chương trình in cử chỉ đoán được
và xác suất các lớp. Cần ~3 giây đầu để thu đủ khung xương cho một cửa sổ.

    python demo_camera.py --ckpt checkpoints/scratch_f16/best.pt
    python demo_camera.py --ckpt checkpoints/scratch_f16/best.pt --camera realsense   # D435i
    python demo_camera.py --ckpt checkpoints/scratch_f16/best.pt --camera 1 --no-show
    python demo_camera.py --ckpt ... --camera đường/dẫn/video.mp4      # thử trên video

Nhấn q (trong cửa sổ camera) hoặc Ctrl+C để thoát.
"""

from __future__ import annotations

import argparse
import time

import cv2

import config
from camera_source import open_source
from gesture_recognizer import GestureRecognizer


def draw_panel(frame, res, names) -> None:
    h = 30 + 24 * len(names)
    cv2.rectangle(frame, (0, 0), (300, h), (0, 0, 0), -1)
    title = f"{res.label} ({res.prob * 100:.0f}%)" if res.label else res.reason or "..."
    cv2.putText(frame, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for i, n in enumerate(names):
        p = float(res.probs[i]) if res.probs is not None else 0.0
        y = 30 + 24 * i + 16
        best = res.label == n
        cv2.putText(frame, n, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if best else (200, 200, 200), 1)
        cv2.rectangle(frame, (130, y - 11), (130 + int(160 * p), y + 2),
                      (0, 255, 0) if best else (120, 120, 120), -1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Thử nhận diện cử chỉ qua camera")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--camera", default="0", help="realsense | chỉ số webcam | đường dẫn video")
    ap.add_argument("--no-yolo", action="store_true", help="bỏ YOLO (nhanh hơn, khác lúc train)")
    ap.add_argument("--no-show", action="store_true", help="không mở cửa sổ, chỉ in ra terminal")
    ap.add_argument("--print-every", type=float, default=0.5, help="in ra terminal mỗi bấy nhiêu giây")
    args = ap.parse_args()

    rec = GestureRecognizer(args.ckpt, yolo_weights=None if args.no_yolo else config.YOLO_WEIGHTS)
    names = rec.class_names
    print(f"Các cử chỉ: {', '.join(names)}")

    cap, is_file = open_source(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Không mở được camera/video {args.camera}")
    file_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    t_print, n = 0.0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # video file: dùng thời gian của video để kết quả giống lúc train; camera: giờ thật
            now = n / file_fps if is_file else time.time()
            n += 1
            res = rec.update(frame, now)

            if now - t_print >= args.print_every:
                t_print = now
                if res.label:
                    top = "  ".join(f"{names[i]}={res.probs[i] * 100:3.0f}%"
                                    for i in res.probs.argsort()[::-1][:3])
                    print(f"=> {res.label:<14} {res.prob * 100:3.0f}%   [{top}]", flush=True)
                else:
                    print(f"..  {res.reason}" + ("" if res.person else " (không thấy người)"), flush=True)

            if not args.no_show:
                draw_panel(frame, res, names)
                cv2.imshow("DD-Net gesture demo", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
