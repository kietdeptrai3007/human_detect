"""
Công cụ thu dataset cử chỉ: mỗi lần bấm phím là một clip ngắn có sẵn nhãn.
Đọc camera RealSense TRỰC TIẾP (pyrealsense2), không cần ROS, không cần camera node / topic.

    python record_node.py
    python record_node.py --subject S11 --out-dir /home/user/gesture_data --manual

Cần: pip install pyrealsense2, camera cắm cổng USB 3. Cửa sổ nhỏ hiện ảnh camera + trạng thái (FPS, đang đếm
ngược / đang quay). Cần màn hình (hoặc ssh -X); không chạy được khi không có DISPLAY.
Tham số: --out-dir --subject --distances --countdown --duration --manual --width --height --fps --serial.

Phím (bấm trong cửa sổ):
    1 2 3  MoveForward Halt MoveInReverse
    l r  Left Right
    p / o  người kế / người trước (S11 -> S12)         d  đổi khoảng cách (vòng qua danh sách `distances`)
    x      huỷ clip đang quay                           u  xoá clip vừa lưu (kể cả dòng annotation)
    SPACE  đang đếm ngược: tạm dừng (người chưa vào kịp), bấm lại để đếm ngược lại từ đầu
           chế độ thủ công: bắt đầu quay / dừng và lưu (clip < 3,5 s thì bỏ, không lưu file hỏng)
           chế độ tự động đang quay: SPACE không làm gì, muốn bỏ clip thì bấm x
    m      đổi chế độ TỰ ĐỘNG (đếm ngược + tự dừng) <-> THỦ CÔNG (chọn nhãn, SPACE bắt đầu, SPACE dừng)
    q      thoát

Bấm phím nhãn -> đếm ngược `countdown` giây -> tự quay `duration` giây -> tự lưu.
Kết quả (cùng định dạng RoCoG, thư mục con trùng tên lớp):
    <out_dir>/real/robot/<Lớp>/S11_5m_Left_take01.mp4
    <out_dir>/annotations/robot_all.txt      dòng: real/robot/Left/S11_5m_Left_take01.mp4 3
Số nhãn: MoveForward=0 Halt=1 MoveInReverse=2 Left=3 Right=4 (code train phải dùng đúng thứ tự này).
"""

import re
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Thứ tự = số nhãn.
CLASS_NAMES = ['MoveForward', 'Halt', 'MoveInReverse', 'Left', 'Right']
KEY_TO_CLASS = {'1': 'MoveForward', '2': 'Halt', '3': 'MoveInReverse',
                'l': 'Left', 'r': 'Right'}

IDLE, COUNTDOWN, RECORDING, READY, PAUSED = 'IDLE', 'COUNTDOWN', 'RECORDING', 'READY', 'PAUSED'
MIN_CLIP_SECS = 3.5      # clip thủ công ngắn hơn mức này thì bỏ (cửa sổ train ~3,2 s nên clip ngắn vô dụng)
MANUAL_MAX_SECS = 30.0   # chế độ thủ công quên bấm dừng thì tự dừng sau chừng này giây


class SessionRecorder:
    """Máy trạng thái thu clip + ghi file. Không phụ thuộc ROS / GUI nên test được riêng."""

    def __init__(self, out_dir: Path, subject: str, distances: List[float],
                 countdown: float, duration: float, min_fps: float = 8.0, manual: bool = False):
        self.out_dir = Path(out_dir)
        self.subject = subject
        self.distances = list(distances) or [5.0]
        self.dist_idx = 0
        self.countdown = countdown
        self.duration = duration
        self.manual = manual                     # True: chọn nhãn xong chờ SPACE mới quay, SPACE lần nữa để dừng
        self.manual_rec = False                  # clip đang quay là clip thủ công (không tự dừng theo duration)
        self.min_fps = min_fps

        self.state = IDLE
        self.label_name = ''
        self.t_start = 0.0                       # lúc bắt đầu đếm ngược
        self.frames: List[Tuple[float, np.ndarray]] = []
        self.message = ''
        self.message_until = 0.0
        self.last_saved: Optional[Tuple[Path, str]] = None   # (file video, dòng annotation)
        self.saving = 0                          # số clip đang ghi nền
        self._lock = threading.Lock()
        self.ann_path = self.out_dir / 'annotations' / 'robot_all.txt'

    # ---- thông tin ----
    @property
    def distance(self) -> float:
        return self.distances[self.dist_idx]

    @property
    def dist_str(self) -> str:
        d = self.distance
        return f'{int(d)}m' if float(d).is_integer() else f'{d:g}m'

    def count_saved(self) -> int:
        d = self.out_dir / 'real' / 'robot'
        return sum(1 for _ in d.rglob('*.mp4')) if d.exists() else 0

    def next_take(self, cls: str) -> int:
        d = self.out_dir / 'real' / 'robot' / cls
        prefix = f'{self.subject}_{self.dist_str}_{cls}_take'
        nums = [int(m.group(1)) for p in d.glob(prefix + '*.mp4')
                if (m := re.search(r'_take(\d+)\.mp4$', p.name))] if d.exists() else []
        return max(nums, default=0) + 1

    def say(self, text: str, secs: float = 3.0):
        self.message, self.message_until = text, time.monotonic() + secs

    # ---- đổi người / khoảng cách ----
    def change_subject(self, step: int):
        if self.state != IDLE:
            return
        m = re.fullmatch(r'S(\d+)', self.subject)
        n = max(0, (int(m.group(1)) if m else 0) + step)
        self.subject = f'S{n:02d}'

    def next_distance(self):
        if self.state == IDLE:
            self.dist_idx = (self.dist_idx + 1) % len(self.distances)

    # ---- vòng đời một clip ----
    def start(self, cls: str, now: float):
        if self.state != IDLE:
            return
        self.label_name, self.t_start, self.frames = cls, now, []
        self.state = READY if self.manual else COUNTDOWN

    def toggle_manual(self):
        if self.state == IDLE:
            self.manual = not self.manual
            self.say('Che do THU CONG: chon nhan roi bam SPACE de quay/dung' if self.manual
                     else 'Che do TU DONG: dem nguoc roi tu quay', 3.0)

    def space(self, now: float):
        """
        SPACE theo trạng thái:
          COUNTDOWN  -> PAUSED (tạm dừng, người chưa vào kịp)
          PAUSED     -> đếm ngược lại từ đầu
          READY      -> bắt đầu quay (chế độ thủ công)
          RECORDING  -> dừng và lưu, CHỈ ở chế độ thủ công; chế độ tự động muốn bỏ clip thì bấm x
        """
        if self.state == COUNTDOWN:
            self.state = PAUSED
        elif self.state == PAUSED:
            self.state, self.t_start = COUNTDOWN, now
        elif self.state == READY:
            self.state, self.t_start, self.frames, self.manual_rec = RECORDING, now, [], True
        elif self.state == RECORDING and self.manual_rec:
            if now - self.t_start < MIN_CLIP_SECS:
                self.state, self.frames, self.manual_rec = IDLE, [], False
                self.say(f'Clip ngan hon {MIN_CLIP_SECS:.1f}s, khong luu', 4.0)
            else:
                self._finish()

    def cancel(self):
        if self.state != IDLE:
            self.state, self.frames = IDLE, []
            self.say('Da huy clip', 2.0)

    def tick(self, now: float):
        """Gọi đều đặn từ vòng lặp GUI: chuyển COUNTDOWN -> RECORDING -> lưu."""
        if self.state == COUNTDOWN and now - self.t_start >= self.countdown:
            self.state, self.t_start, self.frames, self.manual_rec = RECORDING, now, [], False
        elif self.state == RECORDING and now - self.t_start >= (
                MANUAL_MAX_SECS if self.manual_rec else self.duration):
            self._finish()

    def _finish(self):
        frames, cls = self.frames, self.label_name
        self.state, self.frames, self.manual_rec = IDLE, [], False
        self.saving += 1
        threading.Thread(target=self._save, args=(frames, cls, self.subject, self.dist_str),
                         daemon=True).start()

    def on_frame(self, t: float, frame: np.ndarray):
        if self.state == RECORDING:
            self.frames.append((t, frame))

    def time_left(self, now: float) -> float:
        if self.state == COUNTDOWN:
            return max(0.0, self.countdown - (now - self.t_start))
        if self.state == RECORDING:
            return max(0.0, self.duration - (now - self.t_start))
        return 0.0

    def elapsed(self, now: float) -> float:
        return now - self.t_start

    def _save(self, frames, cls: str, subject: str, dist: str):
        try:
            if len(frames) < 10:
                self.say(f'Khong luu: chi nhan {len(frames)} frame (camera dung?)', 5.0)
                return
            span = frames[-1][0] - frames[0][0]
            fps = (len(frames) - 1) / span if span > 0 else 30.0
            with self._lock:
                prefix = f'{subject}_{dist}_{cls}_take'
                d = self.out_dir / 'real' / 'robot' / cls
                d.mkdir(parents=True, exist_ok=True)
                nums = [int(m.group(1)) for p in d.glob(prefix + '*.mp4')
                        if (m := re.search(r'_take(\d+)\.mp4$', p.name))]
                path = d / f'{prefix}{max(nums, default=0) + 1:02d}.mp4'
                h, w = frames[0][1].shape[:2]
                # FPS ghi vào file = FPS thật đo được, vì bước trích keypoint resample theo FPS của file
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                if not writer.isOpened():
                    self.say('Khong mo duoc VideoWriter', 5.0)
                    return
                for _, f in frames:
                    writer.write(f)
                writer.release()
                line = f'real/robot/{cls}/{path.name} {CLASS_NAMES.index(cls)}'
                self.ann_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.ann_path, 'a') as fh:
                    fh.write(line + '\n')
                self.last_saved = (path, line)
            warn = f'  (FPS thap {fps:.1f}!)' if fps < self.min_fps else ''
            self.say(f'Da luu {path.name}  {len(frames)} frame @ {fps:.1f} FPS{warn}', 4.0)
        finally:
            self.saving -= 1

    def undo(self):
        if self.state != IDLE or self.saving or not self.last_saved:
            self.say('Khong co clip de xoa', 2.0)
            return
        path, line = self.last_saved
        with self._lock:
            path.unlink(missing_ok=True)
            if self.ann_path.exists():
                kept = [ln for ln in self.ann_path.read_text().splitlines() if ln != line]
                self.ann_path.write_text('\n'.join(kept) + ('\n' if kept else ''))
        self.last_saved = None
        self.say(f'Da xoa {path.name}', 3.0)


# ---------------------------------------------------------------------------
# Vẽ cửa sổ
# ---------------------------------------------------------------------------
GREEN, RED, YELLOW, WHITE = (60, 200, 60), (50, 50, 230), (0, 200, 255), (235, 235, 235)


def _put(img, text, org, color=WHITE, scale=0.55, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_ui(frame: Optional[np.ndarray], cam_status: str, cam_color, rec: SessionRecorder,
            now: float) -> np.ndarray:
    w, h = (frame.shape[1], frame.shape[0]) if frame is not None else (640, 360)
    view = frame.copy() if frame is not None else np.zeros((h, w, 3), np.uint8)

    if rec.state == COUNTDOWN:
        cv2.rectangle(view, (0, 0), (w - 1, h - 1), YELLOW, 6)
        _put(view, str(int(np.ceil(rec.time_left(now)))), (w // 2 - 30, h // 2 + 40), YELLOW, 4.0, 8)
        _put(view, f'Chuan bi: {rec.label_name}', (w // 2 - 120, 40), YELLOW, 0.9, 2)
    elif rec.state == PAUSED:
        cv2.rectangle(view, (0, 0), (w - 1, h - 1), YELLOW, 6)
        _put(view, f'TAM DUNG: {rec.label_name}', (w // 2 - 150, h // 2 - 10), YELLOW, 1.0, 2)
        _put(view, 'Bam SPACE de dem nguoc lai', (w // 2 - 160, h // 2 + 30), YELLOW, 0.8, 2)
    elif rec.state == READY:
        cv2.rectangle(view, (0, 0), (w - 1, h - 1), YELLOW, 6)
        _put(view, f'San sang: {rec.label_name}', (w // 2 - 130, h // 2 - 10), YELLOW, 1.0, 2)
        _put(view, 'Bam SPACE de bat dau quay', (w // 2 - 150, h // 2 + 30), YELLOW, 0.8, 2)
    elif rec.state == RECORDING:
        cv2.rectangle(view, (0, 0), (w - 1, h - 1), RED, 6)
        cv2.circle(view, (30, 30), 12, RED, -1)
        if rec.manual_rec:
            _put(view, f'REC {rec.label_name}  {rec.elapsed(now):.1f}s  ({len(rec.frames)} frame)  SPACE = dung',
                 (55, 38), RED, 0.7, 2)
        else:
            _put(view, f'REC {rec.label_name}  con {rec.time_left(now):.1f}s  ({len(rec.frames)} frame)',
                 (55, 38), RED, 0.8, 2)
    if frame is None:
        _put(view, 'KHONG CO ANH', (w // 2 - 130, h // 2), RED, 1.2, 3)

    panel = np.full((150, w, 3), 30, np.uint8)
    cv2.circle(panel, (18, 20), 8, cam_color, -1)
    _put(panel, cam_status, (36, 26), cam_color, 0.6, 1)
    _put(panel, f'Nguoi: {rec.subject}    Khoang cach: {rec.dist_str}    Da luu: {rec.count_saved()} clip'
                + (f'  (dang ghi {rec.saving})' if rec.saving else '')
                + f'  [{"THU CONG" if rec.manual else "TU DONG"}]', (10, 56), WHITE, 0.6, 1)
    if rec.state == IDLE and rec.message and now < rec.message_until:
        _put(panel, rec.message, (10, 84), GREEN, 0.55, 1)
    elif rec.state != IDLE:
        _put(panel, 'SPACE = tam dung/tiep tuc/bat dau/dung    x = huy clip', (10, 84), YELLOW, 0.55, 1)
    _put(panel, '1 MoveForward  2 Halt  3 MoveInReverse',
         (10, 112), (170, 170, 170), 0.45, 1)
    _put(panel, 'l Left r Right | SPACE quay/dung | m che do | p/o nguoi | d cach | u xoa | x huy | q thoat',
         (10, 134), (170, 170, 170), 0.45, 1)
    return np.vstack([view, panel])


# ---------------------------------------------------------------------------
# Đọc RealSense trực tiếp (không qua ROS)
# ---------------------------------------------------------------------------
class RealSenseReader:
    """Luồng nền đọc ảnh màu từ RealSense, giữ khung mới nhất và đẩy từng khung vào SessionRecorder."""

    def __init__(self, rec: SessionRecorder, width: int, height: int, fps: int, serial: str = ''):
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise SystemExit('Chua cai pyrealsense2. Cai bang: pip install pyrealsense2') from e
        self.rec = rec
        if len(rs.context().query_devices()) == 0:
            raise SystemExit('Khong thay camera RealSense nao. Kiem tra cap USB 3 va udev rules.')
        cfg = rs.config()
        if serial:
            cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.pipeline = rs.pipeline()
        self.pipeline.start(cfg)
        self.latest: Optional[np.ndarray] = None
        self.last_rx = 0.0
        self.error = ''
        self.rx_times: List[float] = []
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while not self._stop:
            try:
                frames = self.pipeline.wait_for_frames(2000)
                color = frames.get_color_frame()
                if not color:
                    continue
                frame = np.asanyarray(color.get_data()).copy()
            except RuntimeError as e:          # quá thời gian chờ / rút cáp
                self.error = str(e)
                continue
            self.error = ''
            now = time.monotonic()
            self.latest, self.last_rx = frame, now
            self.rx_times.append(now)
            self.rx_times = [t for t in self.rx_times if now - t < 2.0]
            self.rec.on_frame(now, frame)

    def status(self, now: float):
        if self.latest is None:
            return 'CHUA CO ANH tu RealSense' + (f' ({self.error})' if self.error else ''), YELLOW
        if now - self.last_rx > 1.0:
            return f'MAT ANH {now - self.last_rx:.0f}s truoc (camera dung / rut cap?)', RED
        fps = len(self.rx_times) / 2.0
        h, w = self.latest.shape[:2]
        return f'REALSENSE OK  {w}x{h} @ {fps:.0f} FPS', GREEN if fps >= 15 else YELLOW

    def close(self):
        self._stop = True
        self._thread.join(timeout=3.0)
        try:
            self.pipeline.stop()
        except Exception:
            pass


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description='Thu dataset cu chi truc tiep tu RealSense (khong can ROS)')
    ap.add_argument('--out-dir', default=str(Path.home() / 'robot_gesture_data'))
    ap.add_argument('--subject', default='S11')
    ap.add_argument('--distances', type=float, nargs='+', default=[2.0, 4.0, 6.0, 8.0])
    ap.add_argument('--countdown', type=float, default=3.0)
    ap.add_argument('--duration', type=float, default=5.0)
    ap.add_argument('--manual', action='store_true', help='che do thu cong: chon nhan roi bam SPACE de quay/dung')
    ap.add_argument('--width', type=int, default=640)
    ap.add_argument('--height', type=int, default=480)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--serial', default='', help='so serial RealSense neu cam nhieu camera')
    args = ap.parse_args(argv)

    rec = SessionRecorder(Path(args.out_dir).expanduser(), args.subject, args.distances,
                          args.countdown, args.duration, manual=args.manual)
    cam = RealSenseReader(rec, args.width, args.height, args.fps, args.serial)
    cam.start()
    print(f'Luu vao {rec.out_dir}')

    win = 'Thu dataset cu chi'
    try:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        while True:
            now = time.monotonic()
            rec.tick(now)
            status, color = cam.status(now)
            fresh = cam.latest is not None and now - cam.last_rx <= 1.0
            cv2.imshow(win, draw_ui(cam.latest if fresh else None, status, color, rec, now))
            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue
            ch = chr(key).lower()
            if ch == 'q' or key == 27:
                break
            elif ch in KEY_TO_CLASS:
                if not fresh:
                    rec.say('Chua co anh tu camera, khong quay duoc', 3.0)
                elif rec.state == IDLE:
                    rec.start(KEY_TO_CLASS[ch], now)
            elif key == 32:
                rec.space(now)
            elif ch == 'm':
                rec.toggle_manual()
            elif ch == 'x':
                rec.cancel()
            elif ch == 'u':
                rec.undo()
            elif ch == 'p':
                rec.change_subject(+1)
            elif ch == 'o':
                rec.change_subject(-1)
            elif ch == 'd':
                rec.next_distance()
    finally:
        cv2.destroyAllWindows()
        cam.close()
        # đợi clip đang ghi nền xong
        t0 = time.monotonic()
        while rec.saving and time.monotonic() - t0 < 15:
            time.sleep(0.1)


if __name__ == '__main__':
    main()
