"""
gesture_dataset.py - Biến các file .npz (chuỗi khung xương) thành tensor (T, J, 2) cho PyTorch.

Các bước cho mỗi video (clip):
  1. Đọc .npz do extract_keypoints.py tạo.
  2. Loại clip có quá ít frame dùng được (ok_ratio < MIN_OK_RATIO).
  3. Lấp frame hỏng + resample về FPS_TARGET bằng nội suy tuyến tính THEO THỜI GIAN THẬT
     (video 24 FPS và 29 FPS đều về cùng một nhịp; frame hỏng ở giữa được lấp lại thay vì
     bị bỏ đi làm cử chỉ trông như tua nhanh).
  4. Cắt cửa sổ T frame:
       - train: mỗi lần lấy một cửa sổ NGẪU NHIÊN trong clip (kèm co giãn tốc độ),
                bỏ TRIM_SEC ở đầu/cuối clip vì ở đó người có thể chưa làm cử chỉ;
       - val/test: trượt cửa sổ đều đặn (EVAL_STRIDE), sau này lấy trung bình xác suất
                   các cửa sổ để ra nhãn cho cả video (giống cách robot nhìn 3,2 giây gần nhất).
     Clip ngắn hơn T frame: lặp frame đầu/cuối cho đủ (người đứng yên trước/sau cử chỉ),
     KHÔNG kéo giãn, để tốc độ cử chỉ giữ nguyên.
  5. Augment (chỉ train): xoay, phóng to/thu nhỏ, nhiễu, che ngẫu nhiên một khớp,
     lật ngang (tắt mặc định).

Có thêm chế độ "whole": nén/giãn CẢ clip về T frame giống code DD-Net gốc (hàm zoom).
Dùng để so sánh, hoặc khi pretrain bằng JHMDB (clip chỉ dài ~1 giây).

Chạy thử (in thống kê, kiểm tra dữ liệu):
    python gesture_dataset.py --splits train val test
"""

from __future__ import annotations

import argparse
import math
import pickle
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

import config
from rocog_labels import CLASS_NAMES, VideoEntry, read_splits, subject_group


# ---------------------------------------------------------------------------
# Một clip đã resample
# ---------------------------------------------------------------------------
@dataclass
class Clip:
    seq: np.ndarray          # (L, J, 2) float32, đã lấp chỗ trống, nhịp FPS_TARGET
    label: int
    subject: str
    rel_path: str
    fps: float               # nhịp của seq (thường = FPS_TARGET)
    ok_ratio: float = 1.0
    source: str = "real"

    @property
    def subject_group(self) -> str:
        return subject_group(self.subject)

    @property
    def duration(self) -> float:
        return len(self.seq) / self.fps


# ---------------------------------------------------------------------------
# Resample theo thời gian
# ---------------------------------------------------------------------------
def resample_by_time(t: np.ndarray, joints: np.ndarray, ok: np.ndarray,
                     fps: float) -> Optional[np.ndarray]:
    """
    t (N,), joints (N,J,2) (NaN ở frame hỏng), ok (N,) -> (L,J,2) đều nhịp `fps`.

    Chỉ dùng các frame ok làm điểm neo, nội suy tuyến tính ở giữa. Lưới thời gian chạy
    từ frame ok đầu tiên đến frame ok cuối cùng (không bịa dữ liệu ngoài khoảng đó).
    """
    ok = ok & np.isfinite(joints).all(axis=(1, 2))
    if ok.sum() < 2:
        return None
    t_ok = t[ok]
    j_ok = joints[ok].reshape(ok.sum(), -1)          # (n, J*2)
    t_new = np.arange(t_ok[0], t_ok[-1] + 1e-9, 1.0 / fps)
    if len(t_new) < 2:
        return None
    out = np.empty((len(t_new), j_ok.shape[1]), np.float32)
    for c in range(j_ok.shape[1]):
        out[:, c] = np.interp(t_new, t_ok, j_ok[:, c])
    return out.reshape(len(t_new), joints.shape[1], joints.shape[2])


def resample_length(seq: np.ndarray, length: int) -> np.ndarray:
    """Co giãn (L,J,D) về đúng `length` frame bằng nội suy tuyến tính (giống zoom() gốc)."""
    L = len(seq)
    if L == length:
        return seq.copy()
    if L == 1:
        return np.repeat(seq, length, axis=0)
    x_old = np.linspace(0.0, 1.0, L)
    x_new = np.linspace(0.0, 1.0, length)
    flat = seq.reshape(L, -1)
    out = np.empty((length, flat.shape[1]), np.float32)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(x_new, x_old, flat[:, c])
    return out.reshape(length, *seq.shape[1:])


def pad_edges(seq: np.ndarray, length: int, left: int) -> np.ndarray:
    """Clip ngắn: lặp frame đầu `left` lần và frame cuối cho đủ `length` frame."""
    right = length - len(seq) - left
    return np.concatenate([np.repeat(seq[:1], left, 0), seq, np.repeat(seq[-1:], right, 0)], 0)


# ---------------------------------------------------------------------------
# Đọc dữ liệu
# ---------------------------------------------------------------------------
def load_npz_clip(path: Path, fps: float = config.FPS_TARGET,
                  min_ok_ratio: float = config.MIN_OK_RATIO) -> Tuple[Optional[Clip], str]:
    """Đọc một file .npz -> (Clip, "") hoặc (None, lý do bị loại)."""
    if not path.exists():
        return None, "chưa có .npz"
    with np.load(path, allow_pickle=False) as z:
        t, joints, ok = z["t"], z["joints"], z["ok"].astype(bool)
        label = int(z["label"])
        subject = str(z["subject"])
        rel = str(z["rel_path"])
    ok_ratio = float(ok.mean()) if len(ok) else 0.0
    if ok_ratio < min_ok_ratio:
        return None, f"ok_ratio={ok_ratio:.2f}"
    seq = resample_by_time(t, joints, ok, fps)
    if seq is None or len(seq) < 4:
        return None, "quá ngắn sau khi resample"
    return Clip(seq=seq, label=label, subject=subject, rel_path=rel, fps=fps,
                ok_ratio=ok_ratio, source="syn" if rel.startswith("syn") else "real"), ""


def load_clips(entries: Sequence[VideoEntry], kp_dir: Path | str = config.KEYPOINT_DIR,
               fps: float = config.FPS_TARGET, min_ok_ratio: float = config.MIN_OK_RATIO,
               verbose: bool = True) -> List[Clip]:
    clips, skipped = [], Counter()
    examples: Dict[str, str] = {}
    for e in entries:
        clip, reason = load_npz_clip(Path(kp_dir) / e.npz_relpath(), fps, min_ok_ratio)
        if clip is None:
            key = reason.split("=")[0]
            skipped[key] += 1
            examples.setdefault(key, e.rel_path)
            continue
        # .npz trích trước khi gộp lớp lưu số nhãn gốc của file chia (raw_label); trích mới lưu số nhãn model.
        if clip.label not in (e.label, e.raw_label):
            raise ValueError(f"Nhãn trong {e.npz_relpath()} ({clip.label}) khác file chia "
                             f"({e.raw_label}/{e.label}). Hãy trích lại video này.")
        clip.label = e.label
        clips.append(clip)
    if verbose:
        print(f"  nạp {len(clips)}/{len(entries)} clip", end="")
        if skipped:
            print(" | bị loại: " + ", ".join(f"{k}: {v} (vd {examples[k]})" for k, v in skipped.items()), end="")
        print()
    return clips


def load_split_clips(splits: Sequence[str], ann_dir=config.ANNOTATION_DIR,
                     kp_dir=config.KEYPOINT_DIR, fps=config.FPS_TARGET,
                     min_ok_ratio=config.MIN_OK_RATIO) -> List[Clip]:
    print(f"Tập {'+'.join(splits)}:")
    return load_clips(read_splits(list(splits), ann_dir), kp_dir, fps, min_ok_ratio)


def load_jhmdb_clips(pkl_path: Path | str, class_names: Optional[List[str]] = None
                     ) -> Tuple[List[Clip], List[str]]:
    """
    Đọc dữ liệu JHMDB có sẵn trong repo DD-Net (data/JHMDB/GT_train_1.pkl...) để pretrain.

    File pkl là dict {'pose': [mảng (frames,15,2)], 'label': [tên lớp]}. Toạ độ được chuẩn
    hoá GIỐNG skeleton_adapter: gốc = trung điểm hai vai, thước = độ dài thân / 1.3,
    và đổi thứ tự khớp theo config.JHMDB_TO_ADAPTER nếu cần.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if class_names is None:
        class_names = sorted(set(data["label"]))
    name_to_id = {n: i for i, n in enumerate(class_names)}
    order = config.JHMDB_TO_ADAPTER
    rs, ls = config.ADAPTER_R_SHOULDER, config.ADAPTER_L_SHOULDER
    rh, lh = config.ADAPTER_R_HIP, config.ADAPTER_L_HIP

    clips = []
    for i, (pose, name) in enumerate(zip(data["pose"], data["label"])):
        if name not in name_to_id:
            continue
        p = np.asarray(pose, np.float32)
        if order is not None:
            p = p[:, order]
        sh = (p[:, rs] + p[:, ls]) / 2
        hip = (p[:, rh] + p[:, lh]) / 2
        torso = np.median(np.linalg.norm(hip - sh, axis=1))
        scale = max(torso / config.TORSO_TO_SHOULDER_RATIO, 1e-6)
        p = (p - sh[:, None, :]) / scale
        clips.append(Clip(seq=p, label=name_to_id[name], subject=f"jhmdb{i}",
                          rel_path=f"jhmdb/{name}/{i}", fps=config.JHMDB_FPS, source="jhmdb"))
    return clips, class_names


# ---------------------------------------------------------------------------
# Augment
# ---------------------------------------------------------------------------
def _get_mirror_fn() -> Callable[[np.ndarray], np.ndarray]:
    """Dùng mirror_joints() trong skeleton_adapter.py (đổi dấu x + hoán đổi khớp trái/phải)."""
    from skeleton_adapter import mirror_joints

    def fn(seq: np.ndarray) -> np.ndarray:
        try:
            out = np.asarray(mirror_joints(seq), np.float32)   # nếu hàm nhận được cả chuỗi
            if out.shape == seq.shape:
                return out
        except Exception:
            pass
        return np.stack([np.asarray(mirror_joints(f), np.float32) for f in seq])
    return fn


def mirror_label_map(class_names: Sequence[str]) -> Dict[int, int]:
    """Nhãn sau khi lật: LEFT <-> RIGHT nếu có hai lớp đó, các lớp khác giữ nguyên."""
    m = {i: i for i in range(len(class_names))}
    upper = [c.upper() for c in class_names]
    for a, b in (("LEFT", "RIGHT"), ("TURNLEFT", "TURNRIGHT"), ("MOVELEFT", "MOVERIGHT")):
        if a in upper and b in upper:
            ia, ib = upper.index(a), upper.index(b)
            m[ia], m[ib] = ib, ia
    return m


@dataclass
class Augmenter:
    rotate_deg: float = config.AUG_ROTATE_DEG
    scale: Tuple[float, float] = config.AUG_SCALE
    noise_std: float = config.AUG_NOISE_STD
    joint_drop_prob: float = config.AUG_JOINT_DROP_PROB
    mirror: bool = config.USE_MIRROR
    mirror_prob: float = 0.5
    _mirror_fn: Optional[Callable] = field(default=None, repr=False)

    def __call__(self, seq: np.ndarray, label: int, label_map: Dict[int, int],
                 rng: random.Random) -> Tuple[np.ndarray, int]:
        seq = seq.copy()
        T, J, _ = seq.shape

        # Lật ngang (đổi nhãn LEFT<->RIGHT nếu có)
        if self.mirror and rng.random() < self.mirror_prob:
            if self._mirror_fn is None:
                self._mirror_fn = _get_mirror_fn()
            seq = self._mirror_fn(seq)
            label = label_map.get(label, label)

        # Xoay quanh gốc (trung điểm hai vai) + phóng to/thu nhỏ (cho phép lệch nhẹ 2 trục)
        th = math.radians(rng.uniform(-self.rotate_deg, self.rotate_deg))
        s = rng.uniform(*self.scale)
        sx, sy = s * rng.uniform(0.95, 1.05), s * rng.uniform(0.95, 1.05)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], np.float32)
        seq = seq @ R.T * np.array([sx, sy], np.float32)

        # Nhiễu Gauss nhỏ ở từng khớp từng frame (giống keypoint rung)
        if self.noise_std > 0:
            seq = seq + np.random.default_rng(rng.randrange(2**31)).normal(
                0, self.noise_std, seq.shape).astype(np.float32)

        # Che một khớp trong một đoạn frame: giữ nguyên giá trị như adapter điền khớp bị che
        if rng.random() < self.joint_drop_prob:
            j = rng.randrange(J)
            span = rng.randint(3, max(3, T // 2))
            start = rng.randrange(0, max(1, T - span))
            hold = seq[max(start - 1, 0), j].copy()
            seq[start:start + span, j] = hold
        return seq.astype(np.float32), label


# ---------------------------------------------------------------------------
# Dataset cho PyTorch
# ---------------------------------------------------------------------------
def _trim_frames(L: int, T: int, trim: int) -> int:
    """Số frame bỏ mỗi đầu, giảm dần nếu clip không đủ dài."""
    return int(max(0, min(trim, (L - T) // 2)))


class TrainWindowDataset(Dataset):
    """
    Mỗi phần tử = một cửa sổ ngẫu nhiên từ một clip. Mỗi clip xuất hiện `samples_per_clip`
    lần trong một epoch, mỗi lần một cửa sổ, tốc độ và augment khác nhau.
    """

    def __init__(self, clips: List[Clip], T: int = config.T,
                 samples_per_clip: int = config.SAMPLES_PER_CLIP,
                 trim_sec: float = config.TRIM_SEC,
                 speed: Tuple[float, float] = config.AUG_SPEED,
                 augmenter: Optional[Augmenter] = None,
                 class_names: Sequence[str] = CLASS_NAMES,
                 window_mode: str = "sliding", seed: int = config.SEED):
        if not clips:
            raise ValueError("Không có clip nào để train. Đã chạy extract_keypoints.py chưa?")
        self.clips = clips
        self.T = T
        self.k = samples_per_clip
        self.trim_sec = trim_sec
        self.speed = speed
        self.aug = augmenter
        self.label_map = mirror_label_map(class_names)
        self.window_mode = window_mode
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.clips) * self.k

    def _window(self, clip: Clip) -> np.ndarray:
        rng = self.rng
        T = self.T
        s = rng.uniform(*self.speed) if self.speed else 1.0     # >1 = cử chỉ nhanh hơn
        seq = clip.seq
        L = len(seq)
        if self.window_mode == "whole":
            # giống zoom() + sampling_frame() gốc: lấy 85-100% clip rồi nén về T
            keep = max(2, int(round(L * rng.uniform(0.85, 1.0))))
            st = rng.randint(0, L - keep)
            return resample_length(seq[st:st + keep], T)

        Ls = max(2, int(round(L / s)))
        seq = resample_length(seq, Ls)
        if Ls >= T:
            trim = _trim_frames(Ls, T, int(round(self.trim_sec * clip.fps / s)))
            start = rng.randint(trim, Ls - T - trim)
            return seq[start:start + T]
        return pad_edges(seq, T, rng.randint(0, T - Ls))

    def __getitem__(self, idx: int):
        clip = self.clips[idx // self.k]
        x = self._window(clip)
        y = clip.label
        if self.aug is not None:
            x, y = self.aug(x, y, self.label_map, self.rng)
        return torch.from_numpy(np.ascontiguousarray(x, np.float32)), y


class EvalWindowDataset(Dataset):
    """
    Cửa sổ cố định để đánh giá. Mỗi phần tử trả (x, label, clip_index); hàm đánh giá gom
    các cửa sổ cùng clip_index và lấy trung bình xác suất -> nhãn của video.
    """

    def __init__(self, clips: List[Clip], T: int = config.T,
                 stride: int = config.EVAL_STRIDE, trim_sec: float = config.TRIM_SEC,
                 window_mode: str = "sliding"):
        self.clips = clips
        self.T = T
        self.items: List[Tuple[int, np.ndarray]] = []
        for ci, clip in enumerate(clips):
            for w in self.windows(clip.seq, T, stride, int(round(trim_sec * clip.fps)), window_mode):
                self.items.append((ci, w))

    @staticmethod
    def windows(seq: np.ndarray, T: int, stride: int, trim: int, mode: str) -> List[np.ndarray]:
        L = len(seq)
        if mode == "whole":
            return [resample_length(seq, T)]
        if L < T:
            return [pad_edges(seq, T, (T - L) // 2)]
        trim = _trim_frames(L, T, trim)
        last = L - T - trim
        starts = list(range(trim, last + 1, max(1, stride)))
        if starts[-1] != last:
            starts.append(last)
        return [seq[s:s + T] for s in starts]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        ci, w = self.items[idx]
        return torch.from_numpy(np.ascontiguousarray(w, np.float32)), self.clips[ci].label, ci


def worker_init_fn(worker_id: int) -> None:
    """
    Dùng khi DataLoader có num_workers > 0: mỗi tiến trình con nhận một bộ sinh số ngẫu
    nhiên riêng (khác nhau và đổi theo từng epoch). Không có hàm này, các worker sẽ sinh
    CÙNG cửa sổ và CÙNG augment.
    """
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng = random.Random(info.seed)


def class_weights(clips: Sequence[Clip], num_classes: int) -> torch.Tensor:
    """Trọng số lớp tỉ lệ nghịch số clip (trung bình = 1). Lớp vắng mặt có trọng số 0."""
    counts = np.bincount([c.label for c in clips], minlength=num_classes).astype(np.float64)
    w = np.zeros(num_classes)
    nz = counts > 0
    w[nz] = counts[nz].sum() / (nz.sum() * counts[nz])
    return torch.tensor(w, dtype=torch.float32)


def num_joints_of(clips: Sequence[Clip]) -> int:
    return int(clips[0].seq.shape[1])


# ---------------------------------------------------------------------------
# Kiểm tra nhanh
# ---------------------------------------------------------------------------
def describe(name: str, clips: List[Clip], class_names: Sequence[str] = CLASS_NAMES) -> None:
    if not clips:
        print(f"[{name}] trống")
        return
    d = np.array([c.duration for c in clips])
    cnt = Counter(c.label for c in clips)
    subs = sorted({c.subject_group for c in clips})
    print(f"[{name}] {len(clips)} clip | độ dài {d.min():.1f}-{d.max():.1f}s (tb {d.mean():.1f}s) "
          f"| người: {', '.join(subs)}")
    print("   " + "  ".join(f"{class_names[k]}={cnt[k]}" for k in range(len(class_names))))
    allj = np.concatenate([c.seq.reshape(-1, 2) for c in clips])
    print(f"   toạ độ: x [{allj[:, 0].min():.2f}, {allj[:, 0].max():.2f}], "
          f"y [{allj[:, 1].min():.2f}, {allj[:, 1].max():.2f}] (đơn vị độ rộng vai chuẩn)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Kiểm tra dữ liệu cho DD-Net")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--ann-dir", default=str(config.ANNOTATION_DIR))
    ap.add_argument("--kp-dir", default=str(config.KEYPOINT_DIR))
    ap.add_argument("--fps", type=float, default=config.FPS_TARGET)
    ap.add_argument("--T", type=int, default=config.T)
    args = ap.parse_args()

    for s in args.splits:
        clips = load_split_clips([s], args.ann_dir, args.kp_dir, args.fps)
        describe(s, clips)
        if not clips:
            continue
        if s == "train":
            ds = TrainWindowDataset(clips, args.T, augmenter=Augmenter())
            x, y = ds[0]
            print(f"   mẫu train: x {tuple(x.shape)}, nhãn {y}; {len(ds)} mẫu mỗi epoch")
        else:
            ds = EvalWindowDataset(clips, args.T)
            print(f"   {len(ds)} cửa sổ đánh giá (trung bình {len(ds) / len(clips):.1f} cửa sổ/clip)")


if __name__ == "__main__":
    main()