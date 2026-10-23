"""
rocog_labels.py - Chỗ DUY NHẤT định nghĩa "lớp là gì" và "video nào thuộc tập nào".

Dữ liệu RoCoG-v2 (https://github.com/reddyav1/RoCoG-v2):
  - class_dict.json: 0 Advance, 1 Attention, 2 Rally, 3 MoveForward,
                     4 Halt, 5 FollowMe, 6 MoveInReverse
  - annotations/*.txt: mỗi dòng "đường/dẫn/video.mp4 <số nhãn>", ví dụ
        real/ground/Halt/S01_10m_ground_label5_start2363.mp4 4
    Lưu ý: chữ "label5" trong TÊN FILE là mã nội bộ của nhóm tác giả, KHÔNG phải số
    nhãn. Số nhãn đúng là con số cuối dòng (ở đây là 4 = Halt).

Các tập dùng cho real_ground (chia theo người, chính thức):
  real_ground_train_tsonly.txt  162 video  (S01 S03 S05 S07 S08)  -> train
  real_ground_val_tsonly.txt     42 video  (S02 S10)               -> val
  real_ground_test.txt          100 video  (S00 S04 S04b S06 S09)  -> test
  real_ground_train.txt         204 video  = train_tsonly + val_tsonly

Chạy thử:
    python rocog_labels.py
    python rocog_labels.py --ann-dir Dataset/real_ground/annotations --dataset-root Dataset

Lớp của model (CLASS_NAMES) khác 7 lớp RoCoG: bỏ Advance/Attention/Rally/FollowMe (video các lớp này bị bỏ
qua khi đọc file chia), thêm Left/Right. Model không có lớp None: lệnh yếu/không chắc được lọc ở robot bằng ngưỡng conf.
Sửa CLASS_NAMES thì phải train lại.
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import config

# ---------------------------------------------------------------------------
# Bảng lớp
# ---------------------------------------------------------------------------
# 7 lớp gốc của RoCoG: số nhãn trong các file annotations/*.txt của RoCoG theo thứ tự này.
ROCOG_CLASS_NAMES: List[str] = [
    "Advance",        # 0
    "Attention",      # 1
    "Rally",          # 2
    "MoveForward",    # 3
    "Halt",           # 4
    "FollowMe",       # 5
    "MoveInReverse",  # 6
]

# Lớp của model. THỨ TỰ NÀY = số nhãn trong file robot (record_node.py phải dùng đúng thứ tự này).
CLASS_NAMES: List[str] = [
    "MoveForward",    # 0
    "Halt",           # 1
    "MoveInReverse",  # 2
    "Left",           # 3
    "Right",          # 4
]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(CLASS_NAMES)}

# Tên tập -> file annotation. Có thể truyền thẳng tên file (không cần .txt) cho mọi tập
# khác có trong thư mục annotations (ví dụ "syn_ground_train").
SPLIT_FILES = {
    "train": "real_ground_train_tsonly",
    "val": "real_ground_val_tsonly",
    "test": "real_ground_test",
    "train_full": "real_ground_train",
    "syn_train": "syn_ground_train",
    "syn_val": "syn_ground_val",
}


@dataclass
class VideoEntry:
    """Một video trong một file chia."""
    rel_path: str      # đường dẫn như trong file chia: real/ground/Halt/S01_....mp4
    label: int         # số nhãn của model (chỉ số trong CLASS_NAMES)
    class_name: str    # "Halt"
    subject: str       # "S01", "S04b"; video tổng hợp thì là tên scene, ví dụ "Scene1"
    source: str        # "real" hoặc "syn"
    split: str         # tên file chia nó đến từ
    raw_label: int = -1          # số nhãn ghi trong file chia (RoCoG: theo ROCOG_CLASS_NAMES)

    @property
    def stem(self) -> str:
        return Path(self.rel_path).stem

    @property
    def subject_group(self) -> str:
        """S04b là một lượt quay khác của S04 -> gộp chung khi thống kê theo người."""
        return subject_group(self.subject)

    def npz_relpath(self) -> Path:
        """Vị trí file keypoint tương ứng bên trong KEYPOINT_DIR."""
        return Path(self.rel_path).with_suffix(".npz")


# ---------------------------------------------------------------------------
# Tách thông tin từ tên file
# ---------------------------------------------------------------------------
_REAL_SUBJECT_RE = re.compile(r"^(S\d+b?)_", re.IGNORECASE)
_SYN_SCENE_RE = re.compile(r"^(Scene\d+)_", re.IGNORECASE)


def parse_subject(filename: str) -> str:
    """'S04b_10m_ground_label7_start1927.mp4' -> 'S04b'; 'Scene1_0_Advance_...' -> 'Scene1'."""
    name = Path(filename).name
    m = _REAL_SUBJECT_RE.match(name)
    if m:
        return m.group(1)
    m = _SYN_SCENE_RE.match(name)
    if m:
        return m.group(1)
    return "unknown"


def subject_group(subject: str) -> str:
    """Bỏ hậu tố 'b' (S04b -> S04)."""
    return subject[:-1] if re.fullmatch(r"S\d+b", subject) else subject


# ---------------------------------------------------------------------------
# Đọc file chia
# ---------------------------------------------------------------------------
def split_file_path(split: str, ann_dir: Path | str = config.ANNOTATION_DIR) -> Path:
    name = SPLIT_FILES.get(split, split)
    if not name.endswith(".txt"):
        name += ".txt"
    return Path(ann_dir) / name


def read_split(split: str, ann_dir: Path | str = config.ANNOTATION_DIR) -> List[VideoEntry]:
    """
    Đọc một file chia, trả về danh sách VideoEntry.

    Kiểm tra chéo: tên thư mục chứa video (Halt, Rally...) phải trùng với tên lớp của số
    nhãn. Sai thì báo lỗi ngay, tránh train trên nhãn sai mà không biết.
    """
    path = split_file_path(split, ann_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy file chia {path}.\n"
            f"Tải thư mục annotations từ https://github.com/reddyav1/RoCoG-v2 "
            f"và đặt vào {Path(ann_dir)} (hoặc truyền --ann-dir)."
        )

    entries: List[VideoEntry] = []
    errors: List[str] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            errors.append(f"dòng {line_no}: không đúng dạng '<đường dẫn> <nhãn>': {line}")
            continue
        rel_path, label_str = parts
        raw = int(label_str)
        # Video robot tự thu (real/robot/...) dùng số nhãn của CLASS_NAMES; còn lại là RoCoG.
        names = CLASS_NAMES if rel_path.startswith("real/robot/") else ROCOG_CLASS_NAMES
        if not 0 <= raw < len(names):
            errors.append(f"dòng {line_no}: nhãn {raw} ngoài khoảng 0..{len(names) - 1}")
            continue

        name = names[raw]
        folder = Path(rel_path).parent.name
        if folder in names and folder != name:
            errors.append(
                f"dòng {line_no}: thư mục '{folder}' nhưng nhãn {raw} "
                f"({name}) -> {rel_path}"
            )
            continue
        if name not in CLASS_TO_ID:      # lớp RoCoG không dùng (Advance, Attention, Rally, FollowMe)
            continue
        label = CLASS_TO_ID[name]

        source = "syn" if rel_path.startswith("syn") else "real"
        entries.append(VideoEntry(
            rel_path=rel_path,
            label=label,
            raw_label=raw,
            class_name=name,
            subject=parse_subject(rel_path),
            source=source,
            split=path.stem,
        ))

    if errors:
        raise ValueError(f"File chia {path} có {len(errors)} lỗi:\n  " + "\n  ".join(errors[:20]))
    return entries


def read_splits(splits: List[str], ann_dir: Path | str = config.ANNOTATION_DIR) -> List[VideoEntry]:
    out: List[VideoEntry] = []
    for s in splits:
        out.extend(read_split(s, ann_dir))
    return out


def sample_balanced(entries: List[VideoEntry], max_per_class: Optional[int],
                    seed: int = config.SEED) -> List[VideoEntry]:
    """Lấy tối đa max_per_class video mỗi lớp (dùng cho dữ liệu tổng hợp 44.000 video)."""
    if not max_per_class:
        return entries
    rng = random.Random(seed)
    by_class: Dict[int, List[VideoEntry]] = defaultdict(list)
    for e in entries:
        by_class[e.label].append(e)
    out: List[VideoEntry] = []
    for label in sorted(by_class):
        items = by_class[label][:]
        rng.shuffle(items)
        out.extend(items[:max_per_class])
    return out


# ---------------------------------------------------------------------------
# Tìm file video trên đĩa
# ---------------------------------------------------------------------------
_FILENAME_INDEX: Dict[str, Dict[str, Path]] = {}


def _build_filename_index(root: Path) -> Dict[str, Path]:
    key = str(root.resolve())
    if key not in _FILENAME_INDEX:
        index: Dict[str, Path] = {}
        for p in root.rglob("*.mp4"):
            index.setdefault(p.name, p)
        _FILENAME_INDEX[key] = index
    return _FILENAME_INDEX[key]


def resolve_video_path(entry: VideoEntry, dataset_root: Path | str = config.DATASET_ROOT) -> Optional[Path]:
    """
    Tìm file video trên đĩa. Thử lần lượt:
      1. <root>/real/ground/Halt/x.mp4              (giữ nguyên cấu trúc của file chia)
      2. <root>/real_ground/real/ground/Halt/x.mp4  (giải nén real_ground.zip vào <root>:
                                                     đây là cấu trúc thật của dự án này)
      3. <root>/real_ground/Halt/x.mp4
      4. <root>/Halt/x.mp4                          (root trỏ thẳng vào real/ground)
      5. tìm theo tên file trong toàn bộ <root> (chậm lần đầu, sau đó có cache)
    Trả về None nếu không thấy.
    """
    root = Path(dataset_root)
    rel = Path(entry.rel_path)
    parts = rel.parts  # ('real', 'ground', 'Halt', 'x.mp4')
    candidates = [root / rel, root / f"{parts[0]}_{parts[1]}" / rel] if len(parts) >= 3 \
        else [root / rel]
    if len(parts) >= 3:
        candidates.append(root / f"{parts[0]}_{parts[1]}" / Path(*parts[2:]))
    candidates.append(root / rel.parent.name / rel.name)
    for c in candidates:
        if c.exists():
            return c
    if root.exists():
        return _build_filename_index(root).get(rel.name)
    return None


# ---------------------------------------------------------------------------
# Kiểm tra tổng thể
# ---------------------------------------------------------------------------
def summarize(splits: Dict[str, List[VideoEntry]]) -> None:
    """In số video mỗi lớp mỗi tập, và kiểm tra các tập không trùng người."""
    names = list(splits)
    header = f"{'Lớp':<15}" + "".join(f"{n:>12}" for n in names)
    print(header)
    print("-" * len(header))
    counts = {n: Counter(e.class_name for e in es) for n, es in splits.items()}
    for c in CLASS_NAMES:
        print(f"{c:<15}" + "".join(f"{counts[n][c]:>12}" for n in names))
    print(f"{'TỔNG':<15}" + "".join(f"{len(splits[n]):>12}" for n in names))
    print()

    subjects = {n: sorted({e.subject_group for e in es}) for n, es in splits.items()}
    for n in names:
        print(f"Người trong '{n}': {', '.join(subjects[n])}")

    ok = True
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = set(subjects[a]) & set(subjects[b])
            vids = {e.rel_path for e in splits[a]} & {e.rel_path for e in splits[b]}
            if vids:
                print(f"[LỖI] '{a}' và '{b}' trùng {len(vids)} video")
                ok = False
            if overlap:
                print(f"[CẢNH BÁO] '{a}' và '{b}' có chung người: {sorted(overlap)}")
                ok = False
    if ok:
        print("\nKhông có video hay người nào bị trùng giữa các tập.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Kiểm tra file chia RoCoG")
    ap.add_argument("--ann-dir", default=str(config.ANNOTATION_DIR))
    ap.add_argument("--dataset-root", default=str(config.DATASET_ROOT))
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--check-files", action="store_true",
                    help="kiểm tra từng video có tồn tại trên đĩa không")
    args = ap.parse_args()

    splits = {s: read_split(s, args.ann_dir) for s in args.splits}
    summarize(splits)

    if args.check_files:
        missing = [e for es in splits.values() for e in es
                   if resolve_video_path(e, args.dataset_root) is None]
        total = sum(len(es) for es in splits.values())
        print(f"\nTìm thấy {total - len(missing)}/{total} video trong {args.dataset_root}")
        for e in missing[:10]:
            print("  thiếu:", e.rel_path)


if __name__ == "__main__":
    main()