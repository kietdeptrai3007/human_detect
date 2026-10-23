"""
ddnet_model.py - DD-Net (Double-feature Double-motion Network) viết lại bằng PyTorch.

Dựa trên code gốc (Keras) của tác giả: https://github.com/fandulu/DD-Net
  JHMDB/jhmdb_1D_lite.ipynb  -> hàm build_FM(), build_DD_Net(), poses_diff(), c1D(), block()
  JHMDB/utils.py             -> get_CG() (JCD) và norm_scale()
Bài báo: Yang et al., "Make Skeleton-based Action Recognition Model Smaller, Faster and
Better", 2019.

Repo gốc KHÔNG kèm file trọng số (thư mục weights/ trống), nên ở đây ta viết lại kiến
trúc giống từng lớp và train từ đầu (hoặc tự pretrain, xem train_ddnet.py).

Khác code gốc ở 2 chỗ, đều có chủ ý:
  1. JCD và đặc trưng chuyển động được tính BÊN TRONG model (code gốc tính JCD bằng numpy
     ngoài model). Nhờ vậy lúc train, lúc chạy trên robot và khi xuất ONNX chỉ cần đưa
     khung xương (B, T, J, 2) vào, không thể lệch cách tính đặc trưng.
  2. Số lớp, T, số khớp, số filter là tham số.

Sơ đồ (f = filters, mặc định 16 = bản "lite"):

  P (B,T,J,2) ──┬─ JCD (B,T,J(J-1)/2) ─ c1D(2f,1) c1D(f,3) c1D(f,1) MaxPool ─┐  (B,T/2,f)
                ├─ chuyển động chậm (B,T,2J)   ─ c1D(2f,1) c1D(f,3) c1D(f,1) MaxPool ─┤  (B,T/2,f)
                └─ chuyển động nhanh (B,T/2,2J)─ c1D(2f,1) c1D(f,3) c1D(f,1)         ─┘  (B,T/2,f)
                         ghép -> (B,T/2,3f)
                         block(2f) MaxPool -> block(4f) MaxPool -> block(8f)  (B,T/8,8f)
                         GlobalMaxPool -> d1D(128) Dropout -> d1D(128) Dropout -> Linear(num_classes)

Chạy thử:
    python ddnet_model.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config


# ---------------------------------------------------------------------------
# Đặc trưng
# ---------------------------------------------------------------------------
def joint_collection_distances(P: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    JCD - Joint Collection Distances.

    P: (B, T, J, 2) -> (B, T, J*(J-1)/2)

    Ở mỗi frame tính khoảng cách Euclid giữa mọi cặp khớp, chỉ lấy nửa trên của ma trận
    (giống np.triu_indices(J, 1) trong get_CG gốc). Đặc trưng này không đổi khi người
    dịch chuyển hay xoay trong ảnh -> model tập trung vào "dáng".

    Sau đó chuẩn hoá giống norm_scale() gốc: (M - mean) / mean, với mean lấy trên toàn
    bộ cửa sổ của từng mẫu -> không phụ thuộc người to hay nhỏ.

    Khoảng cách tính bằng phép trừ thường (không dùng torch.cdist) để xuất ONNX dễ.
    """
    B, T, J, D = P.shape
    diff = P.unsqueeze(3) - P.unsqueeze(2)                    # (B,T,J,J,D)
    dist = torch.sqrt((diff ** 2).sum(-1) + eps)              # (B,T,J,J); +eps để đạo hàm tại 0 không bị NaN
    iu = torch.triu_indices(J, J, offset=1, device=P.device)
    M = dist[:, :, iu[0], iu[1]]                              # (B,T,J(J-1)/2)
    mean = M.mean(dim=(1, 2), keepdim=True).clamp_min(eps)
    return (M - mean) / mean


def _resize_time(x: torch.Tensor, length: int) -> torch.Tensor:
    """
    Co giãn trục thời gian về `length` bằng nội suy tuyến tính.
    Tương đương tf.image.resize(...) trong poses_diff() gốc (half-pixel = align_corners=False).
    x: (B, L, C) -> (B, length, C)
    """
    if x.shape[1] == length:
        return x
    y = F.interpolate(x.transpose(1, 2), size=length, mode="linear", align_corners=False)
    return y.transpose(1, 2)


def pose_motion(P: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hai đặc trưng chuyển động, giống pose_motion() gốc:
      - chậm: hiệu frame t+1 và t trên cả chuỗi T frame, co giãn lại về T
      - nhanh: lấy 1 frame mỗi 2 frame (T/2 frame), lấy hiệu, co giãn về T/2
    P: (B,T,J,D) -> slow (B,T,J*D), fast (B,T/2,J*D)
    """
    B, T, J, D = P.shape
    slow = P[:, 1:] - P[:, :-1]                                # (B,T-1,J,D)
    slow = _resize_time(slow.reshape(B, T - 1, J * D), T)
    P_fast = P[:, ::2]
    Tf = P_fast.shape[1]
    fast = P_fast[:, 1:] - P_fast[:, :-1]
    fast = _resize_time(fast.reshape(B, Tf - 1, J * D), T // 2)
    return slow, fast


# ---------------------------------------------------------------------------
# Các khối mạng (giống c1D / block / d1D gốc)
# ---------------------------------------------------------------------------
class C1D(nn.Module):
    """Conv1D (không bias) + BatchNorm + LeakyReLU(0.2). Vào/ra dạng (B, C, T)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch, eps=1e-3)   # eps giống Keras
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Block(nn.Module):
    """Hai C1D kernel 3 liên tiếp."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c1 = C1D(in_ch, out_ch, 3)
        self.c2 = C1D(out_ch, out_ch, 3)

    def forward(self, x):
        return self.c2(self.c1(x))


class D1D(nn.Module):
    """Linear (không bias) + BatchNorm + LeakyReLU(0.2)."""

    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        self.fc = nn.Linear(in_f, out_f, bias=False)
        self.bn = nn.BatchNorm1d(out_f, eps=1e-3)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.act(self.bn(self.fc(x)))


def _branch(in_ch: int, f: int, drop: float, pool: bool) -> nn.Sequential:
    """Một nhánh đầu vào: c1D(2f,1) -> c1D(f,3) -> c1D(f,1) [-> MaxPool], xen SpatialDropout."""
    layers = [C1D(in_ch, 2 * f, 1), nn.Dropout1d(drop),
              C1D(2 * f, f, 3), nn.Dropout1d(drop),
              C1D(f, f, 1)]
    if pool:
        layers.append(nn.MaxPool1d(2))
    layers.append(nn.Dropout1d(drop))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# DD-Net
# ---------------------------------------------------------------------------
class DDNet(nn.Module):
    """
    Đầu vào : khung xương (B, T, J, D), toạ độ đã chuẩn hoá bởi skeleton_adapter.
    Đầu ra  : logits (B, num_classes). Muốn xác suất thì softmax(logits).
    """

    def __init__(self, num_classes: int, T: int = config.T, num_joints: int = 15,
                 joint_dim: int = 2, filters: int = config.DDNET_FILTERS,
                 dropout: float = config.DDNET_DROPOUT,
                 fc_dropout: float = config.DDNET_FC_DROPOUT):
        super().__init__()
        if T % 8 != 0:
            raise ValueError(f"T phải chia hết cho 8 (qua 3 lần MaxPool), nhận T={T}")
        self.hparams = dict(num_classes=num_classes, T=T, num_joints=num_joints,
                            joint_dim=joint_dim, filters=filters, dropout=dropout,
                            fc_dropout=fc_dropout)
        f = filters
        feat_d = num_joints * (num_joints - 1) // 2      # 105 với 15 khớp
        motion_d = num_joints * joint_dim                # 30 với 15 khớp 2D

        # ----- phần trích đặc trưng (backbone) -----
        self.jcd_branch = _branch(feat_d, f, dropout, pool=True)
        self.slow_branch = _branch(motion_d, f, dropout, pool=True)
        self.fast_branch = _branch(motion_d, f, dropout, pool=False)

        self.block1 = Block(3 * f, 2 * f)
        self.block2 = Block(2 * f, 4 * f)
        self.block3 = Block(4 * f, 8 * f)
        self.pool = nn.MaxPool1d(2)
        self.drop = nn.Dropout1d(dropout)

        # ----- đầu phân loại -----
        self.fc1 = D1D(8 * f, 128)
        self.fc2 = D1D(128, 128)
        self.fc_drop = nn.Dropout(fc_dropout)
        self.classifier = nn.Linear(128, num_classes)

    # Tên module thuộc "đầu" (bị thay khi đổi số lớp / được train khi đóng băng backbone)
    HEAD_MODULES = ("fc1", "fc2", "fc_drop", "classifier")

    def features(self, P: torch.Tensor) -> torch.Tensor:
        """(B,T,J,D) -> vector đặc trưng (B, 8f) trước đầu phân loại."""
        M = joint_collection_distances(P)                      # (B,T,feat_d)
        slow, fast = pose_motion(P)                            # (B,T,2J), (B,T/2,2J)

        x = self.jcd_branch(M.transpose(1, 2))                 # (B,f,T/2)
        xs = self.slow_branch(slow.transpose(1, 2))            # (B,f,T/2)
        xf = self.fast_branch(fast.transpose(1, 2))            # (B,f,T/2)
        x = torch.cat([x, xs, xf], dim=1)                      # (B,3f,T/2)

        x = self.drop(self.pool(self.block1(x)))               # (B,2f,T/4)
        x = self.drop(self.pool(self.block2(x)))               # (B,4f,T/8)
        x = self.drop(self.block3(x))                          # (B,8f,T/8)
        return x.amax(dim=2)                                   # GlobalMaxPool -> (B,8f)

    def forward(self, P: torch.Tensor) -> torch.Tensor:
        x = self.features(P)
        x = self.fc_drop(self.fc1(x))
        x = self.fc_drop(self.fc2(x))
        return self.classifier(x)

    # ------------------------------------------------------------------
    # Tiện ích cho finetune
    # ------------------------------------------------------------------
    def replace_head(self, num_classes: int) -> None:
        """Thay lớp phân loại cuối (ví dụ 21 lớp JHMDB -> số lớp cử chỉ của robot)."""
        self.classifier = nn.Linear(self.classifier.in_features, num_classes)
        self.hparams["num_classes"] = num_classes

    def freeze_backbone(self) -> None:
        """Đóng băng mọi lớp trừ đầu phân loại. BatchNorm của backbone cũng ngừng cập nhật."""
        for name, module in self.named_children():
            trainable = name in self.HEAD_MODULES
            for p in module.parameters():
                p.requires_grad = trainable
        self._backbone_frozen = True

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
        self._backbone_frozen = False

    def train(self, mode: bool = True):
        """
        Ghi đè train(): khi backbone đang đóng băng, các lớp BatchNorm/Dropout của backbone
        luôn ở chế độ eval. Nếu không, running_mean/var của BatchNorm vẫn bị cập nhật dù
        requires_grad=False, làm hỏng đặc trưng đã pretrain.
        """
        super().train(mode)
        if mode and getattr(self, "_backbone_frozen", False):
            for name, module in self.named_children():
                if name not in self.HEAD_MODULES:
                    module.eval()
        return self

    def num_parameters(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)


# ---------------------------------------------------------------------------
# Lưu / nạp
# ---------------------------------------------------------------------------
def save_checkpoint(path: str | Path, model: DDNet, meta: Optional[Dict] = None) -> None:
    """
    Lưu trọng số KÈM cấu hình. `meta` nên có: class_names, fps, T, pose_model,
    adapter_version... để gesture_recognizer.py lúc chạy thật đọc lại đúng như lúc train.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "hparams": model.hparams,
                "meta": meta or {}}, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> Tuple[DDNet, Dict]:
    """Tạo model đúng cấu hình đã lưu và nạp trọng số. Trả (model, meta)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model = DDNet(**ckpt["hparams"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt.get("meta", {})


def load_pretrained(model: DDNet, path: str | Path, map_location="cpu",
                    verbose: bool = True) -> Tuple[list, list]:
    """
    Nạp trọng số pretrain vào `model`, CHỈ những tensor trùng tên và trùng kích thước.
    Lớp không khớp (thường là classifier khi số lớp khác) giữ nguyên khởi tạo ngẫu nhiên.
    In rõ lớp nào nạp được, lớp nào bị bỏ qua. Trả (loaded, skipped).
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    src = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    dst = model.state_dict()
    loaded, skipped = [], []
    new_state = {}
    for k, v in dst.items():
        if k in src and src[k].shape == v.shape:
            new_state[k] = src[k]
            loaded.append(k)
        else:
            new_state[k] = v
            reason = "không có trong file" if k not in src else f"{tuple(src[k].shape)} != {tuple(v.shape)}"
            skipped.append(f"{k} ({reason})")
    model.load_state_dict(new_state)
    if verbose:
        print(f"Nạp pretrain từ {path}: {len(loaded)}/{len(dst)} tensor")
        for s in skipped:
            print("  bỏ qua:", s)
    return loaded, skipped


def _self_test() -> None:
    torch.manual_seed(0)
    for J in (15,):
        model = DDNet(num_classes=7, T=config.T, num_joints=J)
        x = torch.randn(4, config.T, J, 2)
        y = model(x)
        assert y.shape == (4, 7), y.shape
        print(f"J={J}: đầu ra {tuple(y.shape)}, {model.num_parameters():,} tham số")

    # JCD không đổi khi dịch / xoay / phóng to cả khung xương
    P = torch.randn(2, config.T, 15, 2)
    th = torch.tensor(0.7)
    R = torch.stack([torch.stack([th.cos(), -th.sin()]), torch.stack([th.sin(), th.cos()])])
    P2 = (P @ R.T) * 2.5 + 3.0
    diff = (joint_collection_distances(P) - joint_collection_distances(P2)).abs().max()
    print(f"JCD bất biến dịch/xoay/tỉ lệ: sai khác {diff:.2e}")

    # đóng băng: chỉ đầu phân loại được train
    model.freeze_backbone()
    model.train()
    print(f"Đóng băng: {model.num_parameters(True):,} tham số được train; "
          f"BN backbone ở eval = {not model.block1.c1.bn.training}")
    model.unfreeze()
    print("Hparams:", json.dumps(model.hparams))


if __name__ == "__main__":
    _self_test()