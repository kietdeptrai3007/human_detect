"""
train_ddnet.py - Train DD-Net trên khung xương RoCoG.

Ba chế độ:

  scratch   Train từ đầu chỉ bằng real_ground. Đây là ĐƯỜNG CƠ SỞ, luôn chạy trước.
              python train_ddnet.py --mode scratch --name scratch_f16

  pretrain  Train trên một nguồn dữ liệu khác để lấy trọng số khởi đầu:
              --pretrain-data syn    : video tổng hợp RoCoG (syn_ground); chỉ lấy các lớp có trong CLASS_NAMES (5 lớp).
                                       Cần chạy trước:
                                       python extract_keypoints.py --splits syn_train syn_val --max-per-class 500
              --pretrain-data jhmdb  : dữ liệu JHMDB có sẵn trong repo DD-Net
                                       (git clone https://github.com/fandulu/DD-Net,
                                        file DD-Net/data/JHMDB/GT_train_1.pkl)
              python train_ddnet.py --mode pretrain --pretrain-data syn --name pre_syn
              python train_ddnet.py --mode pretrain --pretrain-data jhmdb \
                     --jhmdb-dir DD-Net/data/JHMDB --name pre_jhmdb

  finetune  Nạp checkpoint pretrain, thay đầu phân loại thành số lớp của CLASS_NAMES (5 lớp), train trên real_ground
            theo 2 giai đoạn:
              1) đóng băng backbone, chỉ train đầu phân loại (--freeze-epochs, lr = --lr)
              2) mở băng toàn bộ, lr nhỏ hơn (--lr * --finetune-lr-mult)
              python train_ddnet.py --mode finetune --pretrained checkpoints/pre_syn/best.pt --name ft_syn

Mỗi epoch: train trên cửa sổ ngẫu nhiên (có augment), đánh giá VAL theo từng video
(trung bình xác suất nhiều cửa sổ). Lưu checkpoint tốt nhất theo độ chính xác val theo video
(hoà thì chọn val loss thấp hơn); dừng sớm khi val không cải thiện sau --patience epoch.

Kết quả trong checkpoints/<name>/:
  best.pt, last.pt     trọng số + cấu hình (T, fps, tên lớp...) để chạy thật đọc lại
  log.csv              loss/accuracy từng epoch
  args.json            toàn bộ tham số của lần chạy
  results_val.json     báo cáo chi tiết trên val của best.pt

Sau khi đã chốt cấu hình, có thể train lại trên train+val (204 video) với số epoch cố định:
    python train_ddnet.py --mode scratch --train-splits train_full --val-split none \
           --epochs <epoch tốt nhất> --name final_scratch
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from ddnet_model import DDNet, load_pretrained, save_checkpoint
from evaluate_ddnet import evaluate, print_report
from gesture_dataset import (Augmenter, Clip, TrainWindowDataset, class_weights,
                             describe, load_clips, load_jhmdb_clips,
                             load_split_clips, num_joints_of, worker_init_fn)
from rocog_labels import CLASS_NAMES, read_splits


# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def current_adapter_version() -> str:
    try:
        from extract_keypoints import adapter_version
        return adapter_version()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Chuẩn bị dữ liệu theo chế độ
# ---------------------------------------------------------------------------
def load_data(args) -> Tuple[List[Clip], List[Clip], List[str], str]:
    """Trả (clips_train, clips_val, class_names, window_mode)."""
    window_mode = args.window_mode

    if args.mode == "pretrain" and args.pretrain_data == "jhmdb":
        d = Path(args.jhmdb_dir)
        train_clips, names = load_jhmdb_clips(d / f"GT_train_{args.jhmdb_split}.pkl")
        val_clips, _ = load_jhmdb_clips(d / f"GT_test_{args.jhmdb_split}.pkl", names)
        print(f"JHMDB split {args.jhmdb_split}: {len(train_clips)} train, {len(val_clips)} val, "
              f"{len(names)} lớp")
        # clip JHMDB chỉ dài ~1 giây -> nén cả clip về T như code gốc
        return train_clips, val_clips, names, "whole"

    if args.mode == "pretrain" and args.pretrain_data == "syn":
        names = list(CLASS_NAMES)
        def existing(split):
            entries = read_splits([split], args.ann_dir)
            have = [e for e in entries if (Path(args.kp_dir) / e.npz_relpath()).exists()]
            print(f"{split}: {len(have)}/{len(entries)} video đã có keypoint")
            return have
        train_clips = load_clips(existing("syn_train"), args.kp_dir, args.fps, args.min_ok_ratio)
        # val: dùng video THẬT (val) để biết pretrain tổng hợp có giúp gì cho dữ liệu thật
        val_clips = [] if args.val_split == "none" else \
            load_split_clips([args.val_split], args.ann_dir, args.kp_dir, args.fps, args.min_ok_ratio)
        return train_clips, val_clips, names, window_mode

    names = list(CLASS_NAMES)
    train_clips = load_split_clips(args.train_splits, args.ann_dir, args.kp_dir, args.fps,
                                   args.min_ok_ratio)
    val_clips = [] if args.val_split == "none" else \
        load_split_clips([args.val_split], args.ann_dir, args.kp_dir, args.fps, args.min_ok_ratio)
    return train_clips, val_clips, names, window_mode


# ---------------------------------------------------------------------------
# Một epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model: DDNet, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    grad_clip: float) -> Tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / max(n, 1), correct / max(n, 1)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Train DD-Net cho cử chỉ RoCoG",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--mode", choices=["scratch", "pretrain", "finetune"], default="scratch")
    ap.add_argument("--name", default=None, help="tên thư mục kết quả trong checkpoints/")
    # dữ liệu
    ap.add_argument("--train-splits", nargs="+", default=["train"])
    ap.add_argument("--val-split", default="val", help="'none' = không dùng val")
    ap.add_argument("--ann-dir", default=str(config.ANNOTATION_DIR))
    ap.add_argument("--kp-dir", default=str(config.KEYPOINT_DIR))
    ap.add_argument("--out-dir", default=str(config.CHECKPOINT_DIR))
    ap.add_argument("--pretrain-data", choices=["syn", "jhmdb"], default="syn")
    ap.add_argument("--jhmdb-dir", default="DD-Net/data/JHMDB")
    ap.add_argument("--jhmdb-split", type=int, default=1, choices=[1, 2, 3])
    # thời gian / cửa sổ
    ap.add_argument("--fps", type=float, default=config.FPS_TARGET)
    ap.add_argument("--T", type=int, default=config.T)
    ap.add_argument("--window-mode", choices=["sliding", "whole"], default="sliding",
                    help="sliding = cửa sổ trượt (giống chạy thật); whole = nén cả clip về T (giống code gốc)")
    ap.add_argument("--trim-sec", type=float, default=config.TRIM_SEC)
    ap.add_argument("--eval-stride", type=int, default=config.EVAL_STRIDE)
    ap.add_argument("--min-ok-ratio", type=float, default=config.MIN_OK_RATIO)
    ap.add_argument("--samples-per-clip", type=int, default=config.SAMPLES_PER_CLIP)
    # model
    ap.add_argument("--filters", type=int, default=config.DDNET_FILTERS)
    ap.add_argument("--dropout", type=float, default=config.DDNET_DROPOUT)
    ap.add_argument("--fc-dropout", type=float, default=config.DDNET_FC_DROPOUT)
    ap.add_argument("--pretrained", default=None, help="checkpoint dùng cho --mode finetune")
    # tối ưu
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    ap.add_argument("--label-smoothing", type=float, default=config.LABEL_SMOOTHING)
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--patience", type=int, default=config.PATIENCE)
    ap.add_argument("--freeze-epochs", type=int, default=20, help="finetune: số epoch chỉ train đầu phân loại")
    ap.add_argument("--finetune-lr-mult", type=float, default=0.1, help="finetune: lr giai đoạn 2 = lr * hệ số này")
    # augment
    ap.add_argument("--use-mirror", action="store_true", default=config.USE_MIRROR)
    ap.add_argument("--no-aug", action="store_true", help="tắt toàn bộ augment (để so sánh)")
    # khác
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--workers", type=int, default=0, help="số tiến trình DataLoader")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--pose-model", default=config.POSE_MODEL_NAME)
    args = ap.parse_args()

    if args.mode == "finetune" and not args.pretrained:
        ap.error("--mode finetune cần --pretrained <checkpoint>")
    if args.name is None:
        args.name = f"{args.mode}_f{args.filters}_T{args.T}_fps{args.fps:g}"
    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device(args.device)

    # ---------------- dữ liệu ----------------
    train_clips, val_clips, class_names, window_mode = load_data(args)
    if not train_clips:
        raise SystemExit("Không có clip train. Kiểm tra --kp-dir và đã chạy extract_keypoints.py chưa.")
    describe("train", train_clips, class_names)
    if val_clips:
        describe("val", val_clips, class_names)
    C = len(class_names)
    J = num_joints_of(train_clips)

    augmenter = None if args.no_aug else Augmenter(mirror=args.use_mirror)
    train_ds = TrainWindowDataset(train_clips, args.T, args.samples_per_clip, args.trim_sec,
                                  speed=None if args.no_aug else config.AUG_SPEED,
                                  augmenter=augmenter, class_names=class_names,
                                  window_mode=window_mode, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=len(train_ds) > args.batch,
                              num_workers=args.workers, worker_init_fn=worker_init_fn,
                              pin_memory=device.type == "cuda")

    weight = None if args.no_class_weight else class_weights(train_clips, C)
    if weight is not None:
        print("Trọng số lớp:", ", ".join(f"{class_names[i]}={w:.2f}" for i, w in enumerate(weight.tolist())))
    criterion = nn.CrossEntropyLoss(weight=weight.to(device) if weight is not None else None,
                                    label_smoothing=args.label_smoothing)

    # ---------------- model ----------------
    if args.mode == "finetune":
        ck = torch.load(args.pretrained, map_location="cpu", weights_only=False)
        hp = ck["hparams"]
        if hp["T"] != args.T or hp["num_joints"] != J:
            raise SystemExit(f"Checkpoint pretrain có T={hp['T']}, J={hp['num_joints']} "
                             f"nhưng dữ liệu hiện tại T={args.T}, J={J}. Phải giống nhau.")
        args.filters = hp["filters"]
        model = DDNet(C, args.T, J, filters=hp["filters"], dropout=args.dropout, fc_dropout=args.fc_dropout)
        load_pretrained(model, args.pretrained)
    else:
        model = DDNet(C, args.T, J, filters=args.filters, dropout=args.dropout, fc_dropout=args.fc_dropout)
    model.to(device)
    print(f"\nDD-Net: {C} lớp, T={args.T}, J={J}, filters={args.filters}, "
          f"{model.num_parameters():,} tham số | thiết bị: {device}")
    print(f"Mỗi epoch: {len(train_ds)} cửa sổ train ({args.samples_per_clip}/clip), "
          f"{len(train_loader)} batch | cửa sổ: {window_mode}\n")

    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    meta_base = dict(class_names=class_names, fps=args.fps, T=args.T, num_joints=J,
                     window_mode=window_mode, eval_stride=args.eval_stride, trim_sec=args.trim_sec,
                     min_ok_ratio=args.min_ok_ratio, pose_model=args.pose_model,
                     adapter_version=current_adapter_version(), mode=args.mode,
                     train_splits=args.train_splits, pretrained=args.pretrained)

    # ---------------- các giai đoạn ----------------
    if args.mode == "finetune":
        stages = [dict(name="đóng băng", epochs=min(args.freeze_epochs, args.epochs), lr=args.lr,
                       freeze=True, early_stop=False),
                  dict(name="mở băng", epochs=max(args.epochs - args.freeze_epochs, 0),
                       lr=args.lr * args.finetune_lr_mult, freeze=False, early_stop=True)]
    else:
        stages = [dict(name="train", epochs=args.epochs, lr=args.lr, freeze=False, early_stop=True)]

    log_path = out_dir / "log.csv"
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["epoch", "stage", "lr", "train_loss", "train_acc_window",
                  "val_loss", "val_acc_video", "val_acc_window", "val_macro_f1", "seconds"])

    best_key: Optional[Tuple[float, float]] = None
    best_epoch = 0
    epoch = 0
    for st in stages:
        if st["epochs"] <= 0:
            continue
        if st["freeze"]:
            model.freeze_backbone()
        else:
            model.unfreeze()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=st["lr"], weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=st["lr"] / 100)
        print(f"--- Giai đoạn '{st['name']}': {st['epochs']} epoch, lr={st['lr']:g}, "
              f"{sum(p.numel() for p in params):,} tham số được train ---")
        since_best = 0

        for _ in range(st["epochs"]):
            epoch += 1
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, args.grad_clip)

            if val_clips:
                m = evaluate(model, val_clips, class_names, args.T, device, stride=args.eval_stride,
                             trim_sec=args.trim_sec, window_mode=window_mode, class_weight=weight)
                v_loss, v_acc, v_wacc, v_f1 = m["loss"], m["accuracy"], m["window_accuracy"], m["macro_f1"]
                scheduler.step(v_loss)
                key = (v_acc, -v_loss)
            else:
                v_loss = v_acc = v_wacc = v_f1 = float("nan")
                scheduler.step(tr_loss)
                key = (float(epoch), 0.0)       # không có val -> luôn lấy epoch mới nhất

            lr_now = optimizer.param_groups[0]["lr"]
            secs = time.time() - t0
            log.writerow([epoch, st["name"], f"{lr_now:.2e}", f"{tr_loss:.4f}", f"{tr_acc:.4f}",
                          f"{v_loss:.4f}", f"{v_acc:.4f}", f"{v_wacc:.4f}", f"{v_f1:.4f}", f"{secs:.1f}"])
            log_f.flush()

            improved = best_key is None or key > best_key
            if improved:
                best_key, best_epoch, since_best = key, epoch, 0
                save_checkpoint(out_dir / "best.pt", model,
                                dict(meta_base, epoch=epoch, val_accuracy=v_acc, val_loss=v_loss))
            else:
                since_best += 1
            save_checkpoint(out_dir / "last.pt", model, dict(meta_base, epoch=epoch, val_accuracy=v_acc))

            print(f"ep {epoch:4d} | lr {lr_now:.1e} | train loss {tr_loss:.3f} acc {tr_acc * 100:5.1f}% | "
                  f"val loss {v_loss:.3f} acc(video) {v_acc * 100:5.1f}% acc(cửa sổ) {v_wacc * 100:5.1f}% "
                  f"| {secs:.1f}s" + ("  *" if improved else ""), flush=True)

            if st["early_stop"] and val_clips and since_best >= args.patience:
                print(f"Dừng sớm: val không cải thiện sau {args.patience} epoch.")
                break
    log_f.close()

    # ---------------- báo cáo cuối ----------------
    print(f"\nCheckpoint tốt nhất: epoch {best_epoch} -> {out_dir / 'best.pt'}")
    if val_clips:
        from ddnet_model import load_checkpoint
        best, _ = load_checkpoint(out_dir / "best.pt", map_location=device)
        best.to(device)
        m = evaluate(best, val_clips, class_names, args.T, device, stride=args.eval_stride,
                     trim_sec=args.trim_sec, window_mode=window_mode)
        print_report(m, class_names, title=f"VAL - best.pt (epoch {best_epoch})")
        m.pop("_y_pred"), m.pop("_probs")
        m["best_epoch"] = best_epoch
        with open(out_dir / "results_val.json", "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)
    print(f"\nLog từng epoch: {log_path}")
    if args.mode != "pretrain":
        print(f"Khi đã chốt cấu hình, đánh giá test:\n"
              f"  python evaluate_ddnet.py --ckpt {out_dir / 'best.pt'} --split test")


if __name__ == "__main__":
    main()