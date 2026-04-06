"""
=============================================================
  AV Perception Lab — Training Loop
=============================================================

Features
────────
  • Cosine Annealing LR schedule with warm restarts
  • Mixed precision training (torch.autocast / GradScaler)
  • Gradient clipping to prevent exploding gradients
  • Early stopping with patience
  • Per-task accuracy tracking
  • Checkpoint saving (best model by combined val accuracy)
  • TensorBoard-compatible metric logging via tqdm + dict
=============================================================
"""

import time
import math
from pathlib import Path
from typing import Optional
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from model   import AVPerceptionNet
from loss    import build_loss
from dataset import build_dataloaders, GTSRBSpeedDataset, TrafficLightDataset


# ─────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────

def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


class EarlyStopping:
    """Stops training when monitored metric stops improving."""

    def __init__(self, patience: int = 8, min_delta: float = 1e-4, mode: str = "max"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode  # 'max' for accuracy, 'min' for loss
        self.best      = float("-inf") if mode == "max" else float("inf")
        self.counter   = 0
        self.triggered = False

    def step(self, value: float) -> bool:
        """Returns True if training should stop."""
        improved = (
            (value > self.best + self.min_delta)
            if self.mode == "max"
            else (value < self.best - self.min_delta)
        )
        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


# ─────────────────────────────────────────────
#  Config dataclass (plain dict-based)
# ─────────────────────────────────────────────

DEFAULT_CFG = {
    # Data
    "speed_root"        : "data/GTSRB",
    "tl_root"           : "data/TrafficLight",
    "batch_size"        : 32,
    "num_workers"       : 4,
    # Model
    "num_speed_classes" : 8,
    "tl_classes"        : 4,
    "freeze_epochs"     : 3,           # freeze backbone for N epochs, then unfreeze
    # Optimiser
    "lr"                : 3e-4,
    "weight_decay"      : 1e-4,
    "grad_clip"         : 1.0,
    # Scheduler
    "T_0"               : 10,          # CosineAnnealingWarmRestarts period
    "T_mult"            : 2,
    # Loss
    "focal_gamma"       : 2.0,
    # Training
    "epochs"            : 40,
    "early_stop_patience": 8,
    # Misc
    "save_dir"          : "checkpoints",
    "amp"               : True,        # mixed-precision
    "seed"              : 42,
}


# ─────────────────────────────────────────────
#  Training step (one batch)
# ─────────────────────────────────────────────

def train_step(
    model:       AVPerceptionNet,
    speed_batch: Optional[dict],
    tl_batch:    Optional[dict],
    criterion:   nn.Module,
    optimizer:   torch.optim.Optimizer,
    scaler:      GradScaler,
    device:      torch.device,
    cfg:         dict,
) -> dict:
    model.train()

    # ── Concatenate available batches into one forward pass ────────
    # In practice both are always provided, but guard for robustness
    assert speed_batch is not None or tl_batch is not None

    speed_imgs, speed_tgts = None, None
    tl_imgs,    tl_tgts    = None, None

    if speed_batch:
        speed_imgs = speed_batch["image"].to(device, non_blocking=True)
        speed_tgts = speed_batch["label"].to(device, non_blocking=True)

    if tl_batch:
        tl_imgs = tl_batch["image"].to(device, non_blocking=True)
        tl_tgts = tl_batch["label"].to(device, non_blocking=True)

    # Merge: stack along batch dim for a single forward pass
    # Both tasks share the backbone — processing together is efficient
    if speed_imgs is not None and tl_imgs is not None:
        imgs = torch.cat([speed_imgs, tl_imgs], dim=0)
    elif speed_imgs is not None:
        imgs = speed_imgs
    else:
        imgs = tl_imgs

    optimizer.zero_grad(set_to_none=True)

    with autocast(enabled=cfg["amp"]):
        out = model(imgs)

        bsz_s = speed_imgs.size(0) if speed_imgs is not None else 0
        speed_logits_out = out["speed"][:bsz_s]
        tl_logits_out    = out["traffic_light"][bsz_s:]

        loss_dict = criterion(
            speed_logits_out, speed_tgts,
            tl_logits_out,    tl_tgts,
        )
        loss = loss_dict["loss"]

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
    scaler.step(optimizer)
    scaler.update()

    return {
        "loss":       loss.item(),
        "loss_speed": loss_dict["loss_speed"].item(),
        "loss_tl":    loss_dict["loss_tl"].item(),
        "acc_speed":  accuracy(speed_logits_out, speed_tgts) if bsz_s > 0 else 0.0,
        "acc_tl":     accuracy(tl_logits_out,    tl_tgts)    if tl_imgs is not None else 0.0,
    }


# ─────────────────────────────────────────────
#  Validation
# ─────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:     AVPerceptionNet,
    speed_dl:  DataLoader,
    tl_dl:     DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    cfg:       dict,
) -> dict:
    model.eval()
    metrics = defaultdict(list)

    # Validate each task independently for clean reporting
    for batch in speed_dl:
        imgs  = batch["image"].to(device)
        tgts  = batch["label"].to(device)
        dummy_tl_logits = torch.zeros(imgs.size(0), cfg["tl_classes"], device=device)
        dummy_tl_tgts   = torch.zeros(imgs.size(0), dtype=torch.long,  device=device)

        with autocast(enabled=cfg["amp"]):
            out  = model(imgs)
            ld   = criterion(out["speed"], tgts, dummy_tl_logits, dummy_tl_tgts)

        metrics["val_loss_speed"].append(ld["loss_speed"].item())
        metrics["val_acc_speed"].append(accuracy(out["speed"], tgts))

    for batch in tl_dl:
        imgs  = batch["image"].to(device)
        tgts  = batch["label"].to(device)
        dummy_sp_logits = torch.zeros(imgs.size(0), cfg["num_speed_classes"], device=device)
        dummy_sp_tgts   = torch.zeros(imgs.size(0), dtype=torch.long,         device=device)

        with autocast(enabled=cfg["amp"]):
            out = model(imgs)
            ld  = criterion(dummy_sp_logits, dummy_sp_tgts, out["traffic_light"], tgts)

        metrics["val_loss_tl"].append(ld["loss_tl"].item())
        metrics["val_acc_tl"].append(accuracy(out["traffic_light"], tgts))

    return {k: sum(v) / len(v) for k, v in metrics.items()}


# ─────────────────────────────────────────────
#  Main Training Loop
# ─────────────────────────────────────────────

def train(cfg: dict = DEFAULT_CFG):
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  AV Perception — Training on {device}")
    print(f"{'='*55}\n")

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets & Loaders ────────────────────────────────────────
    loaders = build_dataloaders(
        speed_root=cfg["speed_root"],
        tl_root=cfg["tl_root"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
    )

    # Class weights for loss
    speed_ds = GTSRBSpeedDataset(cfg["speed_root"], split="train")
    tl_ds    = TrafficLightDataset(cfg["tl_root"],  split="train")

    # ── Model ─────────────────────────────────────────────────────
    model = AVPerceptionNet(
        num_speed_classes=cfg["num_speed_classes"],
        tl_classes=cfg["tl_classes"],
        freeze_backbone=(cfg["freeze_epochs"] > 0),
    ).to(device)

    # ── Loss ──────────────────────────────────────────────────────
    criterion = build_loss(
        speed_class_weights=speed_ds.class_weights(),
        tl_class_weights=tl_ds.class_weights(),
        focal_gamma=cfg["focal_gamma"],
        learnable_weights=True,
    ).to(device)

    # ── Optimiser ─────────────────────────────────────────────────
    # Include criterion params (learnable σ) in optimiser
    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer  = torch.optim.AdamW(
        all_params,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.999),
    )

    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg["T_0"],
        T_mult=cfg["T_mult"],
        eta_min=1e-6,
    )

    scaler    = GradScaler(enabled=cfg["amp"])
    stopper   = EarlyStopping(patience=cfg["early_stop_patience"], mode="max")
    best_acc  = 0.0
    history   = []

    # ── Epoch loop ────────────────────────────────────────────────
    for epoch in range(1, cfg["epochs"] + 1):

        # Phase 2: unfreeze backbone after warm-up
        if epoch == cfg["freeze_epochs"] + 1:
            for p in model.features.parameters():
                p.requires_grad = True
            # Lower LR for fine-tuning
            for pg in optimizer.param_groups:
                pg["lr"] = cfg["lr"] * 0.1
            print(f"\n  [Epoch {epoch}] Backbone unfrozen — LR reduced to {cfg['lr']*0.1:.2e}")

        # ── Train ──────────────────────────────────────────────────
        train_metrics = defaultdict(float)
        n_steps = min(len(loaders["speed"]["train"]), len(loaders["tl"]["train"]))

        speed_iter = iter(loaders["speed"]["train"])
        tl_iter    = iter(loaders["tl"]["train"])

        pbar = tqdm(range(n_steps), desc=f"Epoch {epoch:02d}", leave=False)
        for _ in pbar:
            sb = next(speed_iter, None)
            tb = next(tl_iter,    None)
            if sb is None or tb is None:
                break

            step_m = train_step(model, sb, tb, criterion, optimizer, scaler, device, cfg)
            for k, v in step_m.items():
                train_metrics[k] += v

            pbar.set_postfix(
                loss=f"{step_m['loss']:.3f}",
                sp=f"{step_m['acc_speed']:.2%}",
                tl=f"{step_m['acc_tl']:.2%}",
            )

        scheduler.step(epoch)

        for k in train_metrics:
            train_metrics[k] /= n_steps

        # ── Validate ───────────────────────────────────────────────
        val_m = validate(
            model,
            loaders["speed"]["val"],
            loaders["tl"]["val"],
            criterion, device, cfg,
        )

        avg_acc = (val_m["val_acc_speed"] + val_m["val_acc_tl"]) / 2.0

        row = {"epoch": epoch, **dict(train_metrics), **val_m, "avg_val_acc": avg_acc}
        history.append(row)

        # ── Print ──────────────────────────────────────────────────
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:02d}/{cfg['epochs']}  "
            f"LR={lr_now:.2e}  "
            f"TrainLoss={train_metrics['loss']:.4f}  "
            f"Val[Speed={val_m['val_acc_speed']:.2%}  TL={val_m['val_acc_tl']:.2%}  Avg={avg_acc:.2%}]"
        )

        # ── Checkpoint ────────────────────────────────────────────
        if avg_acc > best_acc:
            best_acc = avg_acc
            ckpt = {
                "epoch"     : epoch,
                "model"     : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "criterion" : criterion.state_dict(),
                "scheduler" : scheduler.state_dict(),
                "best_acc"  : best_acc,
                "cfg"       : cfg,
            }
            torch.save(ckpt, save_dir / "best_model.pt")
            print(f"  ✅ Saved best model  (avg_val_acc={best_acc:.4f})")

        # ── Early stopping ────────────────────────────────────────
        if stopper.step(avg_acc):
            print(f"\n  ⏹  Early stopping at epoch {epoch}  (best={best_acc:.4f})")
            break

    print(f"\n  Training complete — best avg val acc = {best_acc:.4f}")
    return history


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Override config here or via argparse / hydra
    cfg = {**DEFAULT_CFG}
    # Example: quick smoke-test with tiny data
    # cfg["epochs"] = 2
    # cfg["batch_size"] = 8
    history = train(cfg)
