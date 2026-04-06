"""
=============================================================
  AV Perception Lab — Custom Loss Functions
=============================================================

Three loss components are provided:
  1. FocalLoss          — down-weights easy examples, focuses on hard/rare classes
  2. WeightedCELoss     — classic CE with per-class frequency weights
  3. MultiTaskLoss      — combines both task losses with learnable uncertainty weights
                          (Liebel & Görner, 2018 — "Auxiliary Tasks in Multi-task Learning")

Class imbalance in GTSRB
--------------------------
Speed-limit sign distribution is heavily skewed:
  30 km/h : ~3000 samples   (most common)
  20 km/h : ~ 300 samples   (10× rarer)
  120 km/h: ~ 280 samples

Strategies used:
  - WeightedRandomSampler in the DataLoader (see dataset.py)
  - FocalLoss (γ=2) to dynamically downweight easy majorities
  - Optional class-frequency weights passed to CrossEntropyLoss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────
#  1. Focal Loss
# ─────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

    References
    ----------
    Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

    Parameters
    ----------
    gamma      : focusing parameter (0 = standard CE, 2 = recommended)
    alpha      : per-class weight tensor (optional, e.g. inverse frequency)
    reduction  : 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha          # (C,) tensor or None
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C)  — raw unnormalised scores
        targets : (B,)    — ground-truth class indices
        """
        # Standard per-sample log-probabilities
        log_probs = F.log_softmax(logits, dim=-1)           # (B, C)
        probs     = log_probs.exp()                          # (B, C)

        # Gather the log-prob and prob for the true class
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        pt     = probs.gather(1, targets.unsqueeze(1)).squeeze(1)      # (B,)

        # Focal modulation
        focal_weight = (1.0 - pt) ** self.gamma
        loss         = -focal_weight * log_pt               # (B,)

        # Optional class-level alpha weighting
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# ─────────────────────────────────────────────
#  2. Weighted Cross-Entropy Loss
# ─────────────────────────────────────────────
class WeightedCELoss(nn.Module):
    """
    Standard cross-entropy with static per-class frequency weights.
    Simpler than Focal Loss; useful as a baseline or for the TL task
    which has a less severe imbalance.

    Parameters
    ----------
    class_weights : (C,) tensor of per-class weights
    label_smoothing : avoids overconfident predictions (0.0 = off)
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        return F.cross_entropy(
            logits,
            targets,
            weight=weight,
            label_smoothing=self.label_smoothing,
        )


# ─────────────────────────────────────────────
#  3. Multi-Task Loss (Uncertainty Weighting)
# ─────────────────────────────────────────────
class MultiTaskLoss(nn.Module):
    """
    Combines per-task losses using learnable log-variance (σ²) parameters.

    L_total = Σ_i [ (1/2σ_i²) × L_i + log σ_i ]

    This approach automatically balances task contributions during training
    by learning the homoscedastic uncertainty for each task — when a task
    loss is large, the network learns to weight it down.

    Parameters
    ----------
    speed_loss_fn : loss module for speed classification
    tl_loss_fn    : loss module for traffic light detection
    speed_weight  : manual override for speed task (ignored if learnable=True)
    tl_weight     : manual override for TL task
    learnable     : if True, use uncertainty weighting (recommended)
    """

    def __init__(
        self,
        speed_loss_fn: nn.Module,
        tl_loss_fn: nn.Module,
        speed_weight: float = 1.0,
        tl_weight: float = 1.0,
        learnable: bool = True,
    ):
        super().__init__()
        self.speed_loss_fn = speed_loss_fn
        self.tl_loss_fn    = tl_loss_fn
        self.learnable     = learnable

        if learnable:
            # log(σ²) parameters — initialised near 0 (σ² ≈ 1)
            self.log_var_speed = nn.Parameter(torch.zeros(1))
            self.log_var_tl    = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("w_speed", torch.tensor(speed_weight))
            self.register_buffer("w_tl",    torch.tensor(tl_weight))

    def forward(
        self,
        speed_logits:  torch.Tensor,
        tl_logits:     torch.Tensor,
        speed_targets: torch.Tensor,
        tl_targets:    torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Returns a dict with individual task losses and the combined loss.
        """
        L_speed = self.speed_loss_fn(speed_logits, speed_targets)
        L_tl    = self.tl_loss_fn(tl_logits,    tl_targets)

        if self.learnable:
            # Uncertainty weighting
            prec_speed = torch.exp(-self.log_var_speed)   # 1/σ²
            prec_tl    = torch.exp(-self.log_var_tl)

            loss_combined = (
                prec_speed * L_speed + self.log_var_speed +
                prec_tl    * L_tl    + self.log_var_tl
            ).squeeze()
        else:
            loss_combined = self.w_speed * L_speed + self.w_tl * L_tl

        return {
            "loss":       loss_combined,
            "loss_speed": L_speed.detach(),
            "loss_tl":    L_tl.detach(),
        }


# ─────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────
def build_loss(
    speed_class_weights: Optional[torch.Tensor] = None,
    tl_class_weights:    Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    learnable_weights: bool = True,
) -> MultiTaskLoss:
    """
    Recommended setup:
      - FocalLoss (γ=2) for speed classification (high imbalance)
      - WeightedCELoss for traffic light (moderate imbalance)
      - Learnable uncertainty weighting to balance both tasks

    Pass class_weights derived from dataset.class_weights() for best results.
    """
    speed_loss = FocalLoss(
        gamma=focal_gamma,
        alpha=speed_class_weights,
        reduction="mean",
    )
    tl_loss = WeightedCELoss(
        class_weights=tl_class_weights,
        label_smoothing=0.05,
    )
    return MultiTaskLoss(
        speed_loss_fn=speed_loss,
        tl_loss_fn=tl_loss,
        learnable=learnable_weights,
    )
