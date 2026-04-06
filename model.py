"""
=============================================================
  AV Perception Lab — Multi-Task CNN
  Backbone : MobileNetV2 (pre-trained ImageNet, frozen)
  Shared   : GAP → Dropout(0.3) → FC(1280→256) → BN → ReLU
  Task 1   : Speed Limit Classification — FC(256→43)  [GTSRB full]
  Task 2   : Traffic Light State        — FC(256→4)   [Red/Amber/Green/Off]

  Parameters (approx)
  ───────────────────
    Total          : ~3.5 M
    Frozen backbone: ~3.4 M
    Trainable heads: ~140 K
  Input size       : 224 × 224 × 3 (RGB)
=============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class AVPerceptionNet(nn.Module):
    """
    MobileNetV2 dual-head architecture for autonomous vehicle perception.

    Args
    ----
    num_speed_classes : int   GTSRB classes (default 43 — full dataset)
    num_light_classes : int   Traffic light states: Red/Amber/Green/Off (default 4)
    freeze_backbone   : bool  Freeze backbone weights at init (default True)
    """

    # Traffic light state labels (index → name)
    LIGHT_STATES = ["Red", "Amber", "Green", "Off"]

    def __init__(
        self,
        num_speed_classes: int = 43,
        num_light_classes: int = 4,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # Freeze all backbone params (fine-tune later if needed)
        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False

        # Feature extractor: all layers except the original classifier
        self.features = backbone.features          # output: (B, 1280, 7, 7) @ 224×224

        # ── Global Average Pool ───────────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d((1, 1))   # → (B, 1280, 1, 1)

        # ── Shared trunk ──────────────────────────────────────────
        # Dropout → FC 256 → BatchNorm → ReLU
        self.trunk = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # ── Task heads ────────────────────────────────────────────
        # Speed: FC 256 → 43  (GTSRB, weighted CE loss recommended)
        self.speed_head = nn.Linear(256, num_speed_classes)

        # Traffic light: FC 256 → 4  (Focal loss recommended)
        self.light_head = nn.Linear(256, num_light_classes)

    # ── Forward ───────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        x : (B, 3, 224, 224)  — ImageNet-normalised RGB tensor

        Returns
        -------
        speed_logits : (B, num_speed_classes)
        light_logits : (B, num_light_classes)
        """
        x = self.features(x)          # (B, 1280, 7, 7)
        x = self.pool(x).flatten(1)   # (B, 1280)
        x = self.trunk(x)             # (B, 256)
        return self.speed_head(x), self.light_head(x)

    # ── Inference helper ──────────────────────────────────────────
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict:
        """Returns softmax probabilities and argmax class indices."""
        self.eval()
        speed_logits, light_logits = self(x)
        return {
            "speed_probs":  F.softmax(speed_logits, dim=-1),
            "light_probs":  F.softmax(light_logits, dim=-1),
            "speed_class":  speed_logits.argmax(dim=-1),
            "light_class":  light_logits.argmax(dim=-1),
        }

    # ── Unfreeze backbone for full fine-tuning ────────────────────
    def unfreeze_backbone(self, lr_scale: float = 0.1) -> list[dict]:
        """
        Call after warm-up epochs to unfreeze backbone layers.
        Returns param groups suitable for an optimiser with scaled LR.

        Usage
        -----
            optimizer.add_param_group(
                {'params': model.features.parameters(), 'lr': 3e-5}
            )
        """
        for p in self.features.parameters():
            p.requires_grad = True
        return [
            {"params": self.features.parameters(), "lr_scale": lr_scale},
        ]


# ─────────────────────────────────────────────
#  Sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AVPerceptionNet(num_speed_classes=43, num_light_classes=4).to(device)

    dummy              = torch.randn(4, 3, 224, 224, device=device)
    speed_out, tl_out  = model(dummy)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("✅ Model sanity check passed")
    print(f"   Speed logits : {speed_out.shape}   → 43 GTSRB classes")
    print(f"   TL logits    : {tl_out.shape}    → 4 states (R/A/G/Off)")
    print(f"   Total params : {total:,}")
    print(f"   Trainable    : {trainable:,}  (heads only, backbone frozen)")
