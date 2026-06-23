import torch
import torch.nn as nn
from typing import Optional


class TemporalCoherenceLoss(nn.Module):
    """
    Compute temporal coherence loss for a sequence of frames.

    Loss = weight * (magnitude_loss + delta_weight * delta_loss)

    - magnitude_loss: penalizes large frame-to-frame changes (||f_t - f_{t-1}||)
    - delta_loss: when ground-truth sequence provided, penalizes difference between
                  predicted and ground-truth frame deltas: ||(Δpred - Δgt)||

    Supports 'l2' (MSE), 'l1' (MAE) and 'huber' (smooth L1) pointwise penalties.
    Accepts pred_seq and gt_seq shaped either [B, T, H, W] or [B, T, C, H, W].
    """

    def __init__(self,
                 weight: float = 1.0,
                 delta_weight: float = 1.0,
                 loss_type: str = "l2",
                 huber_delta: float = 1.0):
        super().__init__()
        assert loss_type in ("l2", "l1", "huber")
        self.weight = weight
        self.delta_weight = delta_weight
        self.loss_type = loss_type
        self.huber_delta = huber_delta

    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        # mean over all elements
        return x.mean()

    def _pointwise_loss(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        diff = a - b
        if self.loss_type == "l2":
            return self._reduce(diff.pow(2))
        if self.loss_type == "l1":
            return self._reduce(diff.abs())
        # huber / smooth L1
        absd = diff.abs()
        cond = absd <= self.huber_delta
        loss = torch.where(cond, 0.5 * diff.pow(2), self.huber_delta * (absd - 0.5 * self.huber_delta))
        return self._reduce(loss)

    def forward(self,
                pred_seq: torch.Tensor,
                gt_seq: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        pred_seq: [B, T, H, W] or [B, T, C, H, W]
        gt_seq:   optional, same shape as pred_seq (used to compute delta_loss)
        returns: scalar loss tensor
        """
        if pred_seq.dim() not in (4, 5):
            raise ValueError(f"pred_seq must be 4D or 5D (B,T,H,W or B,T,C,H,W), got {pred_seq.shape}")

        # Ensure channel dimension for uniform handling: [B, T, C, H, W]
        if pred_seq.dim() == 4:
            pred = pred_seq.unsqueeze(2)
        else:
            pred = pred_seq

        if gt_seq is not None:
            if gt_seq.dim() == 4:
                gt = gt_seq.unsqueeze(2)
            else:
                gt = gt_seq
            if gt.shape != pred.shape:
                raise ValueError(f"gt_seq shape {gt.shape} != pred_seq shape {pred.shape}")
        else:
            gt = None

        # temporal differences: Δ_t = f_t - f_{t-1}, shape [B, T-1, C, H, W]
        d_pred = pred[:, 1:] - pred[:, :-1]

        # magnitude loss: penalize large deltas (encourages smoothness)
        magnitude_loss = self._pointwise_loss(d_pred, torch.zeros_like(d_pred))

        # delta loss: if ground truth diffs available, penalize difference between predicted & ground-truth deltas
        delta_loss = torch.tensor(0.0, device=pred.device)
        if gt is not None:
            d_gt = gt[:, 1:] - gt[:, :-1]
            delta_loss = self._pointwise_loss(d_pred, d_gt)

        loss = self.weight * (magnitude_loss + self.delta_weight * delta_loss)
        return loss
