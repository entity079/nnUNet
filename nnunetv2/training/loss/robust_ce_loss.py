import torch
from torch import nn, Tensor
import numpy as np


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()
class FocalLoss(nn.CrossEntropyLoss):
    """
    Multi-class focal loss operating on logits.
    """
    def __init__(self, weight=None, ignore_index: int = -100, gamma: float = 2.0,
                 reduction: str = 'mean', label_smoothing: float = 0):
        super().__init__(weight=weight, ignore_index=ignore_index, reduction='none',
                         label_smoothing=label_smoothing)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long()

        ce_loss = super().forward(input, target)
        valid_mask = target != self.ignore_index

        safe_target = torch.where(valid_mask, target, 0)
        pt = torch.softmax(input, dim=1).gather(1, safe_target.unsqueeze(1)).squeeze(1)
        focal_term = torch.pow(1 - pt, self.gamma)
        loss = focal_term * ce_loss
        loss = loss[valid_mask]

        if loss.numel() == 0:
            return input.new_tensor(0.)
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        return loss.mean()


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        target = target.float()
        bce_loss = nn.functional.binary_cross_entropy_with_logits(input, target, reduction='none')
        probs = torch.sigmoid(input)
        pt = probs * target + (1 - probs) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        loss = alpha_t * torch.pow(1 - pt, self.gamma) * bce_loss

        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        return loss.mean()
