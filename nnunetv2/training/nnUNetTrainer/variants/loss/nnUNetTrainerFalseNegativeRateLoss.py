import numpy as np
import torch
from torch import nn

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from nnunetv2.utilities.helpers import softmax_helper_dim1


class FalseNegativeRateLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1e-5,
                 ddp: bool = True):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        axes = [0] + list(range(2, x.ndim)) if self.batch_dice else list(range(2, x.ndim))
        tp, _, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        if self.batch_dice and self.ddp:
            tp = AllGatherGrad.apply(tp).sum(0, dtype=torch.float32)
            fn = AllGatherGrad.apply(fn).sum(0, dtype=torch.float32)

        if not self.do_bg:
            if self.batch_dice:
                tp = tp[1:]
                fn = fn[1:]
            else:
                tp = tp[:, 1:]
                fn = fn[:, 1:]

        fnr = (fn + self.smooth) / (tp + fn + self.smooth).clamp_min(1e-8)
        return fnr.mean()


class nnUNetTrainerFalseNegativeRateLoss(nnUNetTrainer):
    def _build_loss(self):
        loss = FalseNegativeRateLoss(
            batch_dice=self.configuration_manager.batch_dice,
            do_bg=self.label_manager.has_regions,
            smooth=1e-5,
            ddp=self.is_ddp,
            apply_nonlin=torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
