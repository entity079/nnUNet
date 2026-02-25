from typing import Callable

import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn
import logging



from typing import Callable
import torch
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
from torch import nn
import logging


class SoftFNRLoss(nn.Module):

    def __init__(self,
                 apply_nonlin: Callable = None,
                 batch_dice: bool = False,
                 do_bg: bool = True,
                 smooth: float = 1.,
                 ddp: bool = True):

        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if x.ndim != y.ndim:
            y = y.view((y.shape[0], 1, *y.shape[1:]))

        if x.shape != y.shape:
            y_onehot = torch.zeros_like(x, dtype=torch.float32)
            y_onehot.scatter_(1, y.long(), 1)
        else:
            y_onehot = y.float()

        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]

        axes = [0] + list(range(2, x.ndim)) if self.batch_dice else list(range(2, x.ndim))

        if loss_mask is not None:
            mask = torch.tile(loss_mask, (1, x.shape[1], *[1]*(x.ndim-2)))
            x = x * mask
            y_onehot = y_onehot * mask

        tp = (x * y_onehot).sum(dim=axes, dtype=torch.float32)
        fn = ((1 - x) * y_onehot).sum(dim=axes, dtype=torch.float32)

        if self.batch_dice and self.ddp:
            tp = AllGatherGrad.apply(tp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        fnr = (fn + self.smooth) / (tp + fn + self.smooth).clamp_min(1e-8)

        fnr = fnr.mean()

        logging.info(f"FNR computed: {fnr.item():.6f}")
        return fnr

class MemoryEfficientSoftFNRLoss(nn.Module):

    def __init__(self,
                 apply_nonlin: Callable = None,
                 batch_dice: bool = False,
                 do_bg: bool = True,
                 smooth: float = 1.,
                 ddp: bool = True):

        super().__init__()
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if x.ndim != y.ndim:
            y = y.view((y.shape[0], 1, *y.shape[1:]))

        if x.shape != y.shape:
            y_onehot = torch.zeros_like(x, dtype=torch.float32)
            y_onehot.scatter_(1, y.long(), 1)
        else:
            y_onehot = y.float()

        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]

        axes = tuple(range(2, x.ndim))

        if loss_mask is None:
            tp = (x * y_onehot).sum(axes, dtype=torch.float32)
            fn = ((1 - x) * y_onehot).sum(axes, dtype=torch.float32)
        else:
            tp = (x * y_onehot * loss_mask).sum(axes, dtype=torch.float32)
            fn = ((1 - x) * y_onehot * loss_mask).sum(axes, dtype=torch.float32)

        if self.batch_dice:
            if self.ddp:
                tp = AllGatherGrad.apply(tp).sum(0)
                fn = AllGatherGrad.apply(fn).sum(0)

            tp = tp.sum(0)
            fn = fn.sum(0)

        fnr = (fn + self.smooth) / (tp + fn + self.smooth).clamp_min(1e-8)

        fnr = fnr.mean()

        logging.info(f"FNR computed: {fnr.item():.6f}")
        return fnr
def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):

    if axes is None:
        axes = tuple(range(2, net_output.ndim))

    with torch.no_grad():
        if net_output.ndim != gt.ndim:
            gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

        if net_output.shape == gt.shape:
            y_onehot = gt.float()
        else:
            y_onehot = torch.zeros_like(net_output, dtype=torch.float32)
            y_onehot.scatter_(1, gt.long(), 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        mask = torch.tile(mask, (1, tp.shape[1], *[1]*(tp.ndim-2)))
        tp *= mask
        fp *= mask
        fn *= mask
        tn *= mask

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = tp.sum(dim=axes, dtype=torch.float32)
        fp = fp.sum(dim=axes, dtype=torch.float32)
        fn = fn.sum(dim=axes, dtype=torch.float32)
        tn = tn.sum(dim=axes, dtype=torch.float32)

    return tp, fp, fn, tn

if __name__ == '__main__':
    from nnunetv2.utilities.helpers import softmax_helper_dim1
    pred = torch.rand((2, 3, 32, 32, 32))
    ref = torch.randint(0, 3, (2, 32, 32, 32))

    dl_old = SoftfnRLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    dl_new = MemoryEfficientSoftfnRLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=0, ddp=False)
    res_old = dl_old(pred, ref)
    res_new = dl_new(pred, ref)
    print(res_old, res_new)
