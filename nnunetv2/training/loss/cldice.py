import torch
from torch import nn
import torch.nn.functional as F


class SoftCLDiceLoss(nn.Module):
    """
    Soft clDice loss as introduced for tubular structure segmentation.

    This implementation computes class-wise clDice on one-hot encoded probabilities,
    then averages across foreground classes.
    """

    def __init__(self,
                 apply_nonlin=None,
                 iter_: int = 3,
                 smooth: float = 1e-5,
                 include_background: bool = False):
        super().__init__()
        self.apply_nonlin = apply_nonlin
        self.iter_ = iter_
        self.smooth = smooth
        self.include_background = include_background

    @staticmethod
    def _soft_erode(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            p1 = -F.max_pool2d(-x, kernel_size=(3, 1), stride=1, padding=(1, 0))
            p2 = -F.max_pool2d(-x, kernel_size=(1, 3), stride=1, padding=(0, 1))
            return torch.minimum(p1, p2)
        if x.ndim == 5:
            p1 = -F.max_pool3d(-x, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
            p2 = -F.max_pool3d(-x, kernel_size=(1, 3, 1), stride=1, padding=(0, 1, 0))
            p3 = -F.max_pool3d(-x, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1))
            return torch.minimum(torch.minimum(p1, p2), p3)
        raise ValueError(f"Unsupported tensor rank for soft erosion: {x.ndim}")

    @staticmethod
    def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 4:
            return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        if x.ndim == 5:
            return F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
        raise ValueError(f"Unsupported tensor rank for soft dilation: {x.ndim}")

    def _soft_open(self, x: torch.Tensor) -> torch.Tensor:
        return self._soft_dilate(self._soft_erode(x))

    def _soft_skel(self, x: torch.Tensor) -> torch.Tensor:
        x_open = self._soft_open(x)
        skel = torch.relu(x - x_open)
        x_work = x

        for _ in range(self.iter_):
            x_work = self._soft_erode(x_work)
            x_open = self._soft_open(x_work)
            delta = torch.relu(x_work - x_open)
            skel = skel + torch.relu(delta - skel * delta)

        return skel

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        if self.apply_nonlin is not None:
            net_output = self.apply_nonlin(net_output)

        if target.shape[1] != 1:
            raise ValueError(
                f"SoftCLDiceLoss expects target with channel dim = 1 containing labels. Got shape {target.shape}"
            )

        labels = target[:, 0].long()
        num_classes = net_output.shape[1]

        labels = torch.clamp(labels, min=0, max=num_classes - 1)
        target_1h = F.one_hot(labels, num_classes=num_classes).movedim(-1, 1).to(net_output.dtype)

        if not self.include_background and num_classes > 1:
            pred = net_output[:, 1:]
            gt = target_1h[:, 1:]
        else:
            pred = net_output
            gt = target_1h

        if loss_mask is not None:
            if loss_mask.shape[1] != 1:
                raise ValueError(f"loss_mask must have a singleton channel dimension. Got {loss_mask.shape}")
            mask = loss_mask.to(net_output.dtype)
            pred = pred * mask
            gt = gt * mask

        skel_pred = self._soft_skel(pred)
        skel_gt = self._soft_skel(gt)

        dims = tuple(range(2, net_output.ndim))
        tprec = (torch.sum(skel_pred * gt, dim=dims) + self.smooth) / (torch.sum(skel_pred, dim=dims) + self.smooth)
        tsens = (torch.sum(skel_gt * pred, dim=dims) + self.smooth) / (torch.sum(skel_gt, dim=dims) + self.smooth)

        cl_dice = (2.0 * tprec * tsens + self.smooth) / (tprec + tsens + self.smooth)
        return 1.0 - cl_dice.mean()
