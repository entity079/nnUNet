import numpy as np

from nnunetv2.training.loss.compound_losses import CL_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerCLDiceCELoss(nnUNetTrainer):
    def _build_loss(self):
        if self.label_manager.has_regions:
            raise RuntimeError('nnUNetTrainerCLDiceCELoss is not implemented for region-based training.')

        loss = CL_and_CE_loss(
            cldice_kwargs={
                'iter_': 3,
                'smooth': 1e-5,
                'include_background': False,
            },
            ce_kwargs={},
            weight_ce=1,
            weight_cldice=1,
            ignore_label=self.label_manager.ignore_label,
        )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
            # this gives higher resolution outputs more weight in the loss
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0

            # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
            weights = weights / weights.sum()
            # now wrap the loss
            loss = DeepSupervisionWrapper(loss, weights)

        return loss
