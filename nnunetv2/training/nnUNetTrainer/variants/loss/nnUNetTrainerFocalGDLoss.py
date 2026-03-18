import numpy as np

from nnunetv2.training.loss.compound_losses import GDL_and_BCE_focal_loss, GDL_and_Focal_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerFocalGDLoss(nnUNetTrainer):
    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = GDL_and_BCE_focal_loss(
                {'alpha': 0.25, 'gamma': 2.0, 'reduction': 'none'},
                {'batch_dice': self.configuration_manager.batch_dice,
                 'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                weight_focal=1,
                weight_gdl=1,
                use_ignore_label=self.label_manager.ignore_label is not None
            )
        else:
            loss = GDL_and_Focal_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'do_bg': False, 'smooth': 1e-5, 'ddp': self.is_ddp},
                {'gamma': 2.0},
                weight_focal=1,
                weight_gdl=1,
                ignore_label=self.label_manager.ignore_label
            )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()

            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss
