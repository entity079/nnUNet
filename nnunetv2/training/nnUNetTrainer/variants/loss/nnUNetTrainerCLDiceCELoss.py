import numpy as np
import torch

from nnunetv2.training.loss.compound_losses import CL_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1


class nnUNetTrainerCLDiceCELoss(nnUNetTrainer):
    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {
                    'batch_dice': self.configuration_manager.batch_dice,
                    'do_bg': True,
                    'smooth': 1e-5,
                    'ddp': self.is_ddp,
                },
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss,
            )
        else:
            loss = CL_and_CE_loss(
                cldice_kwargs={
                    'apply_nonlin': softmax_helper_dim1,
                    'iter_': 3,
                    'smooth': 1e-5,
                    'include_background': False,
                },
                ce_kwargs={},
                weight_ce=1,
                weight_cldice=1,
                ignore_label=self.label_manager.ignore_label,
            )

        if self._do_i_compile():
            if hasattr(loss, 'dc'):
                loss.dc = torch.compile(loss.dc)
            if hasattr(loss, 'cldice'):
                loss.cldice = torch.compile(loss.cldice)

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and not self._do_i_compile():
                weights[-1] = 1e-6
            else:
                weights[-1] = 0

            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        self.print_to_log_file("loss used is"loss,
                               also_print_to_console=True, add_timestamp=False)

        return loss
