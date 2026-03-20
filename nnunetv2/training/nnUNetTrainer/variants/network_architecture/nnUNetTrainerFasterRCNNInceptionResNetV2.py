import numpy as np
import torch
from torch.optim import Adam

from nnunetv2.training.loss.compound_losses import GDL_and_BCE_focal_loss, GDL_and_Focal_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerMixupMosaic import nnUNetTrainerMixupMosaic
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.faster_rcnn_inception_resnet_v2 import \
    FasterRCNNInceptionResNetV2SegmentationModel


class nnUNetTrainerFasterRCNNInceptionResNetV2(nnUNetTrainerMixupMosaic):
    mixup_probability: float = 0.4
    mixup_alpha: float = 0.2
    mosaic_probability: float = 0.4

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 3e-4
        self.weight_decay = 1e-5
        self.maximum_batch_size = 2
        self.maximum_num_stages = 5
        self.maximum_features_per_stage = 256
        self.maximum_backbone_channels = 64

    def _do_i_compile(self):
        return False

    def _set_batch_size_and_oversample(self):
        super()._set_batch_size_and_oversample()
        self.batch_size = min(self.batch_size, self.maximum_batch_size)

    def _get_deep_supervision_scales(self):
        scales = super()._get_deep_supervision_scales()
        if scales is None:
            return None
        return scales[:self.maximum_num_stages - 1]

    def configure_optimizers(self):
        optimizer = Adam(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

    def _get_focal_class_weights(self):
        configured_weights = self.dataset_json.get('focal_loss_class_weights') or self.dataset_json.get('class_weights')
        if configured_weights is not None:
            weights = torch.as_tensor(configured_weights, dtype=torch.float32, device=self.device)
        else:
            num_classes = len(self.label_manager.all_labels)
            weights = torch.ones(num_classes, dtype=torch.float32, device=self.device)
            if num_classes > 1:
                weights[1:] = 2.0
        return weights

    def _build_loss(self):
        if self.label_manager.has_regions:
            loss = GDL_and_BCE_focal_loss(
                {'alpha': 0.25, 'gamma': 2.0, 'reduction': 'none'},
                {'batch_dice': self.configuration_manager.batch_dice,
                 'do_bg': True, 'smooth': 1e-5, 'ddp': self.is_ddp},
                weight_focal=1,
                weight_gdl=1,
                use_ignore_label=self.label_manager.ignore_label is not None,
            )
        else:
            focal_weights = self._get_focal_class_weights()
            loss = GDL_and_Focal_loss(
                {'batch_dice': self.configuration_manager.batch_dice,
                 'do_bg': False, 'smooth': 1e-5, 'ddp': self.is_ddp},
                {'gamma': 2.0, 'weight': focal_weights},
                weight_focal=1,
                weight_gdl=1,
                ignore_label=self.label_manager.ignore_label,
            )

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss.to(self.device)

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True):
        del architecture_class_name, arch_init_kwargs_req_import
        arch_init_kwargs = dict(arch_init_kwargs)
        features_per_stage = arch_init_kwargs.get('features_per_stage')
        if features_per_stage is not None:
            features_per_stage = [min(int(i), 256) for i in features_per_stage[:5]]
            arch_init_kwargs['features_per_stage'] = features_per_stage

            for key in ('kernel_sizes', 'strides', 'n_blocks_per_stage', 'n_conv_per_stage'):
                if key in arch_init_kwargs and arch_init_kwargs[key] is not None:
                    arch_init_kwargs[key] = arch_init_kwargs[key][:len(features_per_stage)]

            arch_init_kwargs['n_blocks_per_stage'] = [1] * len(features_per_stage)

        arch_init_kwargs['backbone_channels'] = min(int(arch_init_kwargs.get('backbone_channels', 64)), 64)
        return FasterRCNNInceptionResNetV2SegmentationModel(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            **arch_init_kwargs,
        )
