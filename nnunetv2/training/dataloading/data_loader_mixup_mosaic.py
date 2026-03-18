from typing import List, Tuple, Union

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader_mosaic import nnUNetDataLoaderMosaic


class nnUNetDataLoaderMixupMosaic(nnUNetDataLoaderMosaic):
    def __init__(
            self,
            *args,
            mosaic_probability: float = 0.2,
            mixup_probability: float = 0.2,
            mixup_alpha: float = 0.2,
            **kwargs,
    ):
        super().__init__(*args, mosaic_probability=mosaic_probability, **kwargs)
        self.mixup_probability = mixup_probability
        self.mixup_alpha = mixup_alpha

    def _apply_transforms(
            self,
            data_cropped: torch.Tensor,
            seg_cropped: torch.Tensor,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]]:
        if self.transforms is None:
            return data_cropped, seg_cropped

        transformed = self.transforms(**{'image': data_cropped, 'segmentation': seg_cropped})
        return transformed['image'], transformed['segmentation']

    @staticmethod
    def _allocate_or_store_seg(seg_all, seg_sample, batch_size: int, batch_idx: int):
        if isinstance(seg_sample, list):
            if seg_all is None:
                seg_all = [torch.empty((batch_size, *s.shape), dtype=s.dtype) for s in seg_sample]
            for s_idx, s in enumerate(seg_sample):
                seg_all[s_idx][batch_idx] = s
        else:
            if seg_all is None:
                seg_all = torch.empty((batch_size, *seg_sample.shape), dtype=seg_sample.dtype)
            seg_all[batch_idx] = seg_sample
        return seg_all

    def _sample_augmented_case(self, case_identifier: str, force_fg: bool):
        use_mosaic = np.random.uniform() < self.mosaic_probability
        if use_mosaic:
            data_cropped, seg_cropped = self._build_mosaic_sample(case_identifier, force_fg)
        else:
            data_cropped, seg_cropped = self._sample_case_patch(
                case_identifier,
                force_fg,
                self.patch_size[1:] if self.patch_size_was_2d else self.patch_size,
            )
        return self._apply_transforms(data_cropped, seg_cropped)

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = torch.empty(self.data_shape, dtype=torch.float32)
        seg_all = None
        mixup_seg_all = None

        apply_mixup = np.random.uniform() < self.mixup_probability
        mixup_lambda = 1.0
        if apply_mixup:
            mixup_lambda = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
            mixup_lambda = max(mixup_lambda, 1.0 - mixup_lambda)

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, i in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)
                    data_primary, seg_primary = self._sample_augmented_case(i, force_fg)

                    if apply_mixup:
                        partner_identifier = self.indices[np.random.choice(len(self.indices))]
                        data_partner, seg_partner = self._sample_augmented_case(partner_identifier, False)
                        data_sample = mixup_lambda * data_primary + (1.0 - mixup_lambda) * data_partner
                    else:
                        seg_partner = seg_primary
                        data_sample = data_primary

                    data_all[j] = data_sample
                    seg_all = self._allocate_or_store_seg(seg_all, seg_primary, self.batch_size, j)
                    mixup_seg_all = self._allocate_or_store_seg(mixup_seg_all, seg_partner, self.batch_size, j)

        return {
            'data': data_all,
            'target': seg_all,
            'mixup_target': mixup_seg_all,
            'mixup_lambda': torch.tensor(mixup_lambda, dtype=torch.float32),
            'keys': selected_keys,
        }
