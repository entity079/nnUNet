from typing import Tuple, Union, List

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader


class nnUNetDataLoaderMixup(nnUNetDataLoader):
    def __init__(
            self,
            *args,
            mixup_probability: float = 0.2,
            mixup_alpha: float = 0.2,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mixup_probability = mixup_probability
        self.mixup_alpha = mixup_alpha

    def _sample_case_patch(self, case_identifier: str, force_fg: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        data, seg, seg_prev, properties = self._data.load_case(case_identifier)
        shape = data.shape[1:]

        bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
        bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

        data_cropped = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
        seg_cropped = torch.from_numpy(crop_and_pad_nd(seg, bbox, -1)).to(torch.int16)
        if seg_prev is not None:
            seg_prev_cropped = torch.from_numpy(crop_and_pad_nd(seg_prev, bbox, -1)).to(torch.int16)
            seg_cropped = torch.cat((seg_cropped, seg_prev_cropped[None]), dim=0)

        if self.patch_size_was_2d:
            data_cropped = data_cropped[:, 0]
            seg_cropped = seg_cropped[:, 0]

        return data_cropped, seg_cropped

    def _apply_transforms(self, data_cropped: torch.Tensor, seg_cropped: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, List[torch.Tensor]]]:
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
                    data_primary, seg_primary = self._sample_case_patch(i, force_fg)
                    data_primary, seg_primary = self._apply_transforms(data_primary, seg_primary)

                    if apply_mixup:
                        partner_identifier = self.indices[np.random.choice(len(self.indices))]
                        data_partner, seg_partner = self._sample_case_patch(partner_identifier, False)
                        data_partner, seg_partner = self._apply_transforms(data_partner, seg_partner)
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
