from typing import List, Tuple, Union

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader


class nnUNetDataLoaderMosaic(nnUNetDataLoader):
    def __init__(self, *args, mosaic_probability: float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.mosaic_probability = mosaic_probability

    @staticmethod
    def _get_slice_length(slc: slice) -> int:
        return slc.stop - slc.start

    def _get_bbox_for_patch_size(
            self,
            data_shape: np.ndarray,
            force_fg: bool,
            class_locations: Union[dict, None],
            patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
            overwrite_class: Union[int, Tuple[int, ...]] = None,
            verbose: bool = False,
    ):
        final_patch_size = np.array(self.final_patch_size)
        need_to_pad = np.maximum(0, (np.array(patch_size) - final_patch_size).astype(int))
        dim = len(data_shape)

        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < patch_size[d]:
                need_to_pad[d] = patch_size[d] - data_shape[d]

        lbs = [- need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - patch_size[i] for i in range(dim)]

        if not force_fg and not self.has_ignore:
            bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
        else:
            if not force_fg and self.has_ignore:
                selected_class = self.annotated_classes_key
                if len(class_locations[selected_class]) == 0:
                    selected_class = None
            elif force_fg:
                assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
                if overwrite_class is not None:
                    assert overwrite_class in class_locations.keys(), 'desired class ("overwrite_class") does not ' \
                                                                      'have class_locations (missing key)'

                eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]
                tmp = [i == self.annotated_classes_key if isinstance(i, tuple) else False for i in eligible_classes_or_regions]
                if any(tmp) and len(eligible_classes_or_regions) > 1:
                    eligible_classes_or_regions.pop(np.where(tmp)[0][0])

                if len(eligible_classes_or_regions) == 0:
                    selected_class = None
                    if verbose:
                        print('case does not contain any foreground classes')
                else:
                    selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))] if \
                        (overwrite_class is None or overwrite_class not in eligible_classes_or_regions) else overwrite_class
            else:
                raise RuntimeError('lol what!?')

            if selected_class is not None:
                voxels_of_that_class = class_locations[selected_class]
                selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
                bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - patch_size[i] // 2) for i in range(dim)]
            else:
                bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        bbox_ubs = [bbox_lbs[i] + patch_size[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs

    def _sample_case_patch(
            self,
            case_identifier: str,
            force_fg: bool,
            patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        effective_patch_size = patch_size if not self.patch_size_was_2d else (1, *patch_size)
        data, seg, seg_prev, properties = self._data.load_case(case_identifier)
        shape = data.shape[1:]

        bbox_lbs, bbox_ubs = self._get_bbox_for_patch_size(
            shape,
            force_fg,
            properties['class_locations'],
            np.array(effective_patch_size),
        )
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

    @staticmethod
    def _sample_split_point(axis_length: int) -> int:
        if axis_length <= 1:
            return axis_length
        low = max(1, int(round(axis_length * 0.4)))
        high = min(axis_length - 1, int(round(axis_length * 0.6)))
        if low > high:
            return axis_length // 2
        return int(np.random.randint(low, high + 1))

    def _build_mosaic_sample(self, primary_case_identifier: str, force_fg: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_shape = tuple(self.patch_size[1:] if self.patch_size_was_2d else self.patch_size)
        # For 3D patches we mosaic in-plane and keep the remaining axis intact.
        mosaic_axes = tuple(range(max(0, len(spatial_shape) - 2), len(spatial_shape)))
        split_points = {axis: self._sample_split_point(spatial_shape[axis]) for axis in mosaic_axes}

        tile_slices = []
        first_axis = mosaic_axes[0]
        second_axis = mosaic_axes[1]
        for first_is_upper in (True, False):
            for second_is_left in (True, False):
                current_slices = []
                for axis, axis_length in enumerate(spatial_shape):
                    if axis == first_axis:
                        split = split_points[axis]
                        current_slices.append(slice(0, split) if first_is_upper else slice(split, axis_length))
                    elif axis == second_axis:
                        split = split_points[axis]
                        current_slices.append(slice(0, split) if second_is_left else slice(split, axis_length))
                    else:
                        current_slices.append(slice(0, axis_length))
                tile_slices.append(tuple(current_slices))

        reference_data, reference_seg = self._sample_case_patch(primary_case_identifier, force_fg, spatial_shape)
        mosaic_data = torch.empty_like(reference_data)
        mosaic_seg = torch.empty_like(reference_seg)

        for tile_idx, spatial_slices in enumerate(tile_slices):
            tile_case_identifier = primary_case_identifier if tile_idx == 0 else self.indices[np.random.choice(len(self.indices))]
            tile_shape = tuple(self._get_slice_length(slc) for slc in spatial_slices)
            tile_data, tile_seg = self._sample_case_patch(tile_case_identifier, force_fg if tile_idx == 0 else False, tile_shape)

            target_slices = (slice(None), *spatial_slices)
            mosaic_data[target_slices] = tile_data
            mosaic_seg[target_slices] = tile_seg

        return mosaic_data, mosaic_seg

    def generate_train_batch(self):
        selected_keys = self.get_indices()
        data_all = torch.empty(self.data_shape, dtype=torch.float32)
        seg_all = None

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, i in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)
                    use_mosaic = np.random.uniform() < self.mosaic_probability

                    if use_mosaic:
                        data_cropped, seg_cropped = self._build_mosaic_sample(i, force_fg)
                    else:
                        data_cropped, seg_cropped = self._sample_case_patch(
                            i,
                            force_fg,
                            self.patch_size[1:] if self.patch_size_was_2d else self.patch_size,
                        )

                    if self.transforms is not None:
                        transformed = self.transforms(**{'image': data_cropped, 'segmentation': seg_cropped})
                        data_sample = transformed['image']
                        seg_sample = transformed['segmentation']
                    else:
                        data_sample = data_cropped
                        seg_sample = seg_cropped

                    data_all[j] = data_sample

                    if isinstance(seg_sample, list):
                        if seg_all is None:
                            seg_all = [torch.empty((self.batch_size, *s.shape), dtype=s.dtype) for s in seg_sample]
                        for s_idx, s in enumerate(seg_sample):
                            seg_all[s_idx][j] = s
                    else:
                        if seg_all is None:
                            seg_all = torch.empty((self.batch_size, *seg_sample.shape), dtype=seg_sample.dtype)
                        seg_all[j] = seg_sample

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}
