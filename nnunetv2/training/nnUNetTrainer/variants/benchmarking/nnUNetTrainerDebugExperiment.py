import json
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, save_json
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class _DebugComposeTransforms(ComposeTransforms):
    def __init__(self, transforms: List[BasicTransform], owner: "nnUNetTrainerDebugExperiment"):
        super().__init__(transforms)
        self.owner = owner

    def apply(self, data_dict, **params):
        record = {
            'before_augmentation': self.owner._summarize_data_dict(data_dict),
            'applied_transforms': []
        }
        current = data_dict
        for transform in self.transforms:
            current = transform(**current)
            record['applied_transforms'].append({
                'transform': type(transform).__name__,
                'state': self.owner._summarize_data_dict(current)
            })
        record['after_augmentation'] = self.owner._summarize_data_dict(current)
        self.owner._queue_transform_record(record)
        return current


class nnUNetTrainerDebugExperiment(nnUNetTrainer):
    """
    One-epoch debugging trainer that writes batch-wise experiment artifacts so you can inspect
    augmentations, inputs, outputs, activation summaries and gradients.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1
        self.disable_checkpointing = True
        self.debug_tensor_preview_values = 32
        self.debug_parameter_preview_values = 8
        self._debug_artifact_root = None
        self._transform_records = deque()
        self._debug_train_batch_counter = 0
        self._hook_handles = []
        self._current_activation_records = {}
        self._current_activation_gradient_records = {}

    def _do_i_compile(self):
        return False

    def initialize(self):
        super().initialize()
        self._register_debug_hooks()

    def _register_debug_hooks(self):
        if self._hook_handles:
            return

        network = self.network.module if self.is_ddp else self.network
        leaf_modules = [(name, module) for name, module in network.named_modules() if len(list(module.children())) == 0]
        for name, module in leaf_modules:
            handle = module.register_forward_hook(self._make_activation_hook(name))
            self._hook_handles.append(handle)

    def _make_activation_hook(self, module_name: str):
        def hook(_module, _inputs, output):
            self._current_activation_records[module_name] = self._summarize_nested_tensors(output)
            self._attach_gradient_hooks(module_name, output)

        return hook

    def _attach_gradient_hooks(self, module_name: str, output: Any, suffix: str = ''):
        if isinstance(output, torch.Tensor):
            if output.requires_grad:
                key = f'{module_name}{suffix}'
                output.register_hook(lambda grad, hook_key=key: self._store_activation_gradient(hook_key, grad))
        elif isinstance(output, (list, tuple)):
            for idx, item in enumerate(output):
                self._attach_gradient_hooks(module_name, item, f'{suffix}[{idx}]')
        elif isinstance(output, dict):
            for key, item in output.items():
                self._attach_gradient_hooks(module_name, item, f'{suffix}.{key}')

    def _store_activation_gradient(self, key: str, grad: torch.Tensor):
        self._current_activation_gradient_records[key] = self._summarize_tensor(grad)

    def _queue_transform_record(self, record: Dict[str, Any]):
        self._transform_records.append(record)

    def _pop_transform_records(self, num_records: int) -> List[Dict[str, Any]]:
        popped = []
        for _ in range(min(num_records, len(self._transform_records))):
            popped.append(self._transform_records.popleft())
        return popped

    def _clear_transform_records(self):
        self._transform_records.clear()

    def _summarize_scalar(self, value: Union[float, int]) -> float:
        return float(value)

    def _summarize_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]:
        tensor_cpu = tensor.detach().cpu()
        tensor_float = tensor_cpu.float()
        flat = tensor_float.reshape(-1)
        preview = flat[:self.debug_tensor_preview_values].tolist()
        summary = {
            'shape': list(tensor_cpu.shape),
            'dtype': str(tensor_cpu.dtype),
            'numel': int(tensor_cpu.numel()),
            'preview': preview,
        }
        if tensor_cpu.numel() > 0:
            summary.update({
                'min': self._summarize_scalar(tensor_float.min().item()),
                'max': self._summarize_scalar(tensor_float.max().item()),
                'mean': self._summarize_scalar(tensor_float.mean().item()),
                'std': self._summarize_scalar(tensor_float.std(unbiased=False).item()),
            })
        return summary

    def _summarize_nested_tensors(self, obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return self._summarize_tensor(obj)
        if isinstance(obj, list):
            return [self._summarize_nested_tensors(i) for i in obj]
        if isinstance(obj, tuple):
            return [self._summarize_nested_tensors(i) for i in obj]
        if isinstance(obj, dict):
            return {k: self._summarize_nested_tensors(v) for k, v in obj.items()}
        return repr(obj)

    def _summarize_data_dict(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        return {key: self._summarize_nested_tensors(value) for key, value in data_dict.items()}

    def _summarize_parameter_gradients(self) -> Dict[str, Any]:
        parameter_gradients = {}
        for name, parameter in self.network.named_parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().cpu().float()
            flat = grad.reshape(-1)
            parameter_gradients[name] = {
                'shape': list(grad.shape),
                'norm': self._summarize_scalar(torch.linalg.vector_norm(flat).item()),
                'mean': self._summarize_scalar(flat.mean().item()),
                'std': self._summarize_scalar(flat.std(unbiased=False).item()),
                'preview': flat[:self.debug_parameter_preview_values].tolist()
            }
        return parameter_gradients

    def _get_debug_artifact_root(self) -> str:
        if self._debug_artifact_root is None:
            self._debug_artifact_root = join(self.output_folder, 'experiment_debug')
            maybe_mkdir_p(self._debug_artifact_root)
        return self._debug_artifact_root

    def _write_json(self, filename: str, content: Dict[str, Any]):
        with open(filename, 'w') as f:
            json.dump(content, f, indent=2)

    def _serialize_training_transforms(self, transform: BasicTransform) -> Dict[str, Any]:
        if isinstance(transform, _DebugComposeTransforms):
            return {
                'pipeline_class': type(transform).__name__,
                'transforms': [repr(t) for t in transform.transforms]
            }
        return {'pipeline_class': type(transform).__name__, 'repr': repr(transform)}

    def get_dataloaders(self):
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)

        mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
        mt_gen_val = SingleThreadedAugmenter(dl_val, None)

        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        self._clear_transform_records()
        return mt_gen_train, mt_gen_val

    def get_training_transforms(
            self,
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        base_transform = super().get_training_transforms(
            patch_size=patch_size,
            rotation_for_DA=rotation_for_DA,
            deep_supervision_scales=deep_supervision_scales,
            mirror_axes=mirror_axes,
            do_dummy_2d_data_aug=do_dummy_2d_data_aug,
            use_mask_for_norm=use_mask_for_norm,
            is_cascaded=is_cascaded,
            foreground_labels=foreground_labels,
            regions=regions,
            ignore_label=ignore_label,
        )
        if isinstance(base_transform, ComposeTransforms):
            return _DebugComposeTransforms(base_transform.transforms, self)
        return base_transform

    def on_train_start(self):
        super().on_train_start()
        debug_root = self._get_debug_artifact_root()
        save_json(self._serialize_training_transforms(self.dataloader_train.data_loader.transforms),
                  join(debug_root, 'training_transforms.json'), sort_keys=False)

    def on_train_end(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        super().on_train_end()

    def train_step(self, batch: dict) -> dict:
        self._current_activation_records = {}
        self._current_activation_gradient_records = {}

        data = batch['data']
        target = batch['target']
        keys = deepcopy(batch.get('keys', []))
        input_summary = self._summarize_tensor(data)
        target_summary = self._summarize_nested_tensors(target)
        augmentation_records = self._pop_transform_records(len(keys))

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        output = self.network(data)
        loss = self.loss(output, target)
        loss.backward()
        total_grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
        parameter_gradient_summary = self._summarize_parameter_gradients()
        self.optimizer.step()

        debug_root = self._get_debug_artifact_root()
        epoch_dir = join(debug_root, f'epoch_{self.current_epoch:04d}')
        maybe_mkdir_p(epoch_dir)
        artifact_file = join(epoch_dir, f'batch_{self._debug_train_batch_counter:04d}.json')
        self._write_json(artifact_file, {
            'epoch': self.current_epoch,
            'batch_index': self._debug_train_batch_counter,
            'keys': keys,
            'inputs': input_summary,
            'targets': target_summary,
            'augmentations': augmentation_records,
            'outputs': self._summarize_nested_tensors(output),
            'loss': self._summarize_scalar(loss.detach().cpu().item()),
            'loss_backpropagated': True,
            'total_gradient_norm_before_clipping': self._summarize_scalar(total_grad_norm.detach().cpu().item()),
            'activation_summaries': self._current_activation_records,
            'activation_gradient_summaries': self._current_activation_gradient_records,
            'parameter_gradient_summaries': parameter_gradient_summary,
        })
        self.print_to_log_file(f'Debug artifact written to {artifact_file}', also_print_to_console=False)
        self._debug_train_batch_counter += 1
        return {'loss': loss.detach().cpu().numpy()}
