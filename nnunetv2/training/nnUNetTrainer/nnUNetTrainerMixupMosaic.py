import importlib
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from nnunetv2.training.dataloading.data_loader_mixup_mosaic import nnUNetDataLoaderMixupMosaic


class _FakeDataset:
    def __init__(self, shape=(8, 8, 8), num_cases: int = 4):
        self.identifiers = [f'case_{i}' for i in range(num_cases)]
        self._shape = shape

    def load_case(self, identifier):
        case_idx = int(identifier.split('_')[-1])
        data = np.full((1, *self._shape), fill_value=float(case_idx), dtype=np.float32)
        seg = np.full((1, *self._shape), fill_value=case_idx, dtype=np.int16)
        properties = {
            'class_locations': {
                1: np.array([[1, self._shape[0] // 2, self._shape[1] // 2, self._shape[2] // 2]], dtype=np.int64)
            }
        }
        return data, seg, None, properties


class _FakeLabelManager:
    all_labels = [0, 1]
    has_ignore_label = False
    foreground_labels = (1,)
    has_regions = False
    foreground_regions = None
    ignore_label = None


class TestMixupMosaicDataloader(unittest.TestCase):
    def test_generate_train_batch_combines_augmented_samples_with_mixup(self):
        dataset = _FakeDataset()
        loader = nnUNetDataLoaderMixupMosaic(
            dataset,
            batch_size=1,
            patch_size=(8, 8, 8),
            final_patch_size=(8, 8, 8),
            label_manager=_FakeLabelManager(),
            oversample_foreground_percent=0.0,
            transforms=None,
            mosaic_probability=1.0,
            mixup_probability=1.0,
            mixup_alpha=0.2,
        )
        loader.get_indices = lambda: ['case_0']

        primary = (
            torch.full((1, 8, 8, 8), 2.0, dtype=torch.float32),
            torch.zeros((1, 8, 8, 8), dtype=torch.int16),
        )
        partner = (
            torch.full((1, 8, 8, 8), 4.0, dtype=torch.float32),
            torch.ones((1, 8, 8, 8), dtype=torch.int16),
        )

        with patch.object(loader, '_sample_augmented_case', side_effect=[primary, partner]), \
                patch('numpy.random.beta', return_value=0.25), \
                patch('numpy.random.choice', return_value=1):
            batch = loader.generate_train_batch()

        self.assertEqual(batch['data'].shape, (1, 1, 8, 8, 8))
        self.assertAlmostEqual(float(batch['mixup_lambda']), 0.75, places=6)
        self.assertTrue(torch.allclose(batch['data'], torch.full((1, 1, 8, 8, 8), 2.5)))
        self.assertTrue(torch.equal(batch['target'], torch.zeros((1, 1, 8, 8, 8), dtype=torch.int16)))
        self.assertTrue(torch.equal(batch['mixup_target'], torch.ones((1, 1, 8, 8, 8), dtype=torch.int16)))

    def test_trainer_get_dataloaders_smoke(self):
        dataset = _FakeDataset()
        trainer_module_name = 'nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerMixupMosaic'
        base_module_name = 'nnunetv2.training.nnUNetTrainer.nnUNetTrainer'

        fake_base_module = ModuleType(base_module_name)
        fake_base_module.nnUNetTrainer = type('nnUNetTrainer', (), {})

        class DummyTrainer:
            dataset_class = object
            batch_size = 1
            oversample_foreground_percent = 0.0
            probabilistic_oversampling = False
            is_cascaded = False
            device = torch.device('cpu')
            mixup_probability = 1.0
            mixup_alpha = 0.2
            mosaic_probability = 1.0
            configuration_manager = SimpleNamespace(
                patch_size=(8, 8, 8),
                use_mask_for_norm=None,
            )
            label_manager = _FakeLabelManager()

            def _get_deep_supervision_scales(self):
                return None

            def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
                return None, False, (8, 8, 8), (0, 1, 2)

            def get_training_transforms(self, *args, **kwargs):
                return None

            def get_validation_transforms(self, *args, **kwargs):
                return None

            def get_tr_and_val_datasets(self):
                return dataset, dataset

        trainer = DummyTrainer()

        with patch.dict(sys.modules, {base_module_name: fake_base_module}):
            for module_name in (
                    'nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerMixup',
                    trainer_module_name,
            ):
                sys.modules.pop(module_name, None)
            trainer_module = importlib.import_module(trainer_module_name)

            with patch.object(trainer_module, 'get_allowed_n_proc_DA', return_value=0):
                mt_gen_train, mt_gen_val = trainer_module.nnUNetTrainerMixupMosaic.get_dataloaders(trainer)

        train_batch = next(mt_gen_train)
        val_batch = next(mt_gen_val)

        self.assertEqual(train_batch['data'].shape, (1, 1, 8, 8, 8))
        self.assertIn('mixup_target', train_batch)
        self.assertEqual(val_batch['data'].shape, (1, 1, 8, 8, 8))


if __name__ == '__main__':
    unittest.main()
