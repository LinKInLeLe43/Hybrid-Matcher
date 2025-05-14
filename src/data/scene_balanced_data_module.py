from typing import Callable, List

import numpy as np
from joblib import Parallel, delayed
from pytorch_lightning import LightningDataModule
from rich.progress import track
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
)

from ..utils import get_pylogger
from .components.samplers import SceneBalancedSampler

log = get_pylogger(__name__)


class SceneBalancedDataModule(LightningDataModule):
    def __init__(
        self,
        train_scene_list_path: str,
        train_num_samples_per_scene: int,
        train_batch_size_per_device: int,
        train_dataset_wrapper: Callable[[str], Dataset],
        val_scene_list_path: str,
        val_dataset_wrapper: Callable[[str], Dataset],
        test_scene_list_path: str,
        test_dataset_wrapper: Callable[[str], Dataset],
        num_workers: int = 0,
        pin_memory: bool = False,
        seed: int = 66,
        parallel: bool = False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _split_by_rank(self, names: List[str]) -> List[str]:
        new_names = np.random.RandomState(self.hparams.seed).permutation(names)
        rest_count = len(names) % self.trainer.world_size
        if rest_count != 0:
            padding_count = self.trainer.world_size - rest_count
            padding_names = np.random.RandomState(self.hparams.seed).choice(
                names, size=padding_count
            )
            new_names = np.concatenate([new_names, padding_names])

        count_per_rank = len(new_names) // self.trainer.world_size
        start_idx = self.trainer.global_rank * count_per_rank
        end_idx = start_idx + count_per_rank
        names_per_rank = list(new_names[start_idx:end_idx])
        return names_per_rank

    def _build_concat_dataset(
        self,
        dataset_wrapper: Callable[[str], Dataset],
        scene_list_path: str,
        use_for_training: bool,
    ) -> ConcatDataset:
        with open(scene_list_path) as f:
            npz_names = [
                f"{name}.npz" if not name.endswith(".npz") else name
                for name in f.read().splitlines()
            ]
        if use_for_training:
            npz_names = self._split_by_rank(npz_names)
        log.info(f"{len(npz_names)} scenes assigned per rank.")

        npz_names = track(
            npz_names,
            description="Loading scenes...",
            disable=self.trainer.global_rank != 0,
        )
        if self.hparams.parallel:
            datasets = Parallel(n_jobs=self.hparams.num_workers)(
                delayed(dataset_wrapper)(name) for name in npz_names
            )
        else:
            datasets = [dataset_wrapper(name) for name in npz_names]
        dataset = ConcatDataset(datasets)
        return dataset

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = self._build_concat_dataset(
                self.hparams.train_dataset_wrapper,
                self.hparams.train_scene_list_path,
                True,
            )
            self.val_dataset = self._build_concat_dataset(
                self.hparams.val_dataset_wrapper,
                self.hparams.val_scene_list_path,
                False,
            )
        elif stage == "test":
            self.test_dataset = self._build_concat_dataset(
                self.hparams.test_dataset_wrapper,
                self.hparams.test_scene_list_path,
                False,
            )

    def train_dataloader(self) -> DataLoader:
        sampler = SceneBalancedSampler(
            self.train_dataset,
            self.hparams.train_num_samples_per_scene,
            seed=self.hparams.seed,
        )
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size_per_device,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
        log.info(
            "Train `Sampler` and `DataLoader` created. "
            "(should not re-create between epochs)"
        )
        return dataloader

    def val_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self.val_dataset, shuffle=False)
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
        log.info("Validation `Sampler` and `DataLoader` created.")
        return dataloader

    def test_dataloader(self) -> DataLoader:
        sampler = DistributedSampler(self.test_dataset, shuffle=False)
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
        log.info("Test `Sampler` and `DataLoader` created.")
        return dataloader
