import functools
from os import path
from typing import List

import joblib
import numpy as np
from numpy import random
import pytorch_lightning as pl
from rich import progress
from torch.utils import data

from src import utils
from src.data import utils as data_utils


log = utils.get_pylogger(__name__)


class MatchingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_npz_root: str,
        train_list_path: str,
        train_dataset: functools.partial,
        train_sampler: functools.partial,
        train_batch_size_per_gpu: int,
        val_npz_root: str,
        val_list_path: str,
        val_dataset: functools.partial,
        val_sampler: functools.partial,
        test_npz_root: str,
        test_list_path: str,
        test_dataset: functools.partial,
        test_sampler: functools.partial,
        workers_count: int,
        pin_memory: bool = True,
        seed: int = 66,
        parallel: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _create_concat_dataset(
        self,
        dataset: functools.partial,
        npz_paths: List[str]
    ) -> data.ConcatDataset:
        with progress.Progress(disable=self.trainer.global_rank != 0) as p:
            if self.hparams.parallel:
                p.add_task("Loading scenes...", total=len(npz_paths))
                with data_utils.rich_joblib(p):
                    parallel = joblib.Parallel(
                        n_jobs=self.hparams.workers_count)
                    datasets = parallel(joblib.delayed(dataset)(path)
                                        for path in npz_paths)
            else:
                npz_paths = p.track(npz_paths, description="Loading scenes...")
                datasets = [dataset(path) for path in npz_paths]
        dataset = data.ConcatDataset(datasets)
        return dataset

    def _split_names_per_rank(self, names: List[str]) -> List[str]:
        new_names = random.RandomState(self.hparams.seed).permutation(names)
        rest_count = len(names) % self.trainer.world_size
        if rest_count != 0:
            padding_count = self.trainer.world_size - rest_count
            padding_names = random.RandomState(self.hparams.seed).choice(
                names, size=padding_count)
            new_names = np.concatenate([new_names, padding_names])

        count_per_rank = len(new_names) // self.trainer.world_size
        start_idx = self.trainer.global_rank * count_per_rank
        end_idx = start_idx + count_per_rank
        names_per_rank = list(new_names[start_idx:end_idx])
        return names_per_rank

    def _create_npz_paths(
        self,
        npz_root: str,
        scene_list_path: str,
        split: bool
    ) -> List[str]:
        with open(scene_list_path, "r") as f:
            names = [name for name in f.read().splitlines()]
        if split:
            names = self._split_names_per_rank(names)
        log.info(f"{len(names)} scenes assigned per rank.")

        npz_paths = []
        for name in names:
            if path.splitext(name)[1] != ".npz":
                name += ".npz"
            npz_paths.append(path.join(npz_root, name))
        return npz_paths

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_npz_paths = self._create_npz_paths(
                self.hparams.train_npz_root, self.hparams.train_list_path, True)
            self.train_dataset = self._create_concat_dataset(
                self.hparams.train_dataset, train_npz_paths)
            val_npz_paths = self._create_npz_paths(
                self.hparams.val_npz_root, self.hparams.val_list_path, False)
            self.val_dataset = self._create_concat_dataset(
                self.hparams.val_dataset, val_npz_paths)
            log.info(f"Train and validation `Dataset`s created.")

        if stage == "test":
            test_npz_paths = self._create_npz_paths(
                self.hparams.test_npz_root, self.hparams.test_list_path, False)
            self.test_dataset = self._create_concat_dataset(
                self.hparams.test_dataset, test_npz_paths)
            log.info(f"Test `Dataset` created.")

    def train_dataloader(self) -> data.DataLoader:
        dataloader = data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.train_batch_size_per_gpu,
            sampler=self.hparams.train_sampler(self.train_dataset),
            num_workers=self.hparams.workers_count,
            pin_memory=self.hparams.pin_memory)
        log.info("Train `Sampler` and `DataLoader` created. "
                 "(should not re-create between epochs)")
        return dataloader

    def val_dataloader(self) -> data.DataLoader:
        dataloader = data.DataLoader(
            self.val_dataset, batch_size=1,
            sampler=self.hparams.val_sampler(self.val_dataset),
            num_workers=self.hparams.workers_count,
            pin_memory=self.hparams.pin_memory)
        log.info("Validation `Sampler` and `DataLoader` created.")
        return dataloader

    def test_dataloader(self) -> data.DataLoader:
        dataloader = data.DataLoader(
            self.test_dataset, batch_size=1,
            sampler=self.hparams.test_sampler(self.test_dataset),
            num_workers=self.hparams.workers_count, pin_memory=True)
        log.info("Test `Sampler` and `DataLoader` created.")
        return dataloader
