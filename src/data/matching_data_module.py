from os import path
from typing import Any, Callable, Dict, List

import joblib
import numpy as np
import pytorch_lightning as pl
from numpy import random
from rich import progress
from torch.utils import data

from src import utils
from src.data import utils as data_utils

log = utils.get_pylogger(__name__)


class MatchingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_configs: Dict[str, Any],
        train_batch_size_per_gpu: int,
        val_config: Dict[str, Any],
        test_config: Dict[str, Any],
        workers_count: int,
        pin_memory: bool = True,
        seed: int = 66,
        parallel: bool = False
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_datasets = None
        self.val_dataset = None
        self.test_dataset = None

    def _create_concat_dataset(
        self,
        dataset: Callable[[str], data.Dataset],
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
        with open(scene_list_path) as f:
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
            self.train_datasets = {}
            for name, config in self.hparams.train_configs.items():
                npz_paths = self._create_npz_paths(
                    config["npz_root"], config["list_path"], True)
                self.train_datasets[name] = self._create_concat_dataset(
                    config["dataset"], npz_paths)

            config = self.hparams.val_config
            npz_paths = self._create_npz_paths(
                config["npz_root"], config["list_path"], False)
            self.val_dataset = self._create_concat_dataset(
                config["dataset"], npz_paths)
            log.info("Train and validation `Dataset`s created.")

        if stage == "test":
            config = self.hparams.test_config
            npz_paths = self._create_npz_paths(
                config["npz_root"], config["list_path"], False)
            self.test_dataset = self._create_concat_dataset(
                config["dataset"], npz_paths)
            log.info("Test `Dataset` created.")

    def train_dataloader(self) -> data.DataLoader:
        dataloader = {}
        for name, config in self.hparams.train_configs.items():
            dataloader[name] = data.DataLoader(
                self.train_datasets[name],
                batch_size=self.hparams.train_batch_size_per_gpu,
                sampler=config["sampler"](self.train_datasets[name]),
                num_workers=self.hparams.workers_count,
                pin_memory=self.hparams.pin_memory)
        log.info("Train `Sampler` and `DataLoader` created. "
                 "(should not re-create between epochs)")
        return dataloader

    def val_dataloader(self) -> data.DataLoader:
        config = self.hparams.val_config
        dataloader = data.DataLoader(
            self.val_dataset,
            batch_size=1,
            sampler=config["sampler"](self.val_dataset),
            num_workers=self.hparams.workers_count,
            pin_memory=self.hparams.pin_memory)
        log.info("Validation `Sampler` and `DataLoader` created.")
        return dataloader

    def test_dataloader(self) -> data.DataLoader:
        config = self.hparams.test_config
        dataloader = data.DataLoader(
            self.test_dataset,
            batch_size=1,
            sampler=config["sampler"](self.test_dataset),
            num_workers=self.hparams.workers_count,
            pin_memory=True)
        log.info("Test `Sampler` and `DataLoader` created.")
        return dataloader
