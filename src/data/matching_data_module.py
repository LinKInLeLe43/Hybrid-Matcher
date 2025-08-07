import os.path as osp
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from numpy.random import RandomState
from pytorch_lightning import LightningDataModule
from rich.progress import Progress
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class MatchingDataModule(LightningDataModule):
    def __init__(
        self, test_config: Dict[str, Any], num_workers: int, seed: int = 66
    ) -> None:
        super().__init__()
        self.test_config = test_config
        self.num_workers = num_workers
        self.seed = seed

        self.test_dataset: Optional[Dataset] = None

    def _get_local_split(self, items: List[Any]) -> List[Any]:
        permuted_items = RandomState(self.seed).permutation(items)
        remainder = len(items) % self.trainer.world_size
        if remainder != 0:
            pad_size = self.trainer.world_size - remainder
            pad_items = RandomState(self.seed).choice(items, size=pad_size)
            permuted_items = np.concatenate([permuted_items, pad_items])

        num_per_rank = len(permuted_items) // self.trainer.world_size
        start = num_per_rank * self.trainer.global_rank
        end = start + num_per_rank
        local_items = list(permuted_items[start:end])
        return local_items

    def _get_npz_paths(
        self, scene_list_path: str, npz_root: str, split: bool = False
    ) -> List[str]:
        with open(scene_list_path) as f:
            names = f.read().splitlines()
        if split:
            names = self._get_local_split(names)
        n = len(names)
        log.info(f"{n} scene{'s' if n != 1 else ''} assigned per rank")

        npz_paths = []
        for name in names:
            if osp.splitext(name)[1] != ".npz":
                name = name + ".npz"
            npz_paths.append(osp.join(npz_root, name))
        return npz_paths

    def _make_concat_dataset(
        self, dataset_builder: Callable[[str], Dataset], npz_paths: List[str]
    ) -> ConcatDataset:
        with Progress(disable=self.trainer.global_rank != 0) as progress:
            npz_paths = progress.track(
                npz_paths, description="Loading scenes..."
            )
            datasets = [dataset_builder(path) for path in npz_paths]
        concat_dataset = ConcatDataset(datasets)
        return concat_dataset

    def setup(self, stage: str) -> None:
        if stage == "test":
            test_npz_paths = self._get_npz_paths(
                self.test_config["list_path"],
                self.test_config["npz_root"],
                split=False,
            )
            self.test_dataset = self._make_concat_dataset(
                self.test_config["dataset_builder"], test_npz_paths
            )

    def test_dataloader(self) -> DataLoader:
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=1,
            sampler=self.test_config["sampler_builder"](self.test_dataset),
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return dataloader
