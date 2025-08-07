import os.path as osp
from typing import Any, Dict

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch.utils.data import Dataset

from ..utils import load_image


class MegaDepthDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        data_root: str,
        mode: str,
        image_size: int,
        image_factor: int,
        mask_factor: int,
        min_overlap_score: float = 0.0,
        depth: bool = True,
        fp16: bool = False,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.image_size = image_size
        self.mode = mode
        self.image_factor = image_factor
        self.mask_factor = mask_factor
        self.depth = depth
        self.fp16 = fp16

        self.scene_info = np.load(npz_path, allow_pickle=True)
        pair_infos = self.scene_info.pop("pair_infos")
        self.pair_infos = [
            info for info in pair_infos if info[1] > min_overlap_score
        ]
        self.depth_size = 2000

    def __len__(self) -> int:
        length = len(self.pair_infos)
        return length

    def _load_depth(self, path: str) -> ndarray:
        depth = np.array(h5py.File(path)["depth"])
        h, w = depth.shape

        pad_depth = np.zeros(
            (self.depth_size, self.depth_size), dtype=depth.dtype
        )
        pad_depth[:h, :w] = depth
        depth = pad_depth
        return depth

    def __getitem__(self, index: int) -> Dict[str, Any]:
        pair, overlap_score, central_matches = self.pair_infos[index]

        image0_name, image1_name = self.scene_info["image_paths"][pair]
        image0_path = osp.join(self.data_root, image0_name)
        image1_path = osp.join(self.data_root, image1_name)
        image0, mask0, scale0 = load_image(
            image0_path,
            mode=self.mode,
            size=self.image_size,
            factor=self.image_factor,
            pad_to_square=True,
        )
        image1, mask1, scale1 = load_image(
            image1_path,
            mode=self.mode,
            size=self.image_size,
            factor=self.image_factor,
            pad_to_square=True,
        )
        if self.mode == "gray":
            image0, image1 = image0[None] / 255.0, image1[None] / 255.0
        elif self.mode == "color":
            image0 = image0.transpose(2, 0, 1) / 255.0
            image1 = image1.transpose(2, 0, 1) / 255.0
        else:
            raise ValueError("")

        K0, K1 = self.scene_info["intrinsics"][pair].copy()
        T0, T1 = self.scene_info["poses"][pair]
        T0_to_1, T1_to_0 = T1 @ np.linalg.inv(T0), T0 @ np.linalg.inv(T1)

        data = {
            "root": self.data_root,
            "name0": image0_name,
            "name1": image1_name,
            "image0": image0,
            "image1": image1,
            "mask0": mask0,
            "mask1": mask1,
            "scale0": scale0,
            "scale1": scale1,
            "K0": K0,
            "K1": K1,
            "T0_to_1": T0_to_1,
            "T1_to_0": T1_to_0,
        }

        if self.depth:
            depth0_name, depth1_name = self.scene_info["depth_paths"][pair]
            depth0_path = osp.join(self.data_root, depth0_name)
            depth1_path = osp.join(self.data_root, depth1_name)
            data["depth0"] = self._load_depth(depth0_path)
            data["depth1"] = self._load_depth(depth1_path)

        for key, value in data.items():
            if isinstance(value, ndarray):
                if self.fp16 and key in [
                    "image0",
                    "image1",
                    "scale0",
                    "scale1",
                    "depth0",
                    "depth1",
                ]:
                    data[key] = torch.from_numpy(value).half()
                else:
                    data[key] = torch.from_numpy(value).float()

        mask = torch.stack([data["mask0"], data["mask1"]])
        data["mask0"], data["mask1"] = F.max_pool2d(
            mask, self.mask_factor
        ).bool()
        return data
