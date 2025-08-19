import os.path as osp
from typing import Any, Dict, List, Tuple

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch.utils.data import Dataset


class MegaDepthDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        data_root: str,
        image_size: int,
        image_factor: int,
        mask_factors: List[int],
        fp16: bool = False,
        load_depth: bool = True,
        min_overlap_score: float = 0.0,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.image_size = image_size
        self.image_factor = image_factor
        self.mask_factors = mask_factors
        self.fp16 = fp16
        self.depth = load_depth

        self.scene_info = np.load(npz_path, allow_pickle=True)
        pair_infos = self.scene_info.pop("pair_infos")
        self.pair_infos = [
            info for info in pair_infos if info[1] > min_overlap_score
        ]
        self.depth_size = 2000

    def __len__(self) -> int:
        length = len(self.pair_infos)
        return length

    def _read_image(self, path: str) -> Tuple[ndarray, ndarray, ndarray]:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape

        k = self.image_size / max(w, h)
        new_w, new_h = int(round(k * w)), int(round(k * h))
        new_w = int(new_w // self.image_factor * self.image_factor)
        new_h = int(new_h // self.image_factor * self.image_factor)
        image = cv2.resize(image, (new_w, new_h))
        scale = np.array([w / new_w, h / new_h])

        length = max(new_w, new_h)
        padded_image = np.zeros((length, length), dtype=image.dtype)
        padded_image[:new_h, :new_w] = image
        padded_image = padded_image / 255

        mask = np.zeros((length, length), dtype=np.bool_)
        mask[:new_h, :new_w] = True
        return padded_image, mask, scale

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
        image0, mask0, scale0 = self._read_image(image0_path)
        image1, mask1, scale1 = self._read_image(image1_path)
        image0, image1 = image0[None], image1[None]

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

        mask = torch.stack([data.pop("mask0"), data.pop("mask1")])
        for factor in self.mask_factors:
            data[f"mask0_{factor}x"], data[f"mask1_{factor}x"] = F.max_pool2d(
                mask, factor, stride=factor
            ).bool()
        return data
