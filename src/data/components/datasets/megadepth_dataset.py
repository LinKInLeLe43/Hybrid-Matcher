from os import path
from typing import Any, Dict, List, Tuple

import cv2
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils import data


class MegaDepthDataset(data.Dataset):
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
        self.load_depth = load_depth

        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.pair_idxes = self.scene_info.pop("pair_infos")
        self.pair_idxes = [
            pair_info[0]
            for pair_info in self.pair_idxes
            if pair_info[1] > min_overlap_score
        ]
        self.depth_max_size = 2000

    def _read_image(
        self, path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape

        k = self.image_size / max(w, h)
        new_w, new_h = int(round(k * w)), int(round(k * h))
        new_w = int(new_w // self.image_factor * self.image_factor)
        new_h = int(new_h // self.image_factor * self.image_factor)
        image = cv2.resize(image, (new_w, new_h)) / 255.0
        color = cv2.resize(color, (new_w, new_h)) / 255.0
        color = (color - np.array([0.485, 0.456, 0.406])) / np.array(
            [0.229, 0.224, 0.225]
        )
        scale = np.array([w / new_w, h / new_h])

        length = max(new_w, new_h)
        padded_color = np.zeros((length, length, 3), dtype=image.dtype)
        padded_color[:new_h, :new_w] = color
        padded_image = np.zeros((length, length), dtype=image.dtype)
        padded_image[:new_h, :new_w] = image

        mask = np.zeros((length, length), dtype=np.bool_)
        mask[:new_h, :new_w] = True
        return padded_color, padded_image, mask, scale

    def _read_depth(self, path: str) -> np.ndarray:
        depth = np.array(h5py.File(path, "r")["depth"])
        h, w = depth.shape

        padded_depth = np.zeros(
            (self.depth_max_size, self.depth_max_size), dtype=depth.dtype
        )
        padded_depth[:h, :w] = depth
        return padded_depth

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idxes = self.pair_idxes[idx]

        image_name0, image_name1 = self.scene_info["image_paths"][idxes]
        image_path0 = path.join(self.data_root, image_name0)
        image_path1 = path.join(self.data_root, image_name1)
        color0, image0, mask0, scale0 = self._read_image(image_path0)
        color1, image1, mask1, scale1 = self._read_image(image_path1)
        color0, color1 = color0.transpose(2, 0, 1), color1.transpose(2, 0, 1)
        image0, image1 = image0[None], image1[None]

        K0, K1 = self.scene_info["intrinsics"][idxes].copy()

        T0, T1 = self.scene_info["poses"][idxes]
        T0_to_1, T1_to_0 = T1 @ np.linalg.inv(T0), T0 @ np.linalg.inv(T1)

        data = {
            "name0": image_name0,
            "name1": image_name1,
            "color0": color0,
            "color1": color1,
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

        if self.load_depth:
            depth_name0, depth_name1 = self.scene_info["depth_paths"][idxes]
            depth_path0 = path.join(self.data_root, depth_name0)
            depth_path1 = path.join(self.data_root, depth_name1)
            data["depth0"] = self._read_depth(depth_path0)
            data["depth1"] = self._read_depth(depth_path1)

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if self.fp16 and key in [
                    "color0",
                    "color1",
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

    def __len__(self) -> int:
        return len(self.pair_idxes)
