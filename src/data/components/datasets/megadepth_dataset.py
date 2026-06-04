import random
from os import path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils import data

from .sample_homo import sample_homography_sap
from kornia.geometry import homography_warp, normalize_homography, normal_transform_pixel


class MegaDepthDataset(data.Dataset):
    def __init__(
        self,
        npz_path: str,
        data_root: str,
        image_size: int,
        image_factor: int,
        mask_factors: List[int],
        modality_list: Optional[List[str]] = None,
        fp16: bool = False,
        load_depth: bool = True,
        homo: bool = True,
        min_overlap_score: float = 0.0,
        seed: int = 66,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.image_size = image_size
        self.image_factor = image_factor
        self.mask_factors = mask_factors
        self.modality_list = modality_list or ["visible"]
        self.fp16 = fp16
        self.load_depth = load_depth
        self.homo = homo
        self.seed = seed

        self.scene_info = np.load(npz_path, allow_pickle=True)
        self.scene_info = dict(self.scene_info)
        self.pair_idxes = self.scene_info.pop("pair_infos")
        self.pair_idxes = [pair_info[0] for pair_info in self.pair_idxes
                           if pair_info[1] > min_overlap_score]
        self.depth_max_size = 2000
        self.modality_to_root = {
            "visible": self.data_root,
            "infrared": "data/megadepth/train/infrared/",
            "depth": "data/megadepth/train/depth/",
            "normal": "data/megadepth/train/normal/",
            "event": "data/megadepth/train/event/",
            "sketch": "data/megadepth/train/sketch/",
            "paint": "data/megadepth/train/paint/",
        }

    def _read_image(
        self,
        path: str,
        use_homo: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape

        k = self.image_size / max(w, h)
        new_w, new_h = int(round(k * w)), int(round(k * h))
        new_w = int(new_w // self.image_factor * self.image_factor)
        new_h = int(new_h // self.image_factor * self.image_factor)

        if self.homo and use_homo:
            homo_sampled = sample_homography_sap(h, w) # 3*3
            homo_sampled_normed = normalize_homography(
                torch.from_numpy(homo_sampled[None]).to(torch.float32),
                (h, w),
                (h, w),
            )

            image = torch.from_numpy(image).float()[None, None] / 255
            homo_warpped_image = homography_warp(
                image, # 1 * C * H * W
                torch.linalg.inv(homo_sampled_normed),
                (h, w),
            )
            image = (homo_warpped_image[0].permute(1,2,0).numpy() * 255).astype(np.uint8)
        else:
            homo_sampled = torch.eye(3,3)

        image = cv2.resize(image, (new_w, new_h))
        scale = np.array([w / new_w, h / new_h])

        length = max(new_w, new_h)
        padded_image = np.zeros((length, length), dtype=image.dtype)
        padded_image[:new_h, :new_w] = image
        padded_image = padded_image / 255

        mask = np.zeros((length, length), dtype=np.bool_)
        mask[:new_h, :new_w] = True
        return padded_image, mask, scale, homo_sampled

    def _read_depth(self, path: str) -> np.ndarray:
        depth = np.array(h5py.File(path, "r")["depth"])
        h, w = depth.shape

        padded_depth = np.zeros(
            (self.depth_max_size, self.depth_max_size), dtype=depth.dtype)
        padded_depth[:h, :w] = depth
        return padded_depth

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idxes = self.pair_idxes[idx]

        image_name0, image_name1 = self.scene_info["image_paths"][idxes]
        if len(self.modality_list) > 1:
            modality = random.choice(self.modality_list)
        else:
            modality = self.modality_list[0]
        modality_swap = random.choice([True, False])
        if modality_swap:
            root0, root1 = self.modality_to_root[modality], self.data_root
        else:
            root0, root1 = self.data_root, self.modality_to_root[modality]
        image_path0 = path.join(root0, image_name0)
        image_path1 = path.join(root1, image_name1)
        if modality == "event" or modality == "sketch" or modality == "paint":
            if modality_swap:
                image_path0 = path.splitext(image_path0)[0] + ".png"
            else:
                image_path1 = path.splitext(image_path1)[0] + ".png"
        homo_swap = random.choice([True, False])
        if homo_swap:
            use_homo0, use_homo1 = True, False
        else:
            use_homo0, use_homo1 = False, True
        image0, mask0, scale0, H0 = self._read_image(image_path0, use_homo=use_homo0)
        image1, mask1, scale1, H1 = self._read_image(image_path1, use_homo=use_homo1)
        image0, image1 = image0[None], image1[None]

        K0, K1 = self.scene_info["intrinsics"][idxes].copy()

        T0, T1 = self.scene_info["poses"][idxes]
        T0_to_1, T1_to_0 = T1 @ np.linalg.inv(T0), T0 @ np.linalg.inv(T1)

        data = {"name0": image_name0,
                "name1": image_name1,
                "image0": image0,
                "image1": image1,
                "mask0": mask0,
                "mask1": mask1,
                "scale0": scale0,
                "scale1": scale1,
                "H0": H0,
                "H1": H1,
                "K0": K0,
                "K1": K1,
                "T0_to_1": T0_to_1,
                "T1_to_0": T1_to_0}

        if self.load_depth:
            depth_name0, depth_name1 = self.scene_info["depth_paths"][idxes]
            depth_path0 = path.join(self.data_root, depth_name0)
            depth_path1 = path.join(self.data_root, depth_name1)
            data["depth0"] = self._read_depth(depth_path0)
            data["depth1"] = self._read_depth(depth_path1)

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if self.fp16 and key in ["image0", "image1", "scale0", "scale1", "depth0", "depth1"]:
                    data[key] = torch.from_numpy(value).half()
                else:
                    data[key] = torch.from_numpy(value).float()

        mask = torch.stack([data.pop("mask0"), data.pop("mask1")])
        for factor in self.mask_factors:
            data[f"mask0_{factor}x"], data[f"mask1_{factor}x"] = F.max_pool2d(
                mask, factor, stride=factor).bool()
        return data

    def __len__(self) -> int:
        return len(self.pair_idxes)
