from os.path import join
import glob
from typing import Any, Dict, List, Tuple

import cv2
import h5py
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils import data


class ETH3DDataset(data.Dataset):
    def __init__(
        self,
        npz_path: str,
        data_root: str,
        image_size: int,
        image_factor: int,
        mask_factors: List[int],
        fp16: bool = False,
        load_depth: bool = True,
        min_overlap_score: float = 0.0
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.image_size = image_size
        self.image_factor = image_factor
        self.mask_factors = mask_factors
        self.fp16 = fp16

        paths = glob.glob(join(self.data_root, '*.txt'))
        lines = []
        for path in paths:
            with open(path, 'r') as file:
                scene_id = path.rpartition('/')[-1].rpartition('.')[0].split('-')[0]
                line = file.readline().strip().split()
                lines.append([scene_id] + line)
        self.pairs = sorted(lines)

    def _read_image(
        self,
        path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def _read_depth(self, path: str) -> np.ndarray:
        depth = np.array(h5py.File(path, "r")["depth"])
        h, w = depth.shape

        padded_depth = np.zeros(
            (self.depth_max_size, self.depth_max_size), dtype=depth.dtype)
        padded_depth[:h, :w] = depth
        return padded_depth

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair = self.pairs[idx]
        scene_id = pair[0]

        image_name0 = pair[1].rpartition('.')[0]
        image_name1 = pair[2].rpartition('.')[0]

        image_path0 = join(self.data_root, '{}-{}.png'.format(scene_id, image_name0))
        image_path1 = join(self.data_root, '{}-{}.png'.format(scene_id, image_name1))
        image0, mask0, scale0 = self._read_image(image_path0)
        image1, mask1, scale1 = self._read_image(image_path1)
        image0, image1 = image0[None], image1[None]

        K0 = torch.tensor(list(map(float, pair[5:14])), dtype=torch.float).reshape(3, 3)
        K1 = torch.tensor(list(map(float, pair[14:23])), dtype=torch.float).reshape(3, 3)

        T0_to_1 = torch.tensor(list(map(float, pair[23:])), dtype=torch.float).reshape(4, 4)
        T1_to_0 = T0_to_1.inverse()

        data = {"name0": image_name0,
                "name1": image_name1,
                "image0": image0,
                "image1": image1,
                "mask0": mask0,
                "mask1": mask1,
                "scale0": scale0,
                "scale1": scale1,
                "K0": K0,
                "K1": K1,
                "T0_to_1": T0_to_1,
                "T1_to_0": T1_to_0}

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
        return len(self.pairs)
