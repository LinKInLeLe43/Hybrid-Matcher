import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import torch
from numpy import ndarray
from torch.utils.data import Dataset


class ScanNetDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        data_root: str,
        intrinsic_path: str,
        load_depth: bool = True,
        fp16: bool = False,
        min_overlap_score: float = 0.0,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.fp16 = fp16
        self.depth = load_depth

        with np.load(npz_path) as info:
            self.pair_names = info["name"]
            if "score" in info:
                mask = info["score"] > min_overlap_score
                self.pair_names = self.pair_names[mask]
        self.intrinsics = dict(np.load(intrinsic_path))

    def __len__(self) -> int:
        length = len(self.pair_names)
        return length

    def _read_image(self, path: str) -> ndarray:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (640, 480))
        image = image / 255
        return image

    def _load_depth(self, path: str) -> ndarray:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        depth = depth / 1000.0
        return depth

    def _load_pose(self, path: str) -> ndarray:
        pose_camera_to_world = np.loadtxt(path)
        pose_world_to_camera = np.linalg.inv(pose_camera_to_world)
        return pose_world_to_camera

    def __getitem__(self, index: int) -> Dict[str, Any]:
        scene, subscene, stem0, stem1 = self.pair_names[index]
        scene = f"scene{scene:04d}_{subscene:02d}"

        image0_name = osp.join(scene, "color", f"{stem0}.jpg")
        image1_name = osp.join(scene, "color", f"{stem1}.jpg")
        image0_path = osp.join(self.data_root, image0_name)
        image1_path = osp.join(self.data_root, image1_name)
        image0 = self._read_image(image0_path)[None]
        image1 = self._read_image(image1_path)[None]

        K0 = K1 = self.intrinsics[scene].copy().reshape(3, 3)
        pose0_path = osp.join(self.data_root, scene, "pose", f"{stem0}.txt")
        pose1_path = osp.join(self.data_root, scene, "pose", f"{stem1}.txt")
        T0, T1 = self._load_pose(pose0_path), self._load_pose(pose1_path)
        T0_to_1, T1_to_0 = T1 @ np.linalg.inv(T0), T0 @ np.linalg.inv(T1)

        data = {
            "root": self.data_root,
            "name0": image0_name,
            "name1": image1_name,
            "image0": image0,
            "image1": image1,
            "K0": K0,
            "K1": K1,
            "T0_to_1": T0_to_1,
            "T1_to_0": T1_to_0,
        }

        if self.depth:
            depth0_path = osp.join(
                self.data_root, scene, "depth", f"{stem0}.png"
            )
            depth1_path = osp.join(
                self.data_root, scene, "depth", f"{stem1}.png"
            )
            data["depth0"] = self._load_depth(depth0_path)
            data["depth1"] = self._load_depth(depth1_path)

        for key, value in data.items():
            if isinstance(value, ndarray):
                if self.fp16 and key in [
                    "image0",
                    "image1",
                    "depth0",
                    "depth1",
                ]:
                    data[key] = torch.from_numpy(value).half()
                else:
                    data[key] = torch.from_numpy(value).float()
        return data
