from os import path
from typing import Any, Dict

import cv2
import numpy as np
from numpy import linalg
import torch
from torch.utils import data


class ScanNetDataset(data.Dataset):
    def __init__(
        self,
        npz_path: str,
        data_root: str,
        intrinsic_path: str,
        load_depth: bool = True,
        min_overlap_score: float = 0.0
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.intrinsics = dict(np.load(intrinsic_path))
        self.load_depth = load_depth

        with np.load(npz_path) as data:
            self.names = data["name"]
            scores = data.get("score")
            if scores is not None:
                mask = scores > min_overlap_score
                self.names = self.names[mask]

    def _read_image(self, path: str) -> np.ndarray:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (640, 480))
        image = image / 255
        return image

    def _read_depth(self, path: str) -> np.ndarray:
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        depth = depth / 1000
        return depth

    def _read_pose(self, path: str) -> np.ndarray:
        pose_camera_to_world = np.loadtxt(path, delimiter=" ")
        pose_world_to_camera = linalg.inv(pose_camera_to_world)
        return pose_world_to_camera

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scene_name, scene_subname, stem0_name, stem1_name = self.names[idx]
        scene_name = f"scene{scene_name:04d}_{scene_subname:02d}"

        image_name0 = path.join(scene_name, "color", f"{stem0_name}.jpg")
        image_name1 = path.join(scene_name, "color", f"{stem1_name}.jpg")
        image_path0 = path.join(self.data_root, image_name0)
        image_path1 = path.join(self.data_root, image_name1)
        image0 = self._read_image(image_path0)[None]
        image1 = self._read_image(image_path1)[None]

        K0 = K1 = self.intrinsics[scene_name].copy().reshape(3, 3)

        pose0_path = path.join(
            self.data_root, scene_name, "pose", f"{stem0_name}.txt")
        pose1_path = path.join(
            self.data_root, scene_name, "pose", f"{stem1_name}.txt")
        T0, T1 = self._read_pose(pose0_path), self._read_pose(pose1_path)
        T0_to_1, T1_to_0 = T1 @ np.linalg.inv(T0), T0 @ np.linalg.inv(T1)

        data = {"name0": image_name0,
                "name1": image_name1,
                "image0": image0,
                "image1": image1,
                "K0": K0,
                "K1": K1,
                "T0_to_1": T0_to_1,
                "T1_to_0": T1_to_0}

        if self.load_depth:
            depth_path0 = path.join(
                self.data_root, scene_name, "depth", f"{stem0_name}.png")
            depth_path1 = path.join(
                self.data_root, scene_name, "depth", f"{stem1_name}.png")
            data["depth0"] = self._read_depth(depth_path0)
            data["depth1"] = self._read_depth(depth_path1)

        for k, v in data.items():
            if isinstance(v, np.ndarray):
                data[k] = torch.from_numpy(v).float()
        return data

    def __len__(self) -> int:
        return len(self.names)
