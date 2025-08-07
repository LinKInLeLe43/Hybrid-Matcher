from typing import Any, Dict, List, Optional

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.distributed import all_gather_object
from torch.nn import Module

from src.models.utils import compute_error, compute_metric


def _flatten(outputs_by_ranks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    def _append(in_dict: Dict[str, Any], out_dict: Dict[str, Any]) -> None:
        for k, v in in_dict.items():
            if isinstance(v, dict):
                if k not in out_dict:
                    out_dict[k] = {}
                _append(v, out_dict[k])
            else:
                if k not in out_dict:
                    out_dict[k] = []
                out_dict[k].append(v)

    gathered_output = {}
    for outputs in outputs_by_ranks:
        for o in outputs:
            _append(o, gathered_output)
    return gathered_output


class MatchingModule(LightningModule):
    def __init__(
        self,
        net: Module,
        metric_thresholds: Dict[str, Any],
        rel_pose_ransac_config: Dict[str, Any],
        test_task: Optional[str] = None,
        test_fp16_precision: bool = False,
    ) -> None:
        super().__init__()
        assert test_task in [None, "accuracy", "efficiency"]
        self.net = net
        self.save_hyperparameters(ignore=["net"], logger=False)

        # For efficiency
        if test_task == "efficiency":
            self.warmup = False
            self.total_ms = 0.0
            self.num_pairs = 0
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        results = self.net(batch)
        return results

    def on_test_start(self) -> None:
        if (
            self.hparams.test_task == "efficiency"
            and self.trainer.world_size != 1
        ):
            raise ValueError("")

        for m in self.net.modules():
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()

        if self.hparams.test_fp16_precision:
            self.net = self.net.half()

    def test_accuracy(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = self.net(batch)
        error = compute_error(
            batch,
            out,
            self.hparams.rel_pose_ransac_config,
            advanced=self.hparams.advanced_metrics,
            coarse_scale=self.net.scales[0],
        )
        results = {"error": error}
        return results

    def test_efficiency(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if not self.warmup:
            for _ in range(100):
                self.net(batch)
            self.warmup = True
            torch.cuda.synchronize()

        self.start_event.record()
        self.net(batch)
        self.end_event.record()
        torch.cuda.synchronize()
        self.total_ms += self.start_event.elapsed_time(self.end_event)
        self.num_pairs += 1
        return {}

    def test_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, Any]:
        assert self.hparams.test_task in ["accuracy", "efficiency"]
        if self.hparams.test_task == "accuracy":
            results = self.test_accuracy(batch)
        elif self.hparams.test_task == "efficiency":
            results = self.test_efficiency(batch)
        else:
            raise AssertionError()
        return results

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        if self.hparams.test_task == "accuracy":
            outputs_by_ranks = [[] for _ in range(self.trainer.world_size)]
            all_gather_object(outputs_by_ranks, outputs)
            gathered_output = _flatten(outputs_by_ranks)
            del outputs_by_ranks

            error = {
                k: np.concatenate(v)
                for k, v in gathered_output.pop("error").items()
            }
            metric = compute_metric(
                error,
                self.hparams.metric_thresholds,
                advanced=self.hparams.advanced_metrics,
            )
            for key, value in metric.items():
                self.log("test_metric/" + key, value)
        elif self.hparams.test_task == "efficiency":
            self.log("test_metric/runtime@ms", self.total_ms / self.num_pairs)
        else:
            raise AssertionError()
