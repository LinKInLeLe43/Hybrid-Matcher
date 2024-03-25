import functools
import math
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import distributed as dist
from torch import nn

from src.models import utils


def _flatten(
    outputs_per_rank: List[List[Dict[str, Any]]],
    keys: List[str]
) -> List[Any]:
    out = []
    for outputs in outputs_per_rank:
        for o in outputs:
            for k in keys:
                o = o[k]
            out.append(o)
    return out


class MatchingModule(pl.LightningModule):
    def __init__(
        self,
        net: nn.Module,
        loss: nn.Module,
        optimizer: functools.partial,
        scheduler: functools.partial,
        train_batch_size_per_gpu: int,
        canonical_batch_size: int,
        canonical_learning_rate: float,
        canonical_warmup_steps_count: int,
        warmup_ratio: float,
        end_point_thresholds: List[float],
        epipolar_thresholds: List[float],
        pose_thresholds: List[int],
        plot_for_train: bool = True,
        val_figures_count: int = 32,
        test_advanced_results: bool = True,
        dump_dir: Optional[str] = None
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["net", "loss"], logger=False)
        self.net = net
        self.loss = loss

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        result = self.net(batch)
        return result

    def _model_step(
        self,
        batch: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        use_flow = getattr(self.net.coarse_module, "use_flow", False)

        coarse_supervision = utils.create_coarse_supervision(
            batch, self.net.scales[0], use_flow=use_flow)

        result = self.net(batch, gt_idxes=coarse_supervision["gt_idxes"])

        fine_supervision = utils.create_fine_supervision(
            coarse_supervision.pop("point0_to_1"),
            coarse_supervision.pop("point1"), result["fine_idxes"],
            self.net.scales[1], self.net.window_size,
            scale1=batch.get("scale1"))

        loss = self.loss(
            result["coarse_confidences"], coarse_supervision["gt_mask"],
            result["fine_biases"], fine_supervision["gt_biases"],
            result["fine_stddevs"], flow0_to_1=result.get("flow0_to_1"),
            flow1_to_0=result.get("flow1_to_0"),
            gt_coor0_to_1=coarse_supervision.get("gt_coor0_to_1"),
            gt_coor1_to_0=coarse_supervision.get("gt_coor1_to_0"),
            mask0=batch.get("mask0"), mask1=batch.get("mask1"))
        return result, loss

    def on_train_start(self) -> None:
        scale = (self.trainer.world_size *
                 self.hparams.train_batch_size_per_gpu /
                 self.hparams.canonical_batch_size)
        self.hparams.optimizer.keywords["lr"] = (
            scale * self.hparams.canonical_learning_rate)
        self.hparams["warmup_steps_count"] =  math.floor(
            self.hparams.canonical_warmup_steps_count / scale)

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> Dict[str, Any]:
        result, loss = self._model_step(batch)

        loss_scalar = {"train_scalars/" + k: v
                       for k, v in loss["scalar"].items()}
        self.log_dict(loss_scalar)

        if self.net.coarse_matching.type == "optimal_transport":
            ot_bin_score = self.net.coarse_matching.ot_bin_score.detach().cpu()
            self.log("train_scalars/ot_bin_score", ot_bin_score)

        if (self.hparams.plot_for_train and self.trainer.global_rank == 0 and
            self.trainer._logger_connector.should_update_logs):
            error = utils.compute_error(batch, result)
            figures = utils.plot_evaluation_figures(
                batch, result, error, self.hparams.epipolar_thresholds[0])
            self.logger.experiment.add_figure(
                "train_figures", figures, global_step=self.global_step)
        return loss

    def training_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        avg_loss_on_epoch = torch.stack([o["loss"] for o in outputs]).mean()
        self.log("train_scalars/avg_loss_on_epoch", avg_loss_on_epoch)

    def on_validation_start(self) -> None:
        interval = (self.trainer.world_size * self.trainer.num_val_batches[0] //
                    self.hparams.val_figures_count)
        self.hparams.val_figures_interval = max(interval, 1)

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> Dict[str, Any]:
        result, loss = self._model_step(batch)
        error = utils.compute_error(batch, result)

        figures = []
        if batch_idx % self.hparams.val_figures_interval == 0:
            figures = utils.plot_evaluation_figures(
                batch, result, error, self.hparams.epipolar_thresholds[0])

        out = {"loss_scalar": loss["scalar"],
               "error": error,
               "figures": figures}
        return out

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        outputs_per_rank = [[] for _ in range(self.trainer.world_size)]
        dist.all_gather_object(outputs_per_rank, outputs)

        loss_scalar = {}
        for k in outputs[0]["loss_scalar"]:
            v = np.mean(_flatten(outputs_per_rank, ["loss_scalar", k]))
            loss_scalar["val_scalars/" + k] = v
        self.log_dict(loss_scalar)

        error = {}
        for k in outputs[0]["error"]:
            v = np.concatenate(_flatten(outputs_per_rank, ["error", k]))
            error[k] = v
        metric = utils.compute_metric(
            error, self.hparams.end_point_thresholds,
            self.hparams.epipolar_thresholds, self.hparams.pose_thresholds)
        for threshold, precision in zip(self.hparams.epipolar_thresholds,
                                        metric["epipolar_precisions"]):
            self.log(f"val_metrics/epipolar_precision@{threshold}", precision)
        for threshold, auc in zip(self.hparams.pose_thresholds,
                                  metric["pose_aucs"]):
            self.log(f"val_metrics/pose_auc@{threshold}", auc)

        if not self.trainer.sanity_checking:
            figures = np.concatenate(_flatten(outputs_per_rank, ["figures"]))
            for i, figure in enumerate(figures):
                self.logger.experiment.add_figure(
                    f"val_figures/pair-{i}", figure,
                    global_step=self.trainer.current_epoch)

    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> Dict[str, Any]:
        out = {}
        with self.trainer.profiler.profile("net"):
            result = self.net(batch)
        with self.trainer.profiler.profile("compute_error"):
            error = utils.compute_error(
                batch, result,
                advanced_results=self.hparams.test_advanced_results,
                coarse_scale=self.net.scales[0])
            out["error"] = error
        with self.trainer.profiler.profile("dump_results"):
            if self.hparams.dump_dir is not None:
                dumps = []
                for b in range(len(batch["image0"])):
                    item = {}
                    mask = result["b_idxes"] = b
                    for k in ("points0", "points1", "confidences"):
                        item[k] = result[k][mask].cpu().numpy()
                    for k in error.keys():
                        item[k] = error[k][b]
                    dumps.append(item)
                out["dumps"] = dumps
        return out

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        outputs_per_rank = [[] for _ in range(self.trainer.world_size)]
        dist.all_gather_object(outputs_per_rank, outputs)

        error = {}
        for k in outputs[0]["error"]:
            v = np.concatenate(_flatten(outputs_per_rank, ["error", k]))
            error[k] = v
        metric = utils.compute_metric(
            error, self.hparams.end_point_thresholds,
            self.hparams.epipolar_thresholds, self.hparams.pose_thresholds,
            advanced_results=self.hparams.test_advanced_results)
        if self.hparams.test_advanced_results:
            self.log("val_metrics/coarse_precision", metric["coarse_precision"])
            self.log(
                "val_metrics/inlier_coarse_precision",
                metric["inlier_coarse_precision"])
            for threshold, precision in zip(self.hparams.end_point_thresholds,
                                            metric["end_point_precisions"]):
                self.log(
                    f"val_metrics/end_point_precision@{threshold}", precision)
            for threshold, precision in zip(
                self.hparams.end_point_thresholds,
                metric["inlier_end_point_precisions"]):
                self.log(
                    f"val_metrics/inlier_end_point_precision@{threshold}",
                    precision)
        for threshold, precision in zip(self.hparams.epipolar_thresholds,
                                        metric["epipolar_precisions"]):
            self.log(f"val_metrics/epipolar_precision@{threshold}", precision)
        for threshold, auc in zip(self.hparams.pose_thresholds,
                                  metric["pose_aucs"]):
            self.log(f"val_metrics/pose_auc@{threshold}", auc)

        if self.hparams.dump_dir is not None:
            dump_dir = pathlib.Path(self.hparams.dump_dir)
            dump_dir.mkdir(parents=True, exist_ok=True)
            dumps = np.concatenate(_flatten(outputs_per_rank, ["dumps"]))
            np.save(dump_dir / "eval_result.npy", dumps)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(self.parameters())
        lr_scheduler = self.hparams.scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx=0,
        optimizer_closure=None,
        on_tpu=False,
        using_lbfgs=False,
    ) -> None:
        if self.trainer.global_step < self.hparams.warmup_steps_count:
            scale = self.trainer.global_step / self.hparams.warmup_steps_count
            learning_rate = (self.hparams.warmup_ratio *
                             self.hparams.optimizer.keywords["lr"])
            learning_rate += scale * (self.hparams.optimizer.keywords["lr"] -
                                      learning_rate)
            for pg in optimizer.param_groups:
                pg["lr"] = learning_rate
        optimizer.step(closure=optimizer_closure)
