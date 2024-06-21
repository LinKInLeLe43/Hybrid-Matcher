import functools
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import distributed as dist
from torch import nn

from src.models import utils


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
        canonical_warmup_step_count: int,
        warmup_ratio: float,
        end_point_thresholds: List[float],
        epipolar_thresholds: List[float],
        pose_thresholds: List[float],
        train_plot_enabled: bool = False,
        val_plot_count: int = 32,
        test_preparation_enabled: bool = False,
        test_enable_loransac: bool = False,
        advanced_metrics: bool = True,
        dump_dir: Optional[str] = None
    ) -> None:
        super().__init__()
        self.net = net
        self.loss = loss
        self.save_hyperparameters(ignore=["net", "loss"], logger=False)

        self.test_time_profiler = utils.InferenceProfiler()

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        result = self.net(batch)
        return result

    def on_train_start(self) -> None:
        scale = (self.trainer.world_size *
                 self.hparams.train_batch_size_per_gpu /
                 self.hparams.canonical_batch_size)
        self.hparams.optimizer.keywords["lr"] = (
            scale * self.hparams.canonical_learning_rate)
        self.hparams["warmup_step_count"] = (
            self.hparams.canonical_warmup_step_count / scale)

    def model_step(
        self,
        batch: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        supervision = {}
        if self.net.type == "one_stage":
            supervision.update(utils.create_coarse_supervision(
                batch, self.net.scales[0], return_coor=True,
                return_flow=self.net.use_flow))

            if hasattr(self.net, "extra_scale"):
                supervision[f"coarse_extra_gt_mask"] = (
                    utils.create_coarse_supervision(
                        batch, self.net.extra_scale)["coarse_gt_mask"])

            result = self.net(
                batch, gt_idxes=supervision["coarse_gt_idxes"])
            gt_biases = utils.compute_gt_biases(
                supervision.pop("points0_to_1"), supervision.pop("points1"),
                result["coarse_cls_idxes"], self.net.scales[1],
                self.net.reg_window_size)
            supervision["fine_gt_biases"] = gt_biases
        elif self.net.type == "two_stage":
            supervision.update(utils.create_coarse_supervision(
                batch, self.net.scales[0], return_flow=self.net.use_flow))

            if hasattr(self.net, "extra_scale"):
                offset = -(self.net.scales[0] / self.net.extra_scale / 2) + 0.5
                supervision[f"coarse_extra_gt_mask"] = (
                    utils.create_coarse_supervision(
                        batch, self.net.extra_scale,
                        offset=offset)["coarse_gt_mask"])

            result = self.net(
                batch, gt_idxes=supervision["coarse_gt_idxes"])
            supervision.update(utils.create_fine_supervision(
                batch, self.net.scales, result["coarse_cls_idxes"]))
            gt_biases = utils.compute_gt_biases(
                supervision.pop("points0_to_1"), supervision.pop("points1"),
                result["fine_cls_idxes"], self.net.scales[1],
                self.net.fine_reg_window_size)
            supervision["fine_gt_biases"] = gt_biases
        else:
            assert False

        extra_mask0 = extra_mask1 = None
        if hasattr(self.net, "extra_scale"):
            extra_mask0 = batch.get(f"mask0_{self.net.extra_scale}x")
            extra_mask1 = batch.get(f"mask1_{self.net.extra_scale}x")
        loss = self.loss(
            **result, **supervision,
            mask0=batch.get(f"mask0_{self.net.scales[0]}x"),
            mask1=batch.get(f"mask1_{self.net.scales[0]}x"),
            extra_mask0=extra_mask0, extra_mask1=extra_mask1)
        return result, loss

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> Dict[str, Any]:
        result, loss = self.model_step(batch)

        for k, v in loss.pop("scalar").items():
            self.log("train_scalar/" + k, v)
        if self.net.coarse_matching.type == "optimal_transport":
            ot_bin_score = self.net.coarse_matching.ot_bin_score.detach().cpu()
            self.log("train_scalar/ot_bin_score", ot_bin_score)

        if (self.hparams.train_plot_enabled and
            self.trainer.global_rank == 0 and
            self.trainer._logger_connector.should_update_logs):
            error = utils.compute_error(batch, result)
            figures = utils.plot_evaluation_figures(
                batch, result, error, self.hparams.epipolar_thresholds[0])
            self.logger.experiment.add_figure(
                "train_plot", figures, global_step=self.global_step)
        return loss

    def training_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        avg_loss_on_epoch = torch.stack([o["loss"] for o in outputs]).mean()
        self.log(
            "train_scalar/avg_loss_on_epoch", avg_loss_on_epoch, sync_dist=True)

    def on_validation_start(self) -> None:
        plot_count_per_rank = (self.hparams.val_plot_count //
                               self.trainer.world_size)
        intervals = [max(batch_count_per_rank // plot_count_per_rank, 1)
                     for batch_count_per_rank in self.trainer.num_val_batches]
        self.hparams["val_plot_intervals"] = intervals

    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> Dict[str, Any]:
        result, loss = self.model_step(batch)
        loss = loss.pop("scalar")
        error = utils.compute_error(
            batch, result, advanced=self.hparams.advanced_metrics,
            coarse_scale=self.net.scales[0])
        figures = []
        if batch_idx % self.hparams.val_plot_intervals[0] == 0:
            figures = utils.plot_evaluation_figures(
                batch, result, error, self.hparams.epipolar_thresholds[0])
        output = {"loss": loss, "error": error, "figures": figures}
        return output

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        outputs_by_ranks = [[] for _ in range(self.trainer.world_size)]
        dist.all_gather_object(outputs_by_ranks, outputs)
        gathered_output = _flatten(outputs_by_ranks)
        del outputs_by_ranks

        for k, v in gathered_output.pop("loss").items():
            self.log("val_scalar/" + k, torch.stack(v).mean())
        error = {k: np.concatenate(v)
                 for k, v in gathered_output.pop("error").items()}
        metric = utils.compute_metric(
            error, self.hparams.end_point_thresholds,
            self.hparams.epipolar_thresholds, self.hparams.pose_thresholds,
            advanced=self.hparams.advanced_metrics)
        for t, m in zip(self.hparams.epipolar_thresholds,
                        metric.pop("epipolar_precisions")):
            self.log(f"val_metric/epipolar_precision@{t}", m)
        for t, m in zip(self.hparams.pose_thresholds, metric.pop("pose_aucs")):
            self.log(f"val_metric/pose_auc@{t}", m)
        if self.hparams.advanced_metrics:
            self.log(
                "val_metric/coarse_precision", metric.pop("coarse_precision"))
            self.log(
                "val_metric/inlier_coarse_precision",
                metric.pop("inlier_coarse_precision"))
            self.log(
                "val_metric/true_coarse_count", metric.pop("true_coarse_count"))
            self.log(
                "val_metric/inlier_true_coarse_count",
                metric.pop("inlier_true_coarse_count"))
            self.log(
                "val_metric/coarse_3x3_precision",
                metric.pop("coarse_3x3_precision"))
            self.log(
                "val_metric/inlier_coarse_3x3_precision",
                metric.pop("inlier_coarse_3x3_precision"))
            for t, m0, m1 in zip(self.hparams.end_point_thresholds,
                                 metric.pop("end_point_precisions"),
                                 metric.pop("inlier_end_point_precisions")):
                self.log(f"val_metric/end_point_precision@{t}", m0)
                self.log(f"val_metric/inlier_end_point_precision@{t}", m1)

        if not self.trainer.sanity_checking:
            figures = np.concatenate(gathered_output.pop("figures"))
            for i, figure in enumerate(figures):
                self.logger.experiment.add_figure(
                    f"val_plot/pair-{i}", figure,
                    global_step=self.trainer.current_epoch)

    def on_test_start(self) -> None:
        for m in self.net.modules():
            if hasattr(m, "switch_to_deploy"):
                m.switch_to_deploy()

    def test_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0
    ) -> Dict[str, Any]:
        if self.hparams.test_preparation_enabled:
            for _ in range(50):
                result = self.net(batch)
            torch.cuda.synchronize()
            self.hparams.test_preparation_enabled = False

        with self.test_time_profiler.profile("net"):
            result = self.net(batch)
        with self.test_time_profiler.profile("error"):
            error = utils.compute_error(
                batch, result,
                enable_loransac=self.hparams.test_enable_loransac,
                advanced=self.hparams.advanced_metrics,
                coarse_scale=self.net.scales[0])

        dump = {}
        if self.hparams.dump_dir is not None:
            dump = error
            for k in ("points0", "points1", "confidences"):
                dump[k] = result[k].cpu().numpy()
        output = {"error": error, "dump": dump}
        return output

    def test_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        outputs_by_ranks = [[] for _ in range(self.trainer.world_size)]
        dist.all_gather_object(outputs_by_ranks, outputs)
        gathered_output = _flatten(outputs_by_ranks)
        del outputs_by_ranks

        error = {k: np.concatenate(v)
                 for k, v in gathered_output.pop("error").items()}
        metric = utils.compute_metric(
            error, self.hparams.end_point_thresholds,
            self.hparams.epipolar_thresholds, self.hparams.pose_thresholds,
            advanced=self.hparams.advanced_metrics)
        for t, m in zip(self.hparams.epipolar_thresholds,
                        metric.pop("epipolar_precisions")):
            self.log(f"test_metric/epipolar_precision@{t}", m)
        for t, m in zip(self.hparams.pose_thresholds, metric.pop("pose_aucs")):
            self.log(f"test_metric/pose_auc@{t}", m)
        if self.hparams.advanced_metrics:
            self.log(
                "test_metric/coarse_precision", metric.pop("coarse_precision"))
            self.log(
                "test_metric/inlier_coarse_precision",
                metric.pop("inlier_coarse_precision"))
            self.log(
                "test_metric/true_coarse_count",
                metric.pop("true_coarse_count"))
            self.log(
                "test_metric/inlier_true_coarse_count",
                metric.pop("inlier_true_coarse_count"))
            self.log(
                "test_metric/coarse_3x3_precision",
                metric.pop("coarse_3x3_precision"))
            self.log(
                "test_metric/inlier_coarse_3x3_precision",
                metric.pop("inlier_coarse_3x3_precision"))
            for t, m0, m1 in zip(self.hparams.end_point_thresholds,
                                 metric.pop("end_point_precisions"),
                                 metric.pop("inlier_end_point_precisions")):
                self.log(f"test_metric/end_point_precision@{t}", m0)
                self.log(f"test_metric/inlier_end_point_precision@{t}", m1)

        if self.hparams.dump_dir is not None:
            pathlib.Path(
                self.hparams.dump_dir).mkdir(parents=True, exist_ok=True)
            np.save(
                self.hparams.dump_dir + "/test_result", gathered_output["dump"])

        if self.trainer.global_rank == 0:
            print(self.test_time_profiler.summary())

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(self.parameters())
        scheduler = self.hparams.scheduler(optimizer)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler}}

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx=0,
        optimizer_closure=None,
        on_tpu=False,
        using_lbfgs=False
    ) -> None:
        if self.trainer.global_step <= self.hparams.warmup_step_count:
            scale = (self.hparams.warmup_ratio +
                     self.trainer.global_step / self.hparams.warmup_step_count *
                     (1 - self.hparams.warmup_ratio))
            lr = scale * self.hparams.optimizer.keywords["lr"]
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        optimizer.step(closure=optimizer_closure)
