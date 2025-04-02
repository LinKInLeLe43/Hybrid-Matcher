import functools
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch import distributed as dist
from torch import nn

from src.models import utils
from src.models.components.nets.new_matcher.homo.utils.dense_match import DenseMatch

import cv2
from matplotlib import figure
from matplotlib import lines
from matplotlib import pyplot as plt
import numpy as np
def plot_matching_figure(
    image0: np.ndarray,
    image1: np.ndarray,
    matching_points0: np.ndarray,
    matching_points1: np.ndarray,
    colors: np.ndarray,
    key_points0: Optional[np.ndarray] = None,
    key_points1: Optional[np.ndarray] = None,
    dpi: int = 75,
    text: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Optional[figure.Figure]:
    if not len(matching_points0) == len(matching_points1) == len(colors):
        raise ValueError("")
    if (key_points0 is None) == (key_points1 is not None):
        raise ValueError("")

    figure, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(image0, cmap="gray")
    axes[1].imshow(image1, cmap="gray")
    for i in range(2):
        axes[i].get_xaxis().set_ticks([])
        axes[i].get_yaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    if key_points0 is not None:
        axes[0].scatter(key_points0[:, 0], key_points0[:, 1], s=2, c="w")
        axes[1].scatter(key_points1[:, 0], key_points1[:, 1], s=2, c="w")

    n = len(matching_points0)
    if n != 0:
        figure.canvas.draw()
        axes[0].scatter(
            matching_points0[:, 0], matching_points0[:, 1], s=2, c=colors)
        axes[1].scatter(
            matching_points1[:, 0], matching_points1[:, 1], s=2, c=colors)

        inv_figure_trans = figure.transFigure.inverted()
        figure_points0 = axes[0].transData.transform(matching_points0)
        figure_points1 = axes[1].transData.transform(matching_points1)
        figure_points0 = inv_figure_trans.transform(figure_points0)
        figure_points1 = inv_figure_trans.transform(figure_points1)
        for i in range(n):
            x_coors = [figure_points0[i, 0], figure_points1[i, 0]]
            y_coors = [figure_points0[i, 1], figure_points1[i, 1]]
            # line = lines.Line2D(
            #     x_coors, y_coors, lw=1, c=colors[i],
            #     transform=figure.transFigure)
            # figure.lines.append(line)

    if text is not None:
        color = "k" if image0[:100, :200].mean() > 180 else "w"
        text = "\n".join(text)
        figure.text(
            0.01, 0.99, text, size=15, c=color, va="top", ha="left",
            transform=figure.axes[0].transAxes)

    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    else:
        return figure


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
        pose_ransac_count: int,
        pose_thresholds: List[float],
        train_plot_enabled: bool = False,
        val_plot_count: int = 32,
        test_fp16_precision: bool = False,
        test_preparation_enabled: bool = False,
        test_enable_loransac: bool = False,
        advanced_metrics: bool = True,
        dump_dir: Optional[str] = None
    ) -> None:
        super().__init__()
        self.net = net
        self.loss = loss
        self.save_hyperparameters(ignore=["net", "loss"], logger=False)

        self.dense_matcher = DenseMatch()
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
        batch: Dict[str, Any],
        loss_type: str = "full"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        s0, (s1, s2) = self.net.extra_scale, self.net.scales
        if self.net.type == "one_stage":
            supervision = utils.create_coarse_supervision(
                batch, s1, extra_scale=s0, return_coor=True)
            coarse_gt_points1 = supervision.pop("gt_points1")
            result = self.net(
                batch, gt_idxes=supervision["coarse_gt_idxes"],
                extra_gt_idxes=supervision.get("extra_coarse_gt_idxes"))
            supervision.update(utils.create_fine_supervision(
                batch, (s1, 1), result["coarse_cls_idxes"],
                offset=self.net.fine_cls_matching.cls_offset, return_coor=True))
            supervision["fine_gt_biases"] = utils.compute_reg_gt_biases(
                supervision.pop("gt_points0_to_1"), supervision.pop("gt_points1"),
                result["fine_cls_idxes"], s2, self.net.reg_w)
            supervision.update(utils.compute_dense_gt_biases(
                batch, result, self.dense_matcher, coarse_gt_points1,
                result["coarse_cls_idxes"], s2, self.net.reg_w))
        elif self.net.type == "two_stage":
            supervision = utils.create_coarse_supervision(
                batch, s1, extra_scale=s0)
            result = self.net(
                batch, gt_idxes=supervision["coarse_gt_idxes"],
                extra_gt_idxes=supervision.get("extra_coarse_gt_idxes"))
            supervision.update(utils.create_fine_supervision(
                batch, (s1, s2), result["coarse_cls_idxes"],
                offset=self.net.fine_cls_matching.cls_offset, return_coor=True))
            supervision["fine_gt_biases"] = utils.compute_reg_gt_biases(
                supervision.pop("points0_to_1"), supervision.pop("points1"),
                result["fine_cls_idxes"], s2, self.net.reg_w)
        else:
            assert False
        loss = self.loss(
            **result, **supervision, mask0=batch.get(f"mask0_{s1}x"),
            mask1=batch.get(f"mask1_{s1}x"),
            extra_mask0=batch.get(f"mask0_{s0}x"),
            extra_mask1=batch.get(f"mask1_{s0}x"),
            loss_type=loss_type)
        return result, loss

    def model_step_by_or(
        self,
        batch: Dict[str, Any],
        loss_type: str = "full"
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        s0, (s1, s2) = self.net.extra_scale, self.net.scales
        if self.net.type == "one_stage":
            # supervision = utils.create_coarse_supervision(
            #     batch, s1, extra_scale=s0, return_coor=True)
            # coarse_gt_points1 = supervision.pop("gt_points1")
            # result = self.net(
            #     batch, gt_idxes=supervision["coarse_gt_idxes"],
            #     extra_gt_idxes=supervision.get("extra_coarse_gt_idxes"))
            supervision = {}
            result = self.net(batch)

            # gt_b_idxes, gt_i_idxes, gt_j_idxes = supervision["coarse_gt_idxes"]
            # gt_points0 = supervision["gt_points0"][gt_b_idxes, gt_i_idxes]
            # gt_points1 = supervision["gt_points0_to_1"][gt_b_idxes, gt_i_idxes]
            _b_idxes = result["all_idxes"][0]
            _points0 = result["all_points0"]
            _points1 = result["all_points1"]
            or_mask = torch.zeros_like(_b_idxes, dtype=torch.bool)
            for b in range(len(batch["image0"])):
                b_mask = _b_idxes == b
                try:
                    _, _or_mask = cv2.findFundamentalMat(_points0[b_mask].cpu().numpy(), _points1[b_mask].cpu().numpy(), method=cv2.USAC_MAGSAC, ransacReprojThreshold=0.5, maxIters=10000, confidence=0.999999)
                    or_mask[b_mask] = torch.from_numpy(_or_mask.ravel() == 1).to(_points0.device)
                except:
                    or_mask[b_mask] = torch.zeros(len(_points0[b_mask]), dtype=torch.bool, device=_points0.device)
            # or_mask = or_mask[len(gt_b_idxes):]

            # neg_b_idxes, neg_i_idxes, neg_j_idxes = result["all_idxes"][0][~or_mask], result["all_idxes"][1][~or_mask], result["all_idxes"][2][~or_mask]
            # result["coarse_cls_heatmap"][neg_b_idxes, neg_i_idxes, neg_j_idxes] = 1 - result["coarse_cls_heatmap"][neg_b_idxes, neg_i_idxes, neg_j_idxes]
            # supervision["coarse_gt_mask"][neg_b_idxes, neg_i_idxes, neg_j_idxes] = True
            pos_b_idxes, pos_i_idxes, pos_j_idxes = result["all_idxes"][0][or_mask], result["all_idxes"][1][or_mask], result["all_idxes"][2][or_mask]
            supervision["coarse_gt_mask"] = torch.zeros_like(result["coarse_cls_heatmap"], dtype=torch.bool)
            supervision["coarse_gt_mask"][pos_b_idxes, pos_i_idxes, pos_j_idxes] = True

            # extra_neg_mask = torch.zeros_like(supervision["coarse_gt_mask"], dtype=torch.bool)
            # extra_neg_mask[neg_b_idxes, neg_i_idxes, neg_j_idxes] = True
            # _, _, h0, w0 = batch["image0"].shape
            # _, _, h1, w1 = batch["image1"].shape
            # stride = 2
            # fh0, fw0, fh1, fw1 = map(lambda x: x // 16, (h0, w0, h1, w1))
            # extra_neg_mask = extra_neg_mask.reshape(
            #     -1, fh0, stride, fw0, stride, fh1, stride, fw1, stride)
            # extra_neg_mask = extra_neg_mask.sum(dim=(2, 4, 6, 8)).bool()
            # extra_neg_mask = extra_neg_mask.reshape(-1, fh0 * fw0, fh1 * fw1)
            # neg_b_idxes, neg_i_idxes, neg_j_idxes = extra_neg_mask.nonzero(as_tuple=True)
            # result["extra_coarse_cls_heatmap"][neg_b_idxes, neg_i_idxes, neg_j_idxes] = 1 - result["extra_coarse_cls_heatmap"][neg_b_idxes, neg_i_idxes, neg_j_idxes]
            # supervision["extra_coarse_gt_mask"][neg_b_idxes, neg_i_idxes, neg_j_idxes] = True
            extra_pos_mask = torch.zeros_like(supervision["coarse_gt_mask"], dtype=torch.bool)
            extra_pos_mask[pos_b_idxes, pos_i_idxes, pos_j_idxes] = True
            _, _, h0, w0 = batch["image0"].shape
            _, _, h1, w1 = batch["image1"].shape
            stride = 2
            fh0, fw0, fh1, fw1 = map(lambda x: x // 16, (h0, w0, h1, w1))
            extra_pos_mask = extra_pos_mask.reshape(
                -1, fh0, stride, fw0, stride, fh1, stride, fw1, stride)
            extra_pos_mask = extra_pos_mask.sum(dim=(2, 4, 6, 8)).bool()
            extra_pos_mask = extra_pos_mask.reshape(-1, fh0 * fw0, fh1 * fw1)
            pos_b_idxes, pos_i_idxes, pos_j_idxes = extra_pos_mask.nonzero(as_tuple=True)
            supervision["extra_coarse_gt_mask"] = torch.zeros_like(result["extra_coarse_cls_heatmap"], dtype=torch.bool)
            supervision["extra_coarse_gt_mask"][pos_b_idxes, pos_i_idxes, pos_j_idxes] = True

            # supervision.update(utils.create_fine_supervision(
            #     batch, (s1, 1), result["coarse_cls_idxes"],
            #     offset=self.net.fine_cls_matching.cls_offset, return_coor=True))
            # supervision["fine_gt_biases"] = utils.compute_reg_gt_biases(
            #     supervision.pop("gt_points0_to_1"), supervision.pop("gt_points1"),
            #     result["fine_cls_idxes"], s2, self.net.reg_w)
            # supervision.update(utils.compute_dense_gt_biases(
            #     batch, result, self.dense_matcher, coarse_gt_points1,
            #     result["coarse_cls_idxes"], s2, self.net.reg_w))
        elif self.net.type == "two_stage":
            supervision = utils.create_coarse_supervision(
                batch, s1, extra_scale=s0)
            result = self.net(
                batch, gt_idxes=supervision["coarse_gt_idxes"],
                extra_gt_idxes=supervision.get("extra_coarse_gt_idxes"))
            supervision.update(utils.create_fine_supervision(
                batch, (s1, s2), result["coarse_cls_idxes"],
                offset=self.net.fine_cls_matching.cls_offset, return_coor=True))
            supervision["fine_gt_biases"] = utils.compute_reg_gt_biases(
                supervision.pop("points0_to_1"), supervision.pop("points1"),
                result["fine_cls_idxes"], s2, self.net.reg_w)
        else:
            assert False
        loss = self.loss(
            **result, **supervision, mask0=batch.get(f"mask0_{s1}x"),
            mask1=batch.get(f"mask1_{s1}x"),
            extra_mask0=batch.get(f"mask0_{s0}x"),
            extra_mask1=batch.get(f"mask1_{s0}x"),
            loss_type=loss_type)
        return result, loss

    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int
    ) -> Dict[str, Any]:
        total_loss = 0
        for name, sub_batch in batch.items():
            if name == "megadepth":
                loss_weight = 1.0
                loss_type = "full"
                result, loss = self.model_step(sub_batch, loss_type=loss_type)
            elif name == "scannet":
                loss_weight = 0.5
                loss_type = "only_coarse"
                result, loss = self.model_step_by_or(sub_batch, loss_type=loss_type)
            else:
                raise ValueError()

            for k, v in loss.pop("scalar").items():
                self.log(f"train_scalar_{name}/" + k, v)
            total_loss += loss_weight * loss["loss"]

        if (self.hparams.train_plot_enabled and
            self.trainer.global_rank == 0 and
            self.trainer._logger_connector.should_update_logs):
            error = utils.compute_error(
                batch, result, self.hparams.pose_ransac_count)
            figures = utils.plot_evaluation_figures(
                batch, result, error, self.hparams.epipolar_thresholds[0])
            self.logger.experiment.add_figure(
                "train_plot", figures, global_step=self.global_step)
        return total_loss

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
            batch, result, self.hparams.pose_ransac_count,
            advanced=self.hparams.advanced_metrics,
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

        if self.hparams.test_fp16_precision:
            self.net = self.net.half()

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
                batch, result, self.hparams.pose_ransac_count,
                enable_loransac=self.hparams.test_enable_loransac,
                advanced=self.hparams.advanced_metrics,
                coarse_scale=self.net.scales[0])

        dump = {}
        if self.hparams.dump_dir is not None:
            dump = error
            for k in ("points0", "points1", "scores"):
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
