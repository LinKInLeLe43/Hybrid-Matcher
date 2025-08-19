import functools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch.distributed import all_gather_object
from torch.nn import Module

# from src.models.nets.casp.homo.utils.dense_match import DenseMatch
from src.models.utils import (
    compute_dense_gt_biases,
    compute_error,
    compute_metric,
    compute_reg_gt_biases,
    create_coarse_supervision,
    create_fine_supervision,
    make_evaluation_figures,
)


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
        loss: Module,
        optimizer: functools.partial,
        scheduler: functools.partial,
        train_batch_size_per_gpu: int,
        canonical_batch_size: int,
        canonical_learning_rate: float,
        canonical_warmup_step_count: int,
        warmup_ratio: float,
        metric_thresholds: Dict[str, Any],
        rel_pose_ransac_config: Dict[str, Any],
        train_plot_enabled: bool = False,
        val_plot_count: int = 32,
        test_task: Optional[str] = None,
        test_fp16_precision: bool = False,
        advanced_metrics: bool = True,
    ) -> None:
        super().__init__()
        assert test_task in [None, "accuracy", "efficiency"]
        self.net = net
        self.loss = loss
        self.save_hyperparameters(ignore=["net", "loss"], logger=False)

        # self.dense_matcher = DenseMatch()
        # self.rel_pose_config = {
        #     "estimator": "opencv",
        #     "num_repeats": 5,
        #     "params": {"method": cv2.RANSAC},
        # }

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

    def on_train_start(self) -> None:
        scale = (
            self.trainer.world_size
            * self.hparams.train_batch_size_per_gpu
            / self.hparams.canonical_batch_size
        )
        self.hparams.optimizer.keywords["lr"] = (
            scale * self.hparams.canonical_learning_rate
        )
        self.hparams["warmup_step_count"] = (
            self.hparams.canonical_warmup_step_count / scale
        )

    def model_step(
        self, batch: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        s1, s2 = self.net.scales
        s0 = s1 * 2

        supervision = create_coarse_supervision(
            batch, s1, extra_scale=s0, return_coor=True
        )
        # coarse_gt_points1 = supervision.pop("gt_points1")
        result = self.net(
            batch,
            gt_idxes=supervision["coarse_gt_idxes"],
            extra_gt_idxes=supervision.get("extra_coarse_gt_idxes"),
        )
        # supervision.update(
        #     create_fine_supervision(
        #         batch,
        #         (s1, 1),
        #         result["coarse_cls_indices"],
        #         offset=0.5,
        #         return_coor=True,
        #     )
        # )
        supervision["gt_mu"] = compute_reg_gt_biases(
            supervision.pop("gt_points0_to_1"),
            supervision.pop("gt_points1_to_0"),
            supervision.pop("gt_points0"),
            supervision.pop("gt_points1"),
            result["coarse_cls_indices"])
        # supervision.update(
        #     compute_dense_gt_biases(
        #         batch,
        #         result,
        #         self.dense_matcher,
        #         coarse_gt_points1,
        #         result["coarse_cls_indices"],
        #         s2,
        #         self.net.fine_reg_matching.window_size,
        #     )
        # )

        loss = self.loss(
            **result,
            **supervision,
            mask0=batch.get(f"mask0_{s1}x"),
            mask1=batch.get(f"mask1_{s1}x"),
            extra_mask0=batch.get(f"mask0_{s0}x"),
            extra_mask1=batch.get(f"mask1_{s0}x"),
        )
        return result, loss

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Dict[str, Any]:
        _, loss = self.model_step(batch)

        for k, v in loss.pop("scalar").items():
            self.log("train_scalar/" + k, v)
        return loss

    def training_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        avg_loss_on_epoch = torch.stack([o["loss"] for o in outputs]).mean()
        self.log(
            "train_scalar/avg_loss_on_epoch", avg_loss_on_epoch, sync_dist=True
        )

    def on_validation_start(self) -> None:
        plot_count_per_rank = (
            self.hparams.val_plot_count // self.trainer.world_size
        )
        intervals = [
            max(batch_count_per_rank // plot_count_per_rank, 1)
            for batch_count_per_rank in self.trainer.num_val_batches
        ]
        self.hparams["val_plot_intervals"] = intervals

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int, dataloader_idx: int = 0
    ) -> Dict[str, Any]:
        result, loss = self.model_step(batch)
        loss = loss.pop("scalar")
        error = compute_error(
            batch,
            result,
            self.hparams.rel_pose_ransac_config,
            advanced=self.hparams.advanced_metrics,
            coarse_scale=self.net.scales[0],
        )
        figures = []
        if batch_idx % self.hparams.val_plot_intervals[0] == 0:
            figures = make_evaluation_figures(
                batch,
                result,
                error,
                self.hparams.metric_thresholds["sym_epi_prec"][0],
            )
        output = {"loss": loss, "error": error, "figures": figures}
        return output

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        outputs_by_ranks = [[] for _ in range(self.trainer.world_size)]
        all_gather_object(outputs_by_ranks, outputs)
        gathered_output = _flatten(outputs_by_ranks)
        del outputs_by_ranks

        for k, v in gathered_output.pop("loss").items():
            self.log("val_scalar/" + k, torch.stack(v).mean())
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
            self.log("val_metric/" + key, value)

        if not self.trainer.sanity_checking:
            figures = np.concatenate(gathered_output.pop("figures"))
            for i, figure in enumerate(figures):
                self.logger.experiment.add_figure(
                    f"val_plot/pair-{i}",
                    figure,
                    global_step=self.trainer.current_epoch,
                )

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

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(self.parameters())
        scheduler = self.hparams.scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler},
        }

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
        if self.trainer.global_step <= self.hparams.warmup_step_count:
            scale = (
                self.hparams.warmup_ratio
                + self.trainer.global_step
                / self.hparams.warmup_step_count
                * (1 - self.hparams.warmup_ratio)
            )
            lr = scale * self.hparams.optimizer.keywords["lr"]
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        optimizer.step(closure=optimizer_closure)
