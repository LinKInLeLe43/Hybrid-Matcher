import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def evaluate(cfg: DictConfig) -> None:
    assert cfg.ckpt_path

    pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(cfg.trainer)

    log.info("Starting testing")
    if cfg.ckpt_path.endswith(".pth"):
        state_dict = torch.load(cfg.ckpt_path)
        for k in list(state_dict.keys()):
            if "backbone" in k:
                state_dict[k.replace("backbone", "low_level_encoder", 1)] = (
                    state_dict.pop(k)
                )
        for k in list(state_dict.keys()):
            if "local_coc" in k:
                state_dict[k.replace("local_coc", "high_level_encoder", 1)] = (
                    state_dict.pop(k)
                )
        for k in list(state_dict.keys()):
            if k.startswith("net.low_level_encoder."):
                state_dict[
                    k.replace("layer0", "stage1", 1)
                    .replace("layer1", "stage2", 1)
                    .replace("layer2", "stage3", 1)
                    .replace("branch_3x3", "rbr_dense", 1)
                    .replace("branch_1x1", "rbr_1x1", 1)
                    .replace("branch_identity", "rbr_identity", 1)
                ] = state_dict.pop(k)
        for k in list(state_dict.keys()):
            if "mlp.linear0" in k:
                state_dict[k.replace("mlp.linear0", "mlp.0", 1)] = (
                    state_dict.pop(k)
                )
            if "mlp.linear1" in k:
                state_dict[k.replace("mlp.linear1", "mlp.2", 1)] = (
                    state_dict.pop(k)
                )
            if "mlp3x3.linear" in k:
                state_dict[k.replace("mlp3x3.linear", "linear", 1)] = (
                    state_dict.pop(k)
                )
            if "mlp3x3.conv" in k:
                state_dict[k.replace("mlp3x3.conv", "conv", 1)] = (
                    state_dict.pop(k)
                )
        for k in list(state_dict.keys()):
            if "point_reducers" in k:
                state_dict[k.replace("point_reducers", "patch_embeds", 1)] = (
                    state_dict.pop(k)
                )
            if "down_q" in k:
                state_dict[k.replace("down_q", "patch_embed0", 1)] = (
                    state_dict.pop(k)
                )
            if "down_kv" in k:
                state_dict[k.replace("down_kv", "patch_embed1", 1)] = (
                    state_dict.pop(k)
                )
            if "merge." in k:
                state_dict[k.replace("merge", "out_proj", 1)] = (
                    state_dict.pop(k)
                )
            if "x_up" in k:
                state_dict[k.replace("x_up", "proj0", 1)] = (
                    state_dict.pop(k)
                )
            if "y_up" in k:
                state_dict[k.replace("y_up", "proj1", 1)] = (
                    state_dict.pop(k)
                )
            if "down." in k:
                state_dict[k.replace("down", "fusion", 1)] = (
                    state_dict.pop(k)
                )
        for k in list(state_dict.keys()):
            if "global_blocks" in k:
                state_dict[k.replace("global_blocks", "cross_coc_blocks", 1)] = (
                    state_dict.pop(k)
                )
            if "self_blocks" in k:
                state_dict[k.replace("self_blocks", "self_attention_blocks", 1)] = (
                    state_dict.pop(k)
                )
            if "cross_blocks" in k:
                state_dict[k.replace("cross_blocks", "cross_attention_blocks", 1)] = (
                    state_dict.pop(k)
                )
        for k in list(state_dict.keys()):
            if "cluster." in k:
                state_dict[k.replace("cluster", "context_cluster", 1)] = (
                    state_dict.pop(k)
                )
        for k in list(state_dict.keys()):
            if "norm1" in k and ("high_level_encoder" in k or "cross_coc_blocks" in k):
                state_dict[k.replace("norm1", "norm2", 1)] = (
                    state_dict.pop(k)
                )
        for k in list(state_dict.keys()):
            if "norm0" in k and ("high_level_encoder" in k or "cross_coc_blocks" in k):
                state_dict[k.replace("norm0", "norm1", 1)] = (
                    state_dict.pop(k)
                )
        for k in list(state_dict.keys()):
            if "ups" in k:
                state_dict[k.replace("ups", "projs", 1)] = (
                    state_dict.pop(k)
                )
            if "downs" in k:
                state_dict[k.replace("downs", "fusions", 1)] = (
                    state_dict.pop(k)
                )
            if "fused_selective_module" in k:
                state_dict[k.replace("fused_selective_module", "region_selective_module", 1)] = (
                    state_dict.pop(k)
                )

        model.load_state_dict(state_dict)
        cfg.ckpt_path = None
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


@hydra.main(
    config_path="../configs", config_name="eval.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
