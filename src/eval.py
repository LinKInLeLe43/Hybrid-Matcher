import hydra
import pytorch_lightning as pl
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

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)


@hydra.main(
    config_path="../configs", config_name="eval.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


if __name__ == "__main__":
    main()
