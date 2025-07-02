import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.utils import (
    RankedLogger,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
)

log = RankedLogger(__name__, rank_zero_only=True)


def train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating callbacks...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(cfg=cfg, model=model, trainer=trainer)

    log.info("Starting training!")
    trainer.fit(
        model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path")
    )


@hydra.main(
    config_path="../configs", config_name="train.yaml", version_base="1.3"
)
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
