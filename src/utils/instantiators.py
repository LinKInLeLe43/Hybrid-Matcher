from typing import List, Optional

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Callback
from pytorch_lightning.loggers import Logger

from .pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(cfg: Optional[DictConfig]) -> List[Callback]:
    callbacks = []
    if cfg is None:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    for item in cfg.values():
        log.info(f"Instantiating callback <{item._target_}>")
        callbacks.append(hydra.utils.instantiate(item))
    return callbacks


def instantiate_loggers(cfg: Optional[DictConfig]) -> List[Logger]:
    logger = []
    if cfg is None:
        log.warning("No callback configs found! Skipping..")
        return logger

    for item in cfg.values():
        log.info(f"Instantiating callback <{item._target_}>")
        logger.append(hydra.utils.instantiate(item))
    return logger
