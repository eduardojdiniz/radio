#!/usr/bin/env python
# coding=utf-8


from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from . import optimizer as opt
from . import dataset as ds


@dataclass
class ExperimentConfig:
    phase: str = MISSING
    debug: bool = MISSING
    dataset: ds.DatasetConfig = MISSING
    optimizer: opt.OptimizerConfig = MISSING


@dataclass
class PilotConfig(ExperimentConfig):
    phase: str = "train"
    debug: bool = False
    dataset: ds.DatasetConfig = ds.MNISTConfig()
    optimizer: opt.OptimizerConfig = opt.AdamConfig()


def register_configs() -> None:
    config_store = ConfigStore.instance()
    config_store.store(
        group="structured_config/experiment",
        name="base",
        node=ExperimentConfig,
    )
    config_store.store(
        group="structured_config/experiment",
        name="pilot",
        node=PilotConfig,
    )
