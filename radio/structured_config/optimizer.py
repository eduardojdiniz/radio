#!/usr/bin/env python
# coding=utf-8

from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class OptimizerConfig:
    name: str = MISSING
    lr: float = 0.01


@dataclass
class NesterovConfig(OptimizerConfig):
    name: str = "nesterov"


@dataclass
class AdamConfig(OptimizerConfig):
    name: str = "adam"
    lr: float = 0.001
    beta: float = 0.01


def register_configs() -> None:
    config_store = ConfigStore.instance()
    config_store.store(
        group="structured_config/optimizer",
        name="base",
        node=OptimizerConfig,
    )
    config_store.store(
        group="structured_config/optimizer",
        name="nesterov",
        node=NesterovConfig(),
    )
    config_store.store(
        group="structured_config/optimizer",
        name="adam",
        node=AdamConfig(),
    )
