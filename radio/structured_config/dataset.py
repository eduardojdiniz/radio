#!/usr/bin/env python
# coding=utf-8

from dataclasses import dataclass, field, InitVar
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DatasetConfig:
    _target_: str = MISSING
    name: InitVar[str] = MISSING
    root: str = field(init=False)
    train: bool = True
    download: bool = True

    def __post_init__(self, name):
        self.root = f"/data/{name}"


@dataclass
class Cifar10Config(DatasetConfig):
    _target_: str = "torchvision.datasets.CIFAR10"
    name: InitVar[str] = "cifar10"


@dataclass
class MNISTConfig(DatasetConfig):
    _target_: str = "torchvision.datasets.MNIST"
    name: InitVar[str] = "mnist"


def register_configs() -> None:
    config_store = ConfigStore.instance()
    config_store.store(
        group="structured_config/dataset",
        name="base",
        node=DatasetConfig,
    )
    # Use a instance in node, so uninitialized fields get resolved, e.g., path.
    config_store.store(
        group="structured_config/dataset",
        name="cifar10",
        node=Cifar10Config(),
    )
    config_store.store(
        group="structured_config/dataset",
        name="mnist",
        node=MNISTConfig(),
    )
