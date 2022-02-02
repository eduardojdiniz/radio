#!/usr/bin/env python
# coding=utf-8

from . import optimizer as opt
from . import dataset as ds
from . import experiment as exp


def register_configs() -> None:
    """
    dataset register its configs in `structured_config/dataset`.
    optimizer register its configs in `structured_config/optimizer`.
    experiment register its configs in `structured_config/experiment`.
    """
    ds.register_configs()
    opt.register_configs()
    exp.register_configs()
