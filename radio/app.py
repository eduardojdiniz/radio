#!/usr/bin/env python
# coding=utf-8
"""
Package entry point
"""

import hydra
from hydra.utils import instantiate
import radio as rio  # type: ignore
from radio.structured_config.experiment import ExperimentConfig as ExpConfig  # type: ignore # noqa

rio.structured_config.register_configs()


@hydra.main(config_path="../conf", config_name="config")
def app(cfg: ExpConfig) -> None:
    """hydra entry point"""
    dataset = instantiate(cfg.dataset)
    print(dataset)
    data = rio.data.MedicalDecathlonDataModule()
    data.prepare_data()
    data.setup()
    print(f'Training: {len(data.train_set)}')
    print(f'Validation: {len(data.val_set)}')
    print(f'Test: {len(data.test_set)}')


if __name__ == "__main__":
    app()  # pylint: disable = no-value-for-parameter
