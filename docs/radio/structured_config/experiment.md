Module radio.structured_config.experiment
=========================================

Functions
---------

    
`register_configs() ‑> None`
:   

Classes
-------

`ExperimentConfig(phase: str = '???', debug: bool = '???', dataset: radio.structured_config.dataset.DatasetConfig = '???', optimizer: radio.structured_config.optimizer.OptimizerConfig = '???')`
:   ExperimentConfig(phase: str = '???', debug: bool = '???', dataset: radio.structured_config.dataset.DatasetConfig = '???', optimizer: radio.structured_config.optimizer.OptimizerConfig = '???')

    ### Descendants

    * radio.structured_config.experiment.PilotConfig

    ### Class variables

    `dataset: radio.structured_config.dataset.DatasetConfig`
    :

    `debug: bool`
    :

    `optimizer: radio.structured_config.optimizer.OptimizerConfig`
    :

    `phase: str`
    :

`PilotConfig(phase: str = 'train', debug: bool = False, dataset: radio.structured_config.dataset.DatasetConfig = MNISTConfig(_target_='torchvision.datasets.MNIST', root='/data/mnist', train=True, download=True), optimizer: radio.structured_config.optimizer.OptimizerConfig = AdamConfig(name='adam', lr=0.001, beta=0.01))`
:   PilotConfig(phase: str = 'train', debug: bool = False, dataset: radio.structured_config.dataset.DatasetConfig = MNISTConfig(_target_='torchvision.datasets.MNIST', root='/data/mnist', train=True, download=True), optimizer: radio.structured_config.optimizer.OptimizerConfig = AdamConfig(name='adam', lr=0.001, beta=0.01))

    ### Ancestors (in MRO)

    * radio.structured_config.experiment.ExperimentConfig

    ### Class variables

    `dataset: radio.structured_config.dataset.DatasetConfig`
    :

    `debug: bool`
    :

    `optimizer: radio.structured_config.optimizer.OptimizerConfig`
    :

    `phase: str`
    :