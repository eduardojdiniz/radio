Module radio.structured_config.dataset
======================================

Functions
---------

    
`register_configs() ‑> None`
:   

Classes
-------

`Cifar10Config(name: dataclasses.InitVar[str] = 'cifar10', train: bool = True, download: bool = True)`
:   Cifar10Config(_target_: str = 'torchvision.datasets.CIFAR10', name: dataclasses.InitVar[str] = 'cifar10', train: bool = True, download: bool = True)

    ### Ancestors (in MRO)

    * radio.structured_config.dataset.DatasetConfig

    ### Class variables

    `name: dataclasses.InitVar[str]`
    :

`DatasetConfig(name: dataclasses.InitVar[str] = '???', train: bool = True, download: bool = True)`
:   DatasetConfig(_target_: str = '???', name: dataclasses.InitVar[str] = '???', train: bool = True, download: bool = True)

    ### Descendants

    * radio.structured_config.dataset.Cifar10Config
    * radio.structured_config.dataset.MNISTConfig

    ### Class variables

    `download: bool`
    :

    `name: dataclasses.InitVar[str]`
    :

    `root: str`
    :

    `train: bool`
    :

`MNISTConfig(name: dataclasses.InitVar[str] = 'mnist', train: bool = True, download: bool = True)`
:   MNISTConfig(_target_: str = 'torchvision.datasets.MNIST', name: dataclasses.InitVar[str] = 'mnist', train: bool = True, download: bool = True)

    ### Ancestors (in MRO)

    * radio.structured_config.dataset.DatasetConfig

    ### Class variables

    `name: dataclasses.InitVar[str]`
    :