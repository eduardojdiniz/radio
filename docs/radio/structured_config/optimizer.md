Module radio.structured_config.optimizer
========================================

Functions
---------

    
`register_configs() ‑> None`
:   

Classes
-------

`AdamConfig(name: str = 'adam', lr: float = 0.001, beta: float = 0.01)`
:   AdamConfig(name: str = 'adam', lr: float = 0.001, beta: float = 0.01)

    ### Ancestors (in MRO)

    * radio.structured_config.optimizer.OptimizerConfig

    ### Class variables

    `beta: float`
    :

    `lr: float`
    :

    `name: str`
    :

`NesterovConfig(name: str = 'nesterov', lr: float = 0.01)`
:   NesterovConfig(name: str = 'nesterov', lr: float = 0.01)

    ### Ancestors (in MRO)

    * radio.structured_config.optimizer.OptimizerConfig

    ### Class variables

    `name: str`
    :

`OptimizerConfig(name: str = '???', lr: float = 0.01)`
:   OptimizerConfig(name: str = '???', lr: float = 0.01)

    ### Descendants

    * radio.structured_config.optimizer.AdamConfig
    * radio.structured_config.optimizer.NesterovConfig

    ### Class variables

    `lr: float`
    :

    `name: str`
    :