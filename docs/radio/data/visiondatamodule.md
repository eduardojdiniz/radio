Module radio.data.visiondatamodule
==================================
Based on LightningDataModule for managing data. A datamodule is a shareable,
reusable class that encapsulates all the steps needed to process data, i.e.,
decoupling datasets from models to allow building dataset-agnostic models. They
also allow you to share a full dataset without explaining how to download,
split, transform, and process the data.

Classes
-------

`VisionDataModule(*args: Any, root: Union[str, pathlib.Path] = WindowsPath('C:/Users/LIW82/Lab Work/radio/dataset'), train_transforms: Optional[torchio.transforms.transform.Transform] = None, val_transforms: Optional[torchio.transforms.transform.Transform] = None, test_transforms: Optional[torchio.transforms.transform.Transform] = None, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0, pin_memory: bool = True, drop_last: bool = False, num_folds: int = 2, val_split: Union[int, float] = 0.2, seed: int = 41, **kwargs: Any)`
:   Base class For making datasets which are compatible with torchvision.
    
    To create a subclass, you need to implement the following functions:
    
    A VisionDataModule needs to implement 2 key methods + an optional __init__:
    <__init__>:
        (Optionally) Initialize the class, first call super.__init__().
    <default_transforms>:
        Default transforms to use in lieu of train_transforms, val_transforms,
        or test_transforms.
    <teardown>:
        Things to do on every accelerator in distributed mode when finished.
    
    Typical Workflow
    ----------------
    data = VisionDataModule()
    data.prepare_data() # download
    data.setup(stage) # process and split
    data.teardown(stage) # clean-up
    
    Parameters
    ----------
    root : Path or str, optional
        Root directory of dataset. Default = ``DATA_ROOT``.
    train_transforms : Callable, optional
        A function/transform that takes in a sample and returns a
        transformed version, e.g, ``torchvision.transforms.RandomCrop``.
    val_transforms : Callable, optional
        A function/transform that takes in a sample and returns a
        transformed version, e.g, ``torchvision.transforms.RandomCrop``.
    test_transforms : Callable, optional
        A function/transform that takes in a sample and returns a
        transformed version, e.g, ``torchvision.transforms.RandomCrop``.
    batch_size : int, optional
        How many samples per batch to load. Default = ``32``.
    shuffle : bool, optional
        Whether to shuffle the data at every epoch. Default = ``False``.
    num_workers : int, optional
        How many subprocesses to use for data loading. ``0`` means that the
        data will be loaded in the main process. Default: ``0``.
    pin_memory : bool, optional
        If ``True``, the data loader will copy Tensors into CUDA pinned memory
        before returning them.
    drop_last : bool, optional
        Set to ``True`` to drop the last incomplete batch, if the dataset size
        is not divisible by the batch size. If ``False`` and the size of
        dataset is not divisible by the batch size, then the last batch will be
        smaller. Default = ``False``.
    num_folds : int, optional
        Number of folds. Must be at least ``2``. ``2`` corresponds to a single
        train/validation split. Default = ``2``.
    val_split: int or float, optional
        If ``num_folds = 2``, then ``val_split`` specify how the
        train_dataset should be split into train/validation datasets. If
        ``num_folds > 2``, then it is not used. Default = ``0.2``.
    seed : int, optional
        When `shuffle` is True, `seed` affects the ordering of the indices,
        which controls the randomness of each fold. It is also use to seed the
        RNG used by RandomSampler to generate random indexes and
        multiprocessing to generate `base_seed` for workers. Pass an int for
        reproducible output across multiple function calls. Default = ``41``.

    ### Ancestors (in MRO)

    * radio.data.basedatamodule.BaseDataModule
    * pytorch_lightning.core.datamodule.LightningDataModule
    * pytorch_lightning.core.hooks.CheckpointHooks
    * pytorch_lightning.core.hooks.DataHooks
    * pytorch_lightning.core.mixins.hparams_mixin.HyperparametersMixin

    ### Descendants

    * radio.data.datamodules.brain_aging_prediction.BrainAgingPredictionDataModule
    * radio.data.datamodules.klu_apc2.KLUAPC2DataModule
    * radio.data.datamodules.medical_decathlon.MedicalDecathlonDataModule

    ### Static methods

    `get_max_shape(subjects: List[torchio.data.subject.Subject]) ‑> Tuple[int, int, int]`
    :   Get max height, width, and depth accross all subjects.
        
        Parameters
        ----------
        subjects : List[tio.Subject]
            List of TorchIO Subject objects.
        
        Returns
        -------
        shapes_tuple : Tuple[int, int, int]
            Max height, width and depth across all subjects.

    `size_eval_dataset(eval_dataset: Sized) ‑> Union[int, Sequence[int]]`
    :   Compute the size of the test or validation datasets.
        
        Parameters
        ----------
        eval_dataset: EvalDatasetType
            Collection of test or validation datasets.
        
        Returns
        -------
        _ : EvalSizeType
            Collection of test or validation datasets' sizes.

    `size_train_dataset(train_dataset: Sized) ‑> Union[int, Sequence[int], Sequence[Sequence[int]], Sequence[Dict[str, int]], Dict[str, int], Dict[str, Dict[str, int]], Dict[str, Sequence[int]]]`
    :   Compute the size of the train datasets.
        
        Parameters
        ----------
        train_dataset: TrainDatasetType
            Collection of train datasets.
        
        Returns
        -------
        _ : TrainSizeType
            Collection of train datasets' sizes.

    ### Methods

    `check_if_data_split(self, stem: str = '') ‑> None`
    :   Check if data is splitted in train, test and val folders

    `dataloader(self, dataset: Union[torch.utils.data.dataset.Dataset, radio.data.dataset.BaseVisionDataset], batch_size: Optional[int] = None, shuffle: Optional[bool] = None, num_workers: Optional[int] = None, pin_memory: Optional[bool] = None, drop_last: Optional[bool] = None) ‑> torch.utils.data.dataloader.DataLoader`
    :   Instantiate a DataLoader.
        
        Parameters
        ----------
        batch_size : int, optional
            How many samples per batch to load. Default = ``32``.
        shuffle : bool, optional
            Whether to shuffle the data at every epoch. Default = ``False``.
        num_workers : int, optional
            How many subprocesses to use for data loading. ``0`` means that the
            data will be loaded in the main process. Default: ``0``.
        pin_memory : bool, optional
            If ``True``, the data loader will copy Tensors into CUDA pinned
            memory before returning them.
        drop_last : bool, optional
            Set to ``True`` to drop the last incomplete batch, if the dataset
            size is not divisible by the batch size. If ``False`` and the size
            of dataset is not divisible by the batch size, then the last batch
            will be smaller. Default = ``False``.
        
        Returns
        -------
        _ : DataLoader

    `default_transforms(self, stage: Optional[str] = None) ‑> Callable`
    :   Default transforms and augmentations for the dataset.
        
        Parameters
        ----------
        stage: Optional[str]
            Either ``'fit``, ``'validate'``, or ``'test'``.
            If stage = None, set-up all stages. Default = None.
        
        Returns
        -------
        _: Callable
            All preprocessing steps (and if ``'fit'``, augmentation steps too)
            that should be applied to the images.

    `predict_dataloader(self, *args: Any, **kwargs: Any) ‑> Union[torch.utils.data.dataloader.DataLoader, Sequence[torch.utils.data.dataloader.DataLoader]]`
    :   Implement one or multiple PyTorch DataLoaders for prediction.
        
        It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.
        
        - :meth:`~pytorch_lightning.trainer.Trainer.fit`
        - ...
        - :meth:`prepare_data`
        - :meth:`train_dataloader`
        - :meth:`val_dataloader`
        - :meth:`test_dataloader`
        
        Note:
            Lightning adds the correct sampler for distributed and arbitrary hardware
            There is no need to set it yourself.
        
        Return:
            A :class:`torch.utils.data.DataLoader` or a sequence of them specifying prediction samples.
        
        Note:
            In the case where you return multiple prediction dataloaders, the :meth:`predict`
            will have an argument ``dataloader_idx`` which matches the order here.

    `prepare_data(self, *args: Any, **kwargs: Any) ‑> None`
    :   Saves files to data root dir.
        Verify data directory exists.
        Verify if test/train/val splitted.

    `setup(self, stage: Optional[str] = None) ‑> None`
    :   Creates train, validation and test collection of samplers.
        
        Parameters
        ----------
        stage: Optional[str]
            Either ``'fit``, ``'validate'``, or ``'test'``.
            If stage = None, set-up all stages. Default = None.

    `teardown(self, stage: Optional[str] = None) ‑> None`
    :   Called at the end of fit (train + validate), validate, test,
        or predict. Remove root directory if a temporary was used.
        
        Parameters
        ----------
        stage: Optional[str]
            Either ``'fit``, ``'validate'``, or ``'test'``.
            If stage = None, set-up all stages. Default = None.

    `test_dataloader(self, *args: Any, **kwargs: Any) ‑> Union[torch.utils.data.dataloader.DataLoader, Sequence[torch.utils.data.dataloader.DataLoader]]`
    :   Generates one or multiple Pytorch DataLoaders for test.
        
        Returns
        -------
        _ : Collection of DataLoaders
            Collection of test dataloaders specifying test samples.

    `train_dataloader(self, *args: Any, **kwargs: Any) ‑> Union[torch.utils.data.dataloader.DataLoader, Sequence[torch.utils.data.dataloader.DataLoader], Sequence[Sequence[torch.utils.data.dataloader.DataLoader]], Sequence[Dict[str, torch.utils.data.dataloader.DataLoader]], Dict[str, torch.utils.data.dataloader.DataLoader], Dict[str, Dict[str, torch.utils.data.dataloader.DataLoader]], Dict[str, Sequence[torch.utils.data.dataloader.DataLoader]]]`
    :   Generates one or multiple Pytorch DataLoaders for train.
        
        Returns
        -------
        _ : Collection of DataLoader
            Collection of train dataloaders specifying training samples.

    `val_dataloader(self, *args: Any, **kwargs: Any) ‑> Union[torch.utils.data.dataloader.DataLoader, Sequence[torch.utils.data.dataloader.DataLoader]]`
    :   Generates one or multiple Pytorch DataLoaders for validation.
        
        Returns
        -------
        _ : Collection of DataLoader
            Collection of validation dataloaders specifying validation samples.