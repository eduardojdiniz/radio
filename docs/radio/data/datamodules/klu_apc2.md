Module radio.data.datamodules.klu_apc2
======================================
KLU_APC2 Data Module

Functions
---------

    
`plot_klu(batch: dict, num_imgs: int = 5, slice_num: int = 150, train: bool = True) ‑> None`
:   plot images and labels from a batch of train images

Classes
-------

`KLUAPC2DataModule(*args: Any, root: Union[str, pathlib.Path] = WindowsPath('/media/cerebro/Studies/KLU_APC2/Public/Analysis/data'), step: str = 'step02_structural_processing', train_transforms: Optional[Callable] = None, val_transforms: Optional[Callable] = None, test_transforms: Optional[Callable] = None, use_augmentation: bool = True, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0, pin_memory: bool = True, drop_last: bool = False, num_folds: int = 2, val_split: Union[int, float] = 0.2, modalities: List[str] = ['T1w', 'FLAIR', 'T2w'], labels: List[str] = ['WMH'], dims: List[int] = [256, 256, 256], seed: int = 41, **kwargs: Any)`
:   KLU APC2 Data Module.
    
    Typical Workflow
    ----------------
    klu = KLUAPC2DataModule()
    klu.prepare_data() # download
    klu.setup(stage) # process and split
    klu.teardown(stage) # clean-up
    
    Parameters
    ----------
    root : Path or str, optional
        Root directory of dataset.
        Default = ``''/media/cerebro/Studies/KLU_APC2/Public/Analysis/data''``.
    step : str, optional
        Which processing step to use.
        Default = ``''step02_structural_processing''``.
    train_transforms : Callable, optional
        A function/transform that takes in a sample and returns a
        transformed version, e.g, ``torchvision.transforms.RandomCrop``.
    val_transforms : Callable, optional
        A function/transform that takes in a sample and returns a
        transformed version, e.g, ``torchvision.transforms.RandomCrop``.
    test_transforms : Callable, optional
        A function/transform that takes in a sample and returns a
        transformed version, e.g, ``torchvision.transforms.RandomCrop``.
    use_augmentation : bool, optional
        If ``True``, augment samples during the ``fit`` stage.
        Default = ``True``.
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
    val_split : int or float, optional
        If ``num_folds = 2``, then ``val_split`` specify how the
        train_dataset should be split into train/validation datasets. If
        ``num_folds > 2``, then it is not used. Default = ``0.2``.
    dims : List[int], optional
        Max spatial dimensions across subjects' images.
        Default = ``[320, 256, 256]``.
    seed : int, optional
        When `shuffle` is True, `seed` affects the ordering of the indices,
        which controls the randomness of each fold. It is also use to seed the
        RNG used by RandomSampler to generate random indexes and
        multiprocessing to generate `base_seed` for workers. Pass an int for
        reproducible output across multiple function calls. Default = ``41``.

    ### Ancestors (in MRO)

    * radio.data.visiondatamodule.VisionDataModule
    * radio.data.basedatamodule.BaseDataModule
    * pytorch_lightning.core.datamodule.LightningDataModule
    * pytorch_lightning.core.hooks.CheckpointHooks
    * pytorch_lightning.core.hooks.DataHooks
    * pytorch_lightning.core.mixins.hparams_mixin.HyperparametersMixin

    ### Static methods

    `get_augmentation_transforms() ‑> Callable`
    :   "
        Get augmentation transorms to apply to subjects during training.
        
        Returns
        -------
        augment : tio.Compose
            All augmentation steps that should be applied to subjects during
            training.

    `split_dict(dictionary: collections.OrderedDict, test_split: Union[int, float] = 0.2, shuffle: bool = True) ‑> Tuple[collections.OrderedDict, collections.OrderedDict]`
    :   Split dict into two.

    ### Methods

    `default_transforms(self, stage: Optional[str] = None) ‑> Callable`
    :   Default transforms and augmentations for the dataset.
        
        Parameters
        ----------
        stage: Optional[str]
            Either ``'fit``, ``'validate'``, ``'test'``, or ``'predict'``.
            If stage = None, set-up all stages. Default = None.
        
        Returns
        -------
        _: Callable
            All preprocessing steps (and if ``'fit'``, augmentation steps too)
            that should be applied to the subjects.

    `get_max_shape(self, subjects: List[torchio.data.subject.Subject]) ‑> List[int]`
    :   Get max shape.
        
        Parameters
        ----------
        subjects : List[tio.Subject]
            List of TorchIO Subject objects.
        
        Returns
        -------
        _ : np.ndarray((1, 3), np.int_)
            Max height, width and depth across all subjects.

    `get_paths(self) ‑> collections.OrderedDict`
    :   Get subject and scan IDs and the respective paths from the study data
        directory.
        
        Returns
        -------
        _ : Tuple[List[Path], List[Path]]
            Paths to train images and labels.

    `get_preprocessing_transforms(self, size: Optional[List[int]] = [256, 256, 256], t2w_present: bool = False, flair_present: bool = False) ‑> Callable`
    :   Get preprocessing transorms to apply to all subjects.
        
        Returns
        -------
        preprocess : tio.Compose
            All preprocessing steps that should be applied to all subjects.

    `get_subject_dicts(self, step: str = 'step06_WMHz_new', modalities: List[str] = ['T1w', 'FLAIR'], labels: List[str] = ['WMH']) ‑> Tuple[collections.OrderedDict[Tuple[str, str], dict], collections.OrderedDict[Tuple[str, str], dict]]`
    :   Get paths to nii files for train images and labels.
        
        Returns
        -------
        _ : Tuple[List[Path], List[Path]]
            Paths to train images and labels.

    `get_subjects(self, train: bool = True) ‑> List[torchio.data.subject.Subject]`
    :   Get TorchIO Subject train and test subjects.
        
        Parameters
        ----------
        train : bool, optional
            If True, return a loader for the train dataset, else for the
            validation dataset. Default = ``True``.
        
        Returns
        -------
        _ : List[tio.Subject]
            TorchIO Subject train or test subjects.

    `prepare_data(self, *args: Any, **kwargs: Any) ‑> None`
    :   Verify data directory exists.

    `setup(self, stage: Optional[str] = None) ‑> None`
    :   Creates train, validation and test collection of samplers.
        
        Parameters
        ----------
        stage: Optional[str]
            Either ``'fit``, ``'validate'``, ``'test'``, or ``'predict'``.
            If stage = None, set-up all stages. Default = None.