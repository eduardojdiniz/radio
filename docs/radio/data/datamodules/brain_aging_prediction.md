Module radio.data.datamodules.brain_aging_prediction
====================================================
Brain Aging Prediction Data Module

Classes
-------

`BrainAgingPredictionDataModule(*args: Any, root: Union[str, pathlib.Path] = WindowsPath('/media/cerebro/Studies'), study: str = 'Brain_Aging_Prediction', data_dir: str = 'Public/data', step: str = 'step01_structural_processing', train_transforms: Optional[torchio.transforms.transform.Transform] = None, val_transforms: Optional[torchio.transforms.transform.Transform] = None, test_transforms: Optional[torchio.transforms.transform.Transform] = None, use_augmentation: bool = True, use_preprocessing: bool = True, resample: bool = False, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0, pin_memory: bool = True, drop_last: bool = False, num_folds: int = 2, val_split: Union[int, float] = 0.2, intensities: Optional[List[str]] = None, labels: Optional[List[str]] = None, dims: Tuple[int, int, int] = (160, 192, 160), seed: int = 41, verbose: bool = False, **kwargs: Any)`
:   Brain Aging Prediction Data Module.
    
    Typical Workflow
    ----------------
    data = BrainAgingPredictionDataModule()
    data.prepare_data() # download
    data.setup(stage) # process and split
    data.teardown(stage) # clean-up
    
    Parameters
    ----------
    root : Path or str, optional
        Root to GPN's CEREBRO Studies folder.
        Default = ``'/media/cerebro/Studies'``.
    study : str, optional
        Study name. Default = ``'Brain_Aging_Prediction'``.
    data_dir : str, optional
        Subdirectory where the data is located.
        Default = ``'Public/data'``.
    step : str, optional
        Which processing step to use.
        Default = ``'step01_structural_processing'``.
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
    use_preprocessing : bool, optional
        If ``True``, preprocess samples. Default = ``True``.
    resample : bool, optional
        If ``True``, resample all images to ``'T1'``. Default = ``False``.
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
    intensities : List[str], optional
        Which intensities to load. Default = ``['T1']``.
    labels : List[str], optional
        Which labels to load. Default = ``[]``.
    dims : Tuple[int, int, int], optional
        Max spatial dimensions across subjects' images. If ``None``, compute
        dimensions from dataset. Default = ``(160, 192, 160)``.
    seed : int, optional
        When `shuffle` is True, `seed` affects the ordering of the indices,
        which controls the randomness of each fold. It is also use to seed the
        RNG used by RandomSampler to generate random indexes and
        multiprocessing to generate `base_seed` for workers. Pass an int for
        reproducible output across multiple function calls. Default = ``41``.
    verbose : bool, optional
        If ``True``, print debugging messages. Default = ``False``.

    ### Ancestors (in MRO)

    * radio.data.visiondatamodule.VisionDataModule
    * radio.data.basedatamodule.BaseDataModule
    * pytorch_lightning.core.datamodule.LightningDataModule
    * pytorch_lightning.core.hooks.CheckpointHooks
    * pytorch_lightning.core.hooks.DataHooks
    * pytorch_lightning.core.mixins.hparams_mixin.HyperparametersMixin

    ### Descendants

    * radio.data.datamodules.brain_aging_prediction_patch.BrainAgingPredictionPatchDataModule

    ### Class variables

    `intensity2template`
    :

    `label2template: Dict[str, string.Template]`
    :

    ### Static methods

    `get_augmentation_transforms() ‑> torchio.transforms.transform.Transform`
    :   "
        Get augmentation transorms to apply to subjects during training.
        
        Returns
        -------
        augment : tio.Transform
            All augmentation steps that should be applied to subjects during
            training.

    `get_paths(data_root: Union[str, pathlib.Path], stem: str = 'step01_structural_processing', has_train_test_split: bool = False, has_train_val_split: bool = False, test_split: Union[int, float] = 0.2, shuffle: bool = True, seed: int = 41) ‑> Tuple[collections.OrderedDict[Tuple[str, str], pathlib.Path], collections.OrderedDict[Tuple[str, str], pathlib.Path], collections.OrderedDict[Tuple[str, str], pathlib.Path]]`
    :   Get subject and scan IDs and the respective paths from the study data
        directory.
        
        Returns
        -------
        _ : {(str, str): Path}, {(str, str): Path}, {(str, str): Path}
            Paths for respectively, train, test and images and labels.

    ### Methods

    `default_transforms(self, stage: Optional[str] = None) ‑> torchio.transforms.transform.Transform`
    :   Default transforms and augmentations for the dataset.
        
        Parameters
        ----------
        stage: Optional[str]
            Either ``'fit``, ``'validate'``, ``'test'``, or ``'predict'``.
            If stage = None, set-up all stages. Default = None.
        
        Returns
        -------
        _: tio.Transform
            All preprocessing steps (and if ``'fit'``, augmentation steps too)
            that should be applied to the subjects.

    `get_preprocessing_transforms(self, shape: Optional[Tuple[int, int, int]] = None, resample: bool = False) ‑> torchio.transforms.transform.Transform`
    :   Get preprocessing transorms to apply to all subjects.
        
        Returns
        -------
        preprocess : tio.Transform
            All preprocessing steps that should be applied to all subjects.

    `get_subjects(self, fold: str = 'train') ‑> List[torchio.data.subject.Subject]`
    :   Get train, test, or val list of TorchIO Subjects.
        
        Parameters
        ----------
        fold : str, optional
            Identify which type of dataset, ``'train'``, ``'test'``, or
            ``'val'``. Default = ``'train'``.
        
        Returns
        -------
        _ : List[tio.Subject]
            Train, test or val list of TorchIO Subjects.

    `get_subjects_dicts(self, intensities: Optional[List[str]] = None, labels: Optional[List[str]] = None) ‑> Tuple[collections.OrderedDict[Tuple[str, str], collections.OrderedDict[str, Any]], collections.OrderedDict[Tuple[str, str], collections.OrderedDict[str, Any]], collections.OrderedDict[Tuple[str, str], collections.OrderedDict[str, Any]]]`
    :   Get paths to nii files for train/test/val images and labels.
        
        Returns
        -------
        _ : {(str, str): Dict}, {(str, str): Dict}, {(str, str): Dict}
            Paths to, respectively, train, test, and val images and labels.

    `prepare_data(self, *args: Any, **kwargs: Any) ‑> None`
    :   Verify data directory exists and if test/train/val splitted.

    `save(self, dataloader: torch.utils.data.dataloader.DataLoader, root: Union[str, pathlib.Path] = WindowsPath('/media/cerebro/Workspaces/Students/Eduardo_Diniz/Studies'), data_dir: str = 'processed_data', step: str = 'step01_structural_processing', fold: str = 'train') ‑> None`
    :   Arguments
        ---------
        root : Path or str, optional
            Root where to save data. Default = ``'~/LocalCerebro'``.

    `setup(self, stage: Optional[str] = None) ‑> None`
    :   Creates train, validation and test collection of samplers.
        
        Parameters
        ----------
        stage: Optional[str]
            Either ``'fit``, ``'validate'``, or ``'test'``.
            If stage = ``None``, set-up all stages. Default = ``None``.