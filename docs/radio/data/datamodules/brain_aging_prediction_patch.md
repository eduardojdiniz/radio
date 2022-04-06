Module radio.data.datamodules.brain_aging_prediction_patch
==========================================================
Based on BaseDataModule for managing data. A vision datamodule that is
shareable, reusable class that encapsulates all the steps needed to process
data, i.e., decoupling datasets from models to allow building dataset-agnostic
models. They also allow you to share a full dataset without explaining how to
download, split, transform, and process the data.

Classes
-------

`BrainAgingPredictionPatchDataModule(*args: Any, root: Union[str, pathlib.Path] = WindowsPath('/media/cerebro'), study: str = 'Brain_Aging_Prediction', data_dir: str = 'Public/data', step: str = 'step01_structural_processing', train_transforms: Optional[torchio.transforms.transform.Transform] = None, val_transforms: Optional[torchio.transforms.transform.Transform] = None, test_transforms: Optional[torchio.transforms.transform.Transform] = None, use_augmentation: bool = True, use_preprocessing: bool = True, resample: bool = False, patch_size: Union[int, Tuple[int, int, int]] = 96, probability_map: Optional[str] = None, label_name: Optional[str] = None, label_probabilities: Optional[Dict[int, float]] = None, queue_max_length: int = 256, samples_per_volume: int = 16, batch_size: int = 32, shuffle_subjects: bool = True, shuffle_patches: bool = True, num_workers: int = 0, pin_memory: bool = True, start_background: bool = True, drop_last: bool = False, num_folds: int = 2, val_split: Union[int, float] = 0.2, intensities: Optional[List[str]] = None, labels: Optional[List[str]] = None, dims: Tuple[int, int, int] = (160, 192, 160), seed: int = 41, verbose: bool = False, **kwargs: Any)`
:   Base class For making patch-based datasets which are compatible with
    torchvision.
    
    To create a subclass, you need to implement the following functions:
    
    A VisionPatchDataModule needs to implement 2 key methods +
    an optional __init__:
    <__init__>:
        (Optionally) Initialize the class, first call super.__init__().
    <default_transforms>:
        Default transforms to use in lieu of train_transforms, val_transforms,
        or test_transforms.
    <teardown>:
        Things to do on every accelerator in distributed mode when finished.
    
    Typical Workflow
    ----------------
    data = VisionPatchDataModule()
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
    patch_size : int or (int, int, int)
        Tuple of integers ``(w, h, d)`` to generate patches of size ``w x h x
        d``. If a single number ``n`` is provided, ``w = h = d = n``.
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
    probability_map : str, optional
        Name of the image in the input subject that will be used as a sampling
        probability map.  The probability of sampling a patch centered on a
        specific voxel is the value of that voxel in the probability map. The
        probabilities need not be normalized. For example, voxels can have
        values 0, 1 and 5. Voxels with value 0 will never be at the center of a
        patch. Voxels with value 5 will have 5 times more chance of being at
        the center of a patch that voxels with a value of 1. If ``None``,
        uniform sampling is used. Default = ``None``.
    label_name : str, optional
        Name of the label image in the subject that will be used to generate
        the sampling probability map. If ``None`` and ``probability_map`` is
        ``None``, the first image of type ``torchio.LABEL`` found in the
        subject subject will be used. If ``probability_map`` is not ``None``,
        then ``label_name`` and ``label_probability`` are ignored.
        Default = ``None``.
    label_probabilities : Dict[int, float], optional
        Dictionary containing the probability that each class will be sampled.
        Probabilities do not need to be normalized. For example, a value of
        {0: 0, 1: 2, 2: 1, 3: 1} will create a sampler whose patches centers
        will have 50% probability of being labeled as 1, 25% of being 2 and 25%
        of being 3. If None, the label map is binarized and the value is set to
        {0: 0, 1: 1}. If the input has multiple channels, a value of
        {0: 0, 1: 2, 2: 1, 3: 1} will create a sampler whose patches centers
        will have 50% probability of being taken from a non zero value of
        channel 1, 25% from channel 2 and 25% from channel 3. If
        ``probability_map`` is not ``None``, then ``label_name`` and
        ``label_probability`` are ignored. Default = ``None``.
    queue_max_length : int, optional
        Maximum number of patches that can be stored in the queue. Using a
        large number means that the queue needs to be filled less often, but
        more CPU memory is needed to store the patches. Default = ``256``.
    samples_per_volume : int, optional
        Number of patches to extract from each volume. A small number of
        patches ensures a large variability in the queue, but training will be
        slower. Default = ``16``.
    batch_size : int, optional
        How many patches per batch to load. Default = ``32``.
    shuffle_subjects : bool, optional
        Whether to shuffle the subjects dataset at the beginning of every epoch
        (an epoch ends when all the patches from all the subjects have been
        processed). Default = ``True``.
    shuffle_patches : bool, optional
        Whether to shuffle the patches queue at the beginning of every epoch.
        Default = ``True``.
    num_workers : int, optional
        How many subprocesses to use for data loading. ``0`` means that the
        data will be loaded in the main process. Default: ``0``.
    pin_memory : bool, optional
        If ``True``, the data loader will copy Tensors into CUDA pinned memory
        before returning them.
    start_background : bool, optional
        If ``True``, the loader will start working in the background as soon as
        the queues are instantiated. Default = ``True``.
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

    * radio.data.datamodules.brain_aging_prediction.BrainAgingPredictionDataModule
    * radio.data.visiondatamodule.VisionDataModule
    * radio.data.basedatamodule.BaseDataModule
    * pytorch_lightning.core.datamodule.LightningDataModule
    * pytorch_lightning.core.hooks.CheckpointHooks
    * pytorch_lightning.core.hooks.DataHooks
    * pytorch_lightning.core.mixins.hparams_mixin.HyperparametersMixin

    ### Class variables

    `intensity2template`
    :

    `label2template: Dict[str, string.Template]`
    :