#!/usr/bin/env python
# coding=utf-8
"""
Based on BaseDataModule for managing data. A vision datamodule that is
shareable, reusable class that encapsulates all the steps needed to process
data, i.e., decoupling datasets from models to allow building dataset-agnostic
models. They also allow you to share a full dataset without explaining how to
download, split, transform, and process the data.
"""

from typing import Any, Callable, Dict, List, Optional, Union, Tuple, cast
from pathlib import Path
import re
import shutil
from collections import OrderedDict
from string import Template
from torch.utils.data import DataLoader
import numpy as np
import torchio as tio  # type: ignore
from radio.settings.pathutils import PathType, is_dir_or_symlink
from .validation import TrainDataLoaderType, EvalDataLoaderType
from .datatypes import SpatialShapeType
from .basedatamodule import BaseDataModule
from .datatypes import SubjPathType, SubjDictType

__all__ = ["VisionPatchDataModule"]


class VisionPatchDataModule(BaseDataModule):
    """
    Base class For making patch-based datasets which are compatible with
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
    """

    #: Extra arguments for dataset_cls instantiation.
    EXTRA_ARGS: dict = {}
    #: Dataset class to use. E.g., torchvision.datasets.MNIST
    dataset_cls = tio.SubjectsDataset
    #: A tuple describing the shape of the data
    dims: Optional[Tuple[int, int, int]]
    #: Dataset name
    name: str = "brain_aging_prediction"
    intensity2template = {
        "T1": Template('wstrip_m${subj_id}_${scan_id}_T1.nii'),
        "FLAIR": Template('wstrip_mr${subj_id}_${scan_id}_FLAIR.nii'),
    }
    label2template: Dict[str, Template] = {}

    def __init__(
        self,
        *args: Any,
        root: PathType = Path('/media/cerebro/Studies'),
        study: str = 'Brain_Aging_Prediction',
        data_dir: str = 'Public/data',
        step: str = 'step01_structural_processing',
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        use_augmentation: bool = True,
        use_preprocessing: bool = True,
        resample: bool = False,
        patch_size: SpatialShapeType = 96,
        probability_map: Optional[str] = None,
        label_name: Optional[str] = None,
        label_probabilities: Optional[Dict[int, float]] = None,
        queue_max_length: int = 256,
        samples_per_volume: int = 16,
        batch_size: int = 32,
        shuffle_subjects: bool = True,
        shuffle_patches: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        start_background: bool = True,
        drop_last: bool = False,
        num_folds: int = 2,
        val_split: Union[int, float] = 0.2,
        intensities: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        dims: Tuple[int, int, int] = (160, 192, 160),
        seed: int = 41,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        root = Path(root).expanduser() / study / data_dir
        super().__init__(
            *args,
            root=root,
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            test_transforms=test_transforms,
            batch_size=batch_size,
            shuffle=shuffle_subjects,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            num_folds=num_folds,
            val_split=val_split,
            seed=seed,
            **kwargs,
        )
        self.study = study
        self.data_dir = data_dir
        self.step = step
        self.intensities = intensities if intensities else ['T1', 'FLAIR']
        self.labels = labels if labels else []
        self.dims = dims
        self.use_augmentation = use_augmentation
        self.use_preprocessing = use_preprocessing
        self.resample = resample
        self.train_sampler: tio.data.sampler.sampler.PatchSampler

        # Init Train Sampler
        both_something = probability_map is not None and label_name is not None
        if both_something:
            raise ValueError(
                "Both 'probability_map' and 'label_name' cannot be not None ",
                "at the same time",
            )
        if probability_map is None and label_name is None:
            self.train_sampler = tio.UniformSampler(patch_size)
        elif probability_map is not None:
            self.train_sampler = tio.WeightedSampler(patch_size,
                                                     probability_map)
        else:
            self.train_sampler = tio.LabelSampler(patch_size, label_name,
                                                  label_probabilities)

        self.probability_map = probability_map
        self.label_name = label_name
        self.label_probabilities = label_probabilities

        # Queue parameters
        self.train_queue: tio.Queue
        self.val_queue: tio.Queue
        self.queue_max_length = queue_max_length
        self.samples_per_volume = samples_per_volume
        self.shuffle_subjects = shuffle_subjects
        self.shuffle_patches = shuffle_patches
        self.start_background = start_background
        self.verbose = verbose

        # Data folder flags to check if data is splitted already
        self.has_complete_split: bool
        self.has_train_val_split: bool
        self.has_train_test_split: bool

    def check_if_data_split(self) -> None:
        """Check if data is splitted in train, test and val folders"""
        has_train_folder = is_dir_or_symlink(self.root / self.step / "train")
        has_test_folder = is_dir_or_symlink(self.root / self.step / "test")
        has_val_folder = is_dir_or_symlink(self.root / self.step / "val")
        self.has_train_test_split = bool(has_train_folder and has_test_folder)
        self.has_train_val_split = bool(has_train_folder and has_val_folder)
        self.has_complete_split = bool(has_train_folder and has_test_folder
                                       and has_val_folder)

    @staticmethod
    def get_max_shape(subjects: List[tio.Subject]) -> Tuple[int, int, int]:
        """
        Get max height, width, and depth accross all subjects.

        Parameters
        ----------
        subjects : List[tio.Subject]
            List of TorchIO Subject objects.

        Returns
        -------
        shapes_tuple : Tuple[int, int, int]
            Max height, width and depth across all subjects.
        """
        dataset = tio.SubjectsDataset(subjects)
        shapes = np.array([
            image.spatial_shape for subject in dataset.dry_iter()
            for image in subject.get_images()
        ])
        shapes_tuple = tuple(map(int, shapes.max(axis=0).tolist()))
        return cast(Tuple[int, int, int], shapes_tuple)

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """Verify data directory exists and if test/train/val splitted."""
        if not is_dir_or_symlink(self.root):
            raise OSError('Study data directory not found!')
        self.check_if_data_split()

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates train, validation and test collection of samplers.

        Parameters
        ----------
        stage: Optional[str]
            Either ``'fit``, ``'validate'``, ``'test'``, or ``'predict'``.
            If stage = ``None``, set-up all stages. Default = ``None``.
        """
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms(
                stage="fit"
            ) if self.train_transforms is None else self.train_transforms

            val_transforms = self.default_transforms(
                stage="fit"
            ) if self.val_transforms is None else self.val_transforms

            if not self.has_train_val_split:
                train_subjects = self.get_subjects(fold="train")
                train_dataset = self.dataset_cls(
                    train_subjects,
                    transform=train_transforms,
                )
                self.train_datasets.append(train_dataset)
                val_dataset = self.dataset_cls(
                    train_subjects,
                    transform=val_transforms,
                )
                self.val_datasets.append(val_dataset)

                self.train_queue = tio.Queue(
                    cast(tio.SubjectsDataset, self.train_datasets[0]),
                    max_length=self.queue_max_length,
                    samples_per_volume=self.samples_per_volume,
                    sampler=self.train_sampler,
                    num_workers=self.num_workers,
                    shuffle_subjects=self.shuffle_subjects,
                    shuffle_patches=self.shuffle_patches,
                    start_background=self.start_background,
                    verbose=self.verbose)

                self.val_queue = tio.Queue(
                    cast(tio.SubjectsDataset, self.val_datasets[0]),
                    max_length=self.queue_max_length,
                    samples_per_volume=self.samples_per_volume,
                    sampler=self.train_sampler,
                    num_workers=self.num_workers,
                    shuffle_subjects=self.shuffle_subjects,
                    shuffle_patches=self.shuffle_patches,
                    start_background=self.start_background,
                    verbose=self.verbose)

                self.validation = self.val_cls(
                    train_dataset=self.train_queue,
                    val_dataset=self.val_queue,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=self.pin_memory,
                    drop_last=self.drop_last,
                    num_folds=self.num_folds,
                    seed=self.seed,
                )
                self.validation.setup(self.val_split)
                self.has_validation = True
                self.size_train = self.validation.size_train
                self.size_val = self.validation.size_val
            else:
                train_subjects = self.get_subjects(fold="train")
                train_dataset = self.dataset_cls(train_subjects,
                                                 transform=train_transforms)
                self.train_datasets.append(train_dataset)
                self.train_queue = tio.Queue(
                    cast(tio.SubjectsDataset, self.train_datasets[0]),
                    max_length=self.queue_max_length,
                    samples_per_volume=self.samples_per_volume,
                    sampler=self.train_sampler,
                    num_workers=self.num_workers,
                    shuffle_subjects=self.shuffle_subjects,
                    shuffle_patches=self.shuffle_patches,
                    start_background=self.start_background,
                    verbose=self.verbose)
                self.size_train = len(self.train_queue)

                val_subjects = self.get_subjects(fold="val")
                val_dataset = self.dataset_cls(val_subjects,
                                               transform=val_transforms)
                self.val_datasets.append(val_dataset)
                self.val_queue = tio.Queue(
                    cast(tio.SubjectsDataset, self.val_datasets[0]),
                    max_length=self.queue_max_length,
                    samples_per_volume=self.samples_per_volume,
                    sampler=self.train_sampler,
                    num_workers=self.num_workers,
                    shuffle_subjects=self.shuffle_subjects,
                    shuffle_patches=self.shuffle_patches,
                    start_background=self.start_background,
                    verbose=self.verbose)
                self.size_val = len(self.val_queue)

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms(
                stage="test"
            ) if self.test_transforms is None else self.test_transforms
            test_subjects = self.get_subjects(fold="test")
            test_dataset = self.dataset_cls(test_subjects,
                                            transform=test_transforms)
            self.test_datasets.append(test_dataset)
            self.size_test = min([len(data) for data in self.test_datasets])

        if stage == "predict" or stage is None:
            predict_transforms = self.default_transforms(
                stage="predict"
            ) if self.test_transforms is None else self.test_transforms
            predict_subjects = self.get_subjects(fold="test")
            predict_dataset = self.dataset_cls(predict_subjects,
                                               transform=predict_transforms)
            self.predict_datasets.append(predict_dataset)
            self.size_predict = min(
                [len(data) for data in self.predict_datasets])

    def get_subjects(self, fold: str = "train") -> List[tio.Subject]:
        """
        Get train, test, or val list of TorchIO Subjects.

        Parameters
        ----------
        fold : str, optional
            Identify which type of dataset, ``'train'``, ``'test'``, or
            ``'val'``. Default = ``'train'``.

        Returns
        -------
        _ : List[tio.Subject]
            Train, test or val list of TorchIO Subjects.
        """
        train_subjs, test_subjs, val_subjs = self.get_subjects_dicts(
            intensities=self.intensities, labels=self.labels)
        if fold == "train":
            subjs_dict = train_subjs
        elif fold == "test":
            subjs_dict = test_subjs
        else:
            subjs_dict = val_subjs

        subjects = []
        for _, subject_dict in subjs_dict.items():
            subject = tio.Subject(subject_dict)
            subjects.append(subject)
        return subjects

    def get_subjects_dicts(
        self,
        intensities: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> Tuple[SubjDictType, SubjDictType, SubjDictType]:
        """
        Get paths to nii files for train/test/val images and labels.

        Returns
        -------
        _ : {(str, str): Dict}, {(str, str): Dict}, {(str, str): Dict}
            Paths to, respectively, train, test, and val images and labels.
        """

        def _get_dict(
            paths_dict: OrderedDict[Tuple[str, str], Path],
            intensities: List[str],
            labels: List[str],
            train: bool = True,
        ) -> SubjDictType:
            subjects_dict: SubjDictType = OrderedDict()
            for (subj_id, scan_id), path in paths_dict.items():
                subjects_dict[(subj_id, scan_id)] = {
                    "subj_id": subj_id,
                    "scan_id": scan_id
                }
                for intensity in intensities:
                    intensity_path = path / self.intensity2template[
                        intensity].substitute(subj_id=subj_id, scan_id=scan_id)
                    if intensity_path.is_file():
                        subjects_dict[(subj_id, scan_id)].update(
                            {intensity: tio.ScalarImage(intensity_path)})
                    else:
                        subjects_dict.pop((subj_id, scan_id), None)

                if train:
                    for label in labels:
                        label_path = path / self.label2template[
                            label].substitute(subj_id=subj_id, scan_id=scan_id)
                        if label_path.is_file():
                            subjects_dict[(subj_id, scan_id)].update(
                                {label: tio.LabelMap(label_path)})
                        else:
                            subjects_dict.pop((subj_id, scan_id), None)
            return subjects_dict

        intensities = intensities if intensities else ['T1', 'FLAIR']
        labels = labels if labels else []

        for intensity in intensities:
            assert intensity in self.intensity2template

        for label in labels:
            assert label in self.label2template

        subj_train_paths, subj_test_paths, subj_val_paths = self.get_paths(
            self.root,
            stem=self.step,
            has_train_test_split=self.has_train_test_split,
            has_train_val_split=self.has_train_val_split,
            shuffle=self.shuffle,
            seed=self.seed,
        )

        subj_train_dict = _get_dict(subj_train_paths,
                                    intensities,
                                    labels,
                                    train=True)
        subj_val_dict = _get_dict(subj_val_paths,
                                  intensities,
                                  labels,
                                  train=True)
        subj_test_dict = _get_dict(subj_test_paths,
                                   intensities,
                                   labels,
                                   train=False)

        return subj_train_dict, subj_test_dict, subj_val_dict

    @staticmethod
    def get_paths(
        data_root: PathType,
        stem: str = 'step01_structural_processing',
        has_train_test_split: bool = False,
        has_train_val_split: bool = False,
        test_split: Union[int, float] = 0.2,
        shuffle: bool = True,
        seed: int = 41,
    ) -> Tuple[SubjPathType, SubjPathType, SubjPathType]:
        """
        Get subject and scan IDs and the respective paths from the study data
        directory.

        Returns
        -------
        _ : {(str, str): Path}, {(str, str): Path}, {(str, str): Path}
            Paths for respectively, train, test and images and labels.
        """

        def _split_subj_train_paths(
                paths: OrderedDict) -> Tuple[OrderedDict, OrderedDict]:
            """Split dictionary into two proportially to `test_split`."""
            len_paths = len(paths)
            if isinstance(test_split, int):
                train_len = len_paths - test_split
                splits = [train_len, test_split]
            elif isinstance(test_split, float):
                test_len = int(np.floor(test_split * len_paths))
                train_len = len_paths - test_len
                splits = [train_len, test_len]
            else:
                raise ValueError(f"Unsupported type {type(test_split)}")
            indexes = list(range(len_paths))
            if shuffle:
                np.random.seed(seed)
                np.random.shuffle(indexes)
            train_idx, test_idx = indexes[:splits[0]], indexes[:splits[1]]

            paths_list = list(paths.items())

            subj_train_paths = OrderedDict([
                value for idx, value in enumerate(paths_list)
                if idx in train_idx
            ])
            subj_test_paths = OrderedDict([
                value for idx, value in enumerate(paths_list)
                if idx in test_idx
            ])
            return subj_train_paths, subj_test_paths

        data_root = Path(data_root)
        subj_pattern = r"[a-zA-z]{3}_[a-zA-Z]{2}_\d{4}"
        scan_pattern = r"[a-zA-z]{4}\d{3}"
        no_split_regex = re.compile("(" + subj_pattern + ")" + "/" +
                                    subj_pattern + "_" + "(" + scan_pattern +
                                    ")")
        has_split_regex = re.compile("(" + subj_pattern + ")" + "_" + "(" +
                                     scan_pattern + ")")

        def _get_subj_paths(data_root, regex):
            subj_paths = OrderedDict()
            for item in data_root.glob("*"):
                if not item.is_dir() and not item.name.startswith('.'):
                    match = regex.search(str(item))
                    if match is not None:
                        subj_id, scan_id = match.groups()
                        subj_paths[(subj_id, scan_id)] = data_root
            return subj_paths

        if not has_train_test_split:
            paths = OrderedDict()
            for item in data_root.glob("*/*"):
                if item.is_dir() and not item.name.startswith('.'):
                    match = no_split_regex.search(str(item))
                    if match is not None:
                        subj_id, scan_id = match.groups()
                        paths[(subj_id, scan_id)] = data_root / subj_id / (
                            subj_id + "_" + scan_id) / stem
            subj_train_paths, subj_test_paths = _split_subj_train_paths(paths)
        else:
            train_root = data_root / stem / "train"
            subj_train_paths = _get_subj_paths(train_root, has_split_regex)
            test_root = data_root / stem / "test"
            subj_test_paths = _get_subj_paths(test_root, has_split_regex)

        val_root = data_root / stem / "val"
        subj_val_paths = _get_subj_paths(
            val_root,
            has_split_regex) if has_train_val_split else OrderedDict()

        return subj_train_paths, subj_test_paths, subj_val_paths

    def get_preprocessing_transforms(
        self,
        shape: Optional[Tuple[int, int, int]] = None,
        resample: bool = False,
    ) -> tio.Transform:
        """
        Get preprocessing transorms to apply to all subjects.

        Returns
        -------
        preprocess : tio.Transform
            All preprocessing steps that should be applied to all subjects.
        """
        preprocess_list: List[tio.transforms.Transform] = []

        # Use standard orientation for all images
        preprocess_list.append(tio.ToCanonical())

        # If true, resample to T1
        if resample:
            preprocess_list.append(tio.Resample('T1'))

        if shape is None:
            train_subjects = self.get_subjects(fold="train")
            test_subjects = self.get_subjects(fold="test")
            shape = self.get_max_shape(train_subjects + test_subjects)
        else:
            shape = self.dims

        preprocess_list.extend([
            tio.RescaleIntensity((-1, 1)),
            tio.CropOrPad(shape),
            tio.EnsureShapeMultiple(8),  # for the U-Net
            tio.OneHot()
        ])

        return tio.Compose(preprocess_list)

    @staticmethod
    def get_augmentation_transforms() -> tio.Transform:
        """"
        Get augmentation transorms to apply to subjects during training.

        Returns
        -------
        augment : tio.Transform
            All augmentation steps that should be applied to subjects during
            training.
        """
        augment = tio.Compose([
            tio.RandomAffine(),
            tio.RandomGamma(p=0.5),
            tio.RandomNoise(p=0.5),
            tio.RandomMotion(p=0.1),
            tio.RandomBiasField(p=0.25),
        ])
        return augment

    def default_transforms(
        self,
        stage: Optional[str] = None,
    ) -> tio.Transform:
        """
        Default transforms and augmentations for the dataset.

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
        """
        transforms: List[tio.transforms.Transform] = []
        if self.use_preprocessing:
            preprocess = self.get_preprocessing_transforms(
                shape=self.dims,
                resample=self.resample,
            )
            transforms.append(preprocess)
        if (stage == "fit" or stage is None) and self.use_augmentation:
            augment = self.get_augmentation_transforms()
            transforms.append(augment)

        return tio.Compose(transforms)

    def train_dataloader(self, *args: Any,
                         **kwargs: Any) -> TrainDataLoaderType:
        """
        Generates one or multiple Pytorch DataLoaders for train.

        Returns
        -------
        _ : Collection of DataLoader
            Collection of train dataloaders specifying training samples.
        """
        if self.has_validation:
            return self.validation.train_dataloader()
        return [
            DataLoader(
                dataset=self.train_queue,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
            )
        ]

    def val_dataloader(self, *args: Any, **kwargs: Any) -> EvalDataLoaderType:
        """
        Generates one or multiple Pytorch DataLoaders for validation.

        Returns
        -------
        _ : Collection of DataLoader
            Collection of validation dataloaders specifying validation samples.
        """
        if self.has_validation:
            return self.validation.val_dataloader()
        return [
            DataLoader(
                dataset=self.val_queue,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
            )
        ]

    def test_dataloader(self, *args: Any, **kwargs: Any) -> EvalDataLoaderType:
        """
        Generates one or multiple Pytorch DataLoaders for test.

        Returns
        -------
        _ : Collection of DataLoaders
            Collection of test dataloaders specifying test samples.
        """
        dataloaders = []
        num_dataloaders = len(self.test_datasets)
        for idx in range(num_dataloaders):
            dataloaders.append(
                DataLoader(
                    dataset=self.test_datasets[idx],
                    batch_size=1,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=self.drop_last,
                ))
        return dataloaders

    def predict_dataloader(self, *args: Any,
                           **kwargs: Any) -> EvalDataLoaderType:
        """
        Generates one or multiple Pytorch DataLoaders for prediction.

        Returns
        -------
        _ : Collection of DataLoaders
            Collection of prediction dataloaders specifying prediction samples.
        """
        dataloaders = []
        num_dataloaders = len(self.test_datasets)
        for idx in range(num_dataloaders):
            dataloaders.append(
                DataLoader(
                    dataset=self.test_datasets[idx],
                    batch_size=1,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=self.drop_last,
                ))
        return dataloaders

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Called at the end of fit (train + validate), validate, test,
        or predict. Remove root directory if a temporary was used.

        Parameters
        ----------
        stage: Optional[str]
            Either ``'fit``, ``'validate'``, ``'test'``, or ``'predict'``.
            Default = None.
        """
        if self.is_temp_dir:
            shutil.rmtree(self.root)
