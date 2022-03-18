#!/usr/bin/env python
# coding=utf-8
"""
Based on BaseDataModule for managing data. A vision datamodule that is
shareable, reusable class that encapsulates all the steps needed to process
data, i.e., decoupling datasets from models to allow building dataset-agnostic
models. They also allow you to share a full dataset without explaining how to
download, split, transform, and process the data.
"""

from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from pathlib import Path
import re
import shutil
from collections import OrderedDict
from string import Template
from torch.utils.data import DataLoader
import numpy as np
import torchio as tio  # type: ignore
from radio.settings.pathutils import PathType
from .validation import TrainDataLoaderType, EvalDataLoaderType
from .datatypes import SpatialShapeType
from .basedatamodule import BaseDataModule

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
    datamodule = VisionPatchDataModule()
    datamodule.prepare_data() # download
    datamodule.setup(stage) # process and split
    datamodule.teardown(stage) # clean-up

    Parameters
    ----------
    root : Path or str, optional
        Root to GPN's CEREBRO Studies folder.
        Default = ``"/media/cerebro/Studies"``.
    study : str, optional
        Study name. Default = ``"Brain_Aging_Prediction"``.
    data_dir : str, optional
        Subdirectory where the data is located.
        Default = ``"Public/data"``.
    step : str, optional
        Which processing step to use.
        Default = ``''step01_structural_processing''``.
    modalities : List[str], optional
        Which modalilities to load. Default = ``['T1w']``.
    labels : List[str], optional
        Which labels to load. Default = ``[]``.
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
    dims: List[int] = [256, 256, 256]
    #: Dataset name
    name: str = "brain_aging_prediction"

    def __init__(
        self,
        *args: Any,
        root: PathType = Path("/media/cerebro/Studies"),
        study: str = "Brain_Aging_Prediction",
        data_dir: str = "Public/data",
        step: str = "step01_structural_processing",
        modalities: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        train_subjects: Optional[List[tio.Subject]] = None,
        test_subjects: Optional[List[tio.Subject]] = None,
        patch_size: SpatialShapeType = 96,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        use_augmentation: bool = False,
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
        seed: int = 41,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        root = Path(root) / study / data_dir
        super().__init__(
            *args,
            root=root,
            train_subjects=train_subjects,
            test_subjects=test_subjects,
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
        self.step = step
        self.modalities = modalities if modalities is not None else ['T1w']
        self.labels = labels if labels is not None else []
        self.use_augmentation = use_augmentation
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

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """Saves files to data root dir."""
        # self.dataset_cls(self.root, train=True, download=True)
        # self.dataset_cls(self.root, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates train, validation and test collection of samplers.

        Parameters
        ----------
        stage: Optional[str]
            Either ``'fit``, ``'validate'``, ``'test'``, or ``'predict'``.
            If stage = None, set-up all stages. Default = None.
        """
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms(
                stage="fit"
            ) if self.train_transforms is None else self.train_transforms

            val_transforms = self.default_transforms(
                stage="fit"
            ) if self.val_transforms is None else self.val_transforms

            if self.train_subjects is not None:
                self.train_dataset = self.dataset_cls(
                    self.train_subjects, transform=train_transforms)
                self.val_dataset = self.dataset_cls(self.train_subjects,
                                                    transform=val_transforms)
            else:
                train_subjects = self.get_subjects()
                self.train_dataset = self.dataset_cls(
                    train_subjects, transform=train_transforms)
                self.val_dataset = self.dataset_cls(train_subjects,
                                                    transform=val_transforms)

            self.train_queue = tio.Queue(
                self.train_dataset,
                max_length=self.queue_max_length,
                samples_per_volume=self.samples_per_volume,
                sampler=self.train_sampler,
                num_workers=self.num_workers,
                shuffle_subjects=self.shuffle_subjects,
                shuffle_patches=self.shuffle_patches,
                start_background=self.start_background,
                verbose=self.verbose)

            self.val_queue = tio.Queue(
                self.val_dataset,
                max_length=self.queue_max_length,
                samples_per_volume=self.samples_per_volume,
                sampler=self.train_sampler,
                num_workers=self.num_workers,
                shuffle_subjects=self.shuffle_subjects,
                shuffle_patches=self.shuffle_patches,
                start_background=self.start_background,
                verbose=self.verbose)

            self.validation = self.val_cls(train_dataset=self.train_queue,
                                           val_dataset=self.val_queue,
                                           batch_size=self.batch_size,
                                           shuffle=False,
                                           num_workers=0,
                                           pin_memory=self.pin_memory,
                                           drop_last=self.drop_last,
                                           num_folds=self.num_folds,
                                           seed=self.seed)

            self.validation.setup(self.val_split)
            self.size_train = self.validation.size_train
            self.size_val = self.validation.size_val

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms(
                stage="test"
            ) if self.test_transforms is None else self.test_transforms
            if self.test_subjects is not None:
                test_dataset = self.dataset_cls(self.test_subjects,
                                                transform=test_transforms)
            else:
                test_subjects = self.get_subjects(train=False)
                # dims.append([self.get_max_shape(test_subjects)])
                test_dataset = self.dataset_cls(test_subjects,
                                                transform=test_transforms)
            self.test_datasets.append(test_dataset)
            self.size_test = min([len(data) for data in self.test_datasets])

        if stage == "predict" or stage is None:
            predict_transforms = self.default_transforms(
                stage="predict"
            ) if self.test_transforms is None else self.test_transforms
            if self.test_subjects is not None:
                predict_dataset = self.dataset_cls(
                    self.test_subjects, transform=predict_transforms)
            else:
                predict_subjects = self.get_subjects(train=False)
                # dims.append([self.get_max_shape(test_subjects)])
                predict_dataset = self.dataset_cls(
                    predict_subjects, transform=predict_transforms)
            self.predict_datasets.append(predict_dataset)
            self.size_predict = min(
                [len(data) for data in self.predict_datasets])

    def get_paths(self) -> OrderedDict[Tuple[str, str], Path]:
        """
        Get subject and scan IDs and the respective paths from the study data
        directory.

        Returns
        -------
        _ : Tuple[List[Path], List[Path]]
            Paths to train images and labels.
        """
        paths = OrderedDict()
        # 6-digit subject ID, followed by 6-digit scan ID
        subj_regex = r"[a-zA-z]{3}_[a-zA-Z]{2}_\d{4}"
        scan_regex = r"[a-zA-z]{4}\d{3}"
        regex = re.compile("(" + subj_regex + ")" + "/" + "(" + subj_regex +
                           "_" + scan_regex + ")")
        for item in self.root.glob("*/*"):
            if item.is_dir() and not item.name.startswith('.'):
                match = regex.search(str(item))
                if match is not None:
                    subj_id, scan_id = match.groups()
                    paths[(subj_id, scan_id)] = self.root / subj_id / scan_id

        return paths

    @staticmethod
    def split_dict(dictionary: OrderedDict,
                   test_split: Union[int, float] = 0.2,
                   shuffle: bool = True,
                   seed: int = 41) -> Tuple[OrderedDict, OrderedDict]:
        """Split dict into two."""
        len_dict = len(dictionary)
        if isinstance(test_split, int):
            train_len = len_dict - test_split
            splits = [train_len, test_split]
        elif isinstance(test_split, float):
            test_len = int(np.floor(test_split * len_dict))
            train_len = len_dict - test_len
            splits = [train_len, test_len]
        else:
            raise ValueError(f"Unsupported type {type(test_split)}")
        indexes = list(range(len_dict))
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indexes)
        train_idx, test_idx = indexes[:splits[0]], indexes[:splits[1]]

        dictionary_list = list(dictionary.items())

        train_dict = OrderedDict([
            value for idx, value in enumerate(dictionary_list)
            if idx in train_idx
        ])
        test_dict = OrderedDict([
            value for idx, value in enumerate(dictionary_list)
            if idx in test_idx
        ])
        return train_dict, test_dict

    def get_subject_dicts(
        self,
        step: str = 'step01_structural_processing',
        modalities: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> Tuple[OrderedDict[Tuple[str, str], dict], OrderedDict[Tuple[str, str],
                                                               dict]]:
        """
        Get paths to nii files for train images and labels.

        Returns
        -------
        _ : Tuple[List[Path], List[Path]]
            Paths to train images and labels.
        """
        mod2template = {
            "T1w": Template('wstrip_m${scan_id}_T1.nii'),
            "FLAIR": Template('wstrip_mr${scan_id}_FLAIR.nii'),
        }

        label2template: Dict[str, Template] = {}

        if labels is None:
            labels = []
        if modalities is None:
            modalities = ['T1w']

        for mod in modalities:
            assert mod in mod2template

        for label in labels:
            assert label in label2template

        paths_dict = self.get_paths()
        train_paths_dict, test_paths_dict = self.split_dict(
            paths_dict, shuffle=self.shuffle, seed=self.seed)
        training_dict = OrderedDict()
        testing_dict = OrderedDict()

        # Get training dict
        for (subj_id, scan_id), path in train_paths_dict.items():
            training_dict[(subj_id, scan_id)] = {
                "subj_id": subj_id,
                "scan_id": scan_id
            }
            for mod in modalities:
                mod_path = path / step / mod2template[mod].substitute(
                    scan_id=scan_id)
                if mod_path.is_file():
                    training_dict[(subj_id, scan_id)].update(
                        {mod: tio.ScalarImage(mod_path)})
                else:
                    training_dict.pop((subj_id, scan_id), None)

            for label in labels:
                label_path = path / step / label2template[label].substitute(
                    scan_id=scan_id)
                if label_path.is_file():
                    training_dict[(subj_id, scan_id)].update(
                        {label: tio.LabelMap(label_path)})
                else:
                    training_dict.pop((subj_id, scan_id), None)

        # Get testing dict
        for (subj_id, scan_id), path in test_paths_dict.items():
            testing_dict[(subj_id, scan_id)] = {
                "subj_id": subj_id,
                "scan_id": scan_id
            }
            for mod in modalities:
                mod_path = path / step / mod2template[mod].substitute(
                    scan_id=scan_id)
                if mod_path.is_file():
                    testing_dict[(subj_id, scan_id)].update(
                        {mod: tio.ScalarImage(mod_path)})
                else:
                    testing_dict.pop((subj_id, scan_id), None)

        return training_dict, testing_dict

    def get_subjects(self, train: bool = True) -> List[tio.Subject]:
        """
        Get TorchIO Subject train and test subjects.

        Parameters
        ----------
        train : bool, optional
            If True, return a loader for the train dataset, else for the
            validation dataset. Default = ``True``.

        Returns
        -------
        _ : List[tio.Subject]
            TorchIO Subject train or test subjects.
        """
        if train:
            training_dict, _ = self.get_subject_dicts(
                step=self.step, modalities=self.modalities, labels=self.labels)
            train_subjects = []
            for _, subject_dict in training_dict.items():
                # 'image' and 'label' are arbitrary names for the images
                subject = tio.Subject(subject_dict)
                train_subjects.append(subject)
            return train_subjects

        _, testing_dict = self.get_subject_dicts(step=self.step,
                                                 modalities=self.modalities,
                                                 labels=self.labels)
        test_subjects = []
        for _, subject_dict in testing_dict.items():
            subject = tio.Subject(subject_dict)
            test_subjects.append(subject)

        return test_subjects

    @staticmethod
    def get_preprocessing_transforms(
            size: Union[int, Tuple[int, int, int]] = 256) -> tio.Transform:
        """
        Get preprocessing transorms to apply to all subjects.

        Returns
        -------
        preprocess : tio.Compose
            All preprocessing steps that should be applied to all subjects.
        """
        if isinstance(size, int):
            size = 3 * (size)

        preprocess_list = []

        # Use standard orientation for all images
        # preprocess_list.append(tio.ToCanonical())

        preprocess_list.extend([tio.RescaleIntensity((-1, 1))])

        return tio.Compose(preprocess_list)

    @staticmethod
    def get_augmentation_transforms() -> tio.Transform:
        """"
        Get augmentation transorms to apply to subjects during training.

        Returns
        -------
        augment : tio.Compose
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

    def default_transforms(self, stage: Optional[str] = None) -> Callable:
        """
        Default transforms and augmentations for the dataset.

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
        """
        transforms = []
        preprocess = self.get_preprocessing_transforms()
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
        return self.validation.train_dataloader()

    def val_dataloader(self, *args: Any, **kwargs: Any) -> EvalDataLoaderType:
        """
        Generates one or multiple Pytorch DataLoaders for validation.

        Returns
        -------
        _ : Collection of DataLoader
            Collection of validation dataloaders specifying validation samples.
        """
        return self.validation.val_dataloader()

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
