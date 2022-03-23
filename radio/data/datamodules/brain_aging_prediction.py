#!/usr/bin/env python
# coding=utf-8
"""
Brain Aging Prediction Data Module
"""

from typing import Any, Dict, List, Optional, Tuple, Union, cast
from pathlib import Path
import re
from string import Template
from collections import OrderedDict
import numpy as np
import torchio as tio  # type: ignore
from torch.utils.data import DataLoader
from radio.settings.pathutils import is_dir_or_symlink, PathType, ensure_exists
from ..visiondatamodule import VisionDataModule
from ..datautils import get_subjects_from_batch
from ..datatypes import SubjPathType, SubjDictType

__all__ = ["BrainAgingPredictionDataModule"]


class BrainAgingPredictionDataModule(VisionDataModule):
    """
    Brain Aging Prediction Data Module.

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
    """
    name: str = "brain_aging_prediction"
    dataset_cls = tio.SubjectsDataset
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
        train_transforms: Optional[tio.Transform] = None,
        val_transforms: Optional[tio.Transform] = None,
        test_transforms: Optional[tio.Transform] = None,
        use_augmentation: bool = True,
        use_preprocessing: bool = True,
        resample: bool = False,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
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
            shuffle=shuffle,
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
                val_dataset = self.dataset_cls(
                    train_subjects,
                    transform=val_transforms,
                )
                self.validation = self.val_cls(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=self.drop_last,
                    num_folds=self.num_folds,
                    seed=self.seed,
                )
                self.validation.setup(self.val_split)
                self.has_validation = True
                self.train_dataset = train_dataset
                self.size_train = self.size_train_dataset(
                    self.validation.train_samplers)
                self.val_dataset = val_dataset
                self.size_val = self.size_eval_dataset(
                    self.validation.val_samplers)
            else:
                train_subjects = self.get_subjects(fold="train")
                self.train_dataset = self.dataset_cls(
                    train_subjects, transform=train_transforms)
                self.size_train = self.size_train_dataset(self.train_dataset)

                val_subjects = self.get_subjects(fold="val")
                self.val_dataset = self.dataset_cls(val_subjects,
                                                    transform=val_transforms)
                self.size_val = self.size_eval_dataset(self.val_dataset)

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms(
                stage="test"
            ) if self.test_transforms is None else self.test_transforms
            test_subjects = self.get_subjects(fold="test")
            self.test_dataset = self.dataset_cls(test_subjects,
                                                 transform=test_transforms)
            self.size_test = self.size_eval_dataset(self.test_dataset)

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

    def save(self,
             dataloader: DataLoader,
             root: PathType = "~/LocalCerebro/Studies",
             fold: str = "train") -> None:
        """
        Arguments
        ---------
        root : Path or str, optional
            Root where to save data. Default = ``'~/LocalCerebro/Studies'``.
        """
        save_root = ensure_exists(
            Path(root) / self.study / 'Public' / 'preprocessed_data' /
            self.step / fold)

        for batch in dataloader:
            subjects = get_subjects_from_batch(cast(Dict[str, Any], batch))
            for subject in subjects:
                subj_id = subject["subj_id"]
                scan_id = subject["scan_id"]
                for image_name in subject.get_images_names():
                    filename = self.intensity2template[image_name].substitute(
                        subj_id=subj_id, scan_id=scan_id)
                    subject[image_name].save(save_root / filename)
