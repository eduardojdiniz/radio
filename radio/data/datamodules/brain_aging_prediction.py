#!/usr/bin/env python
# coding=utf-8
"""
Brain Aging Prediction Data Module
"""

from typing import Any, Callable, List, Optional, Tuple, Union, Dict, cast
from pathlib import Path
import re
from string import Template
from collections import OrderedDict
import numpy as np
import torchio as tio  # type: ignore
from radio.settings.pathutils import is_dir_or_symlink, PathType
from ..visiondatamodule import VisionDataModule

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
    resample : bool, optional
        If ``True``, resample all images to ``'T1w'``. Default = ``False``.
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
        Which intensities to load. Default = ``['T1w']``.
    labels : List[str], optional
        Which labels to load. Default = ``[]``.
    dims : Tuple[int, int, int], optional
        Max spatial dimensions across subjects' images.
        Default = ``(256, 256, 256)``.
    seed : int, optional
        When `shuffle` is True, `seed` affects the ordering of the indices,
        which controls the randomness of each fold. It is also use to seed the
        RNG used by RandomSampler to generate random indexes and
        multiprocessing to generate `base_seed` for workers. Pass an int for
        reproducible output across multiple function calls. Default = ``41``.
    """
    name: str = "brain_aging_prediction"
    dataset_cls = tio.SubjectsDataset

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
        dims: Tuple[int, int, int] = (256, 256, 256),
        seed: int = 41,
        **kwargs: Any,
    ) -> None:
        root = Path(root) / study / data_dir
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
        self.step = step
        self.intensities = intensities if intensities else ['T1w']
        self.labels = labels if labels else []
        self.dims = dims
        self.use_augmentation = use_augmentation
        self.resample = resample

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
        """Verify data directory exists."""
        if not is_dir_or_symlink(self.root):
            raise OSError('Study data directory not found!')

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

            train_subjects = self.get_subjects()
            self.train_dataset = self.dataset_cls(train_subjects,
                                                  transform=train_transforms)
            self.val_dataset = self.dataset_cls(train_subjects,
                                                transform=val_transforms)

            self.validation = self.val_cls(train_dataset=self.train_dataset,
                                           val_dataset=self.val_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=self.shuffle,
                                           num_workers=self.num_workers,
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
            test_subjects = self.get_subjects(train=False)
            test_dataset = self.dataset_cls(test_subjects,
                                            transform=test_transforms)
            self.test_datasets.append(test_dataset)
            self.size_test = min([len(data) for data in self.test_datasets])

        if stage == "predict" or stage is None:
            predict_transforms = self.default_transforms(
                stage="predict"
            ) if self.test_transforms is None else self.test_transforms
            predict_subjects = self.get_subjects(train=False)
            predict_dataset = self.dataset_cls(predict_subjects,
                                               transform=predict_transforms)
            self.predict_datasets.append(predict_dataset)
            self.size_predict = min(
                [len(data) for data in self.predict_datasets])

    @staticmethod
    def get_paths(root: Path) -> OrderedDict[Tuple[str, str], Path]:
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
        for item in root.glob("*/*"):
            if item.is_dir() and not item.name.startswith('.'):
                match = regex.search(str(item))
                if match is not None:
                    subj_id, scan_id = match.groups()
                    paths[(subj_id, scan_id)] = root / subj_id / scan_id

        return paths

    @staticmethod
    def split_train_dict(dictionary: OrderedDict,
                         test_split: Union[int, float] = 0.2,
                         shuffle: bool = True,
                         seed: int = 41) -> Tuple[OrderedDict, OrderedDict]:
        """Split dictionary into two proportially to `test_split`."""
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
        intensities: Optional[List[str]] = None,
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
        intensity2template = {
            "T1w": Template('wstrip_m${scan_id}_T1.nii'),
            "FLAIR": Template('wstrip_mr${scan_id}_FLAIR.nii'),
        }

        label2template: Dict[str, Template] = {}

        def _get_dict(
            paths_dict: OrderedDict[Tuple[str, str], Path],
            intensities: List[str],
            labels: List[str],
            train: bool = True,
        ) -> OrderedDict[Tuple[str, str], dict]:
            subjects_dict: OrderedDict[Tuple[str, str], dict] = OrderedDict()
            for (subj_id, scan_id), path in paths_dict.items():
                subjects_dict[(subj_id, scan_id)] = {
                    "subj_id": subj_id,
                    "scan_id": scan_id
                }
                for intensity in intensities:
                    intensity_path = path / step / intensity2template[
                        intensity].substitute(scan_id=scan_id)
                    if intensity_path.is_file():
                        subjects_dict[(subj_id, scan_id)].update(
                            {intensity: tio.ScalarImage(intensity_path)})
                    else:
                        subjects_dict.pop((subj_id, scan_id), None)

                if train:
                    for label in labels:
                        label_path = path / step / label2template[
                            label].substitute(scan_id=scan_id)
                        if label_path.is_file():
                            subjects_dict[(subj_id, scan_id)].update(
                                {label: tio.LabelMap(label_path)})
                        else:
                            subjects_dict.pop((subj_id, scan_id), None)
            return subjects_dict

        intensities = intensities if intensities else ['T1w']
        labels = labels if labels else []

        for intensity in intensities:
            assert intensity in intensity2template

        for label in labels:
            assert label in label2template

        paths_dict = self.get_paths(self.root)
        train_paths_dict, test_paths_dict = self.split_train_dict(
            paths_dict, shuffle=self.shuffle, seed=self.seed)

        training_dict = _get_dict(train_paths_dict,
                                  intensities,
                                  labels,
                                  train=True)
        testing_dict = _get_dict(test_paths_dict,
                                 intensities,
                                 labels,
                                 train=False)

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
                step=self.step,
                intensities=self.intensities,
                labels=self.labels)
            train_subjects = []
            for _, subject_dict in training_dict.items():
                # 'image' and 'label' are arbitrary names for the images
                subject = tio.Subject(subject_dict)
                train_subjects.append(subject)
            return train_subjects

        _, testing_dict = self.get_subject_dicts(step=self.step,
                                                 intensities=self.intensities,
                                                 labels=self.labels)
        test_subjects = []
        for _, subject_dict in testing_dict.items():
            subject = tio.Subject(subject_dict)
            test_subjects.append(subject)

        return test_subjects

    def get_preprocessing_transforms(
        self,
        size: Optional[Tuple[int, int, int]] = (256, 256, 256),
        resample: bool = False,
    ) -> tio.transforms.Compose:
        """
        Get preprocessing transorms to apply to all subjects.

        Returns
        -------
        preprocess : tio.transforms.Compose
            All preprocessing steps that should be applied to all subjects.
        """
        preprocess_list: List[tio.transforms.Transform] = []

        # Use standard orientation for all images
        preprocess_list.append(tio.ToCanonical())

        # If true, resample to T1w
        if resample:
            preprocess_list.append(tio.Resample('T1w'))

        if size is None:
            train_subjects = self.get_subjects()
            test_subjects = self.get_subjects(train=False)
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
    def get_augmentation_transforms() -> tio.transforms.Compose:
        """"
        Get augmentation transorms to apply to subjects during training.

        Returns
        -------
        augment : tio.transforms.Compose
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
    ) -> tio.transforms.Compose:
        """
        Default transforms and augmentations for the dataset.

        Parameters
        ----------
        stage: Optional[str]
            Either ``'fit``, ``'validate'``, ``'test'``, or ``'predict'``.
            If stage = None, set-up all stages. Default = None.

        Returns
        -------
        _: tio.transforms.Compose
            All preprocessing steps (and if ``'fit'``, augmentation steps too)
            that should be applied to the subjects.
        """
        transforms: List[tio.transforms.Transform] = []
        preprocess = self.get_preprocessing_transforms(
            size=self.dims,
            resample=self.resample,
        )
        transforms.append(preprocess)
        if stage == "fit" or stage is None and self.use_augmentation:
            augment = self.get_augmentation_transforms()
            transforms.append(augment)

        return tio.Compose(transforms)
