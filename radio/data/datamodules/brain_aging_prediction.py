#!/usr/bin/env python
# coding=utf-8
"""
Brain Aging Prediction Data Module
"""

from typing import Any, Callable, List, Optional, Tuple, Union, Dict
from pathlib import Path
import re
from string import Template
from collections import OrderedDict
import random
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torchio as tio  # type: ignore
from radio.settings.pathutils import is_dir_or_symlink, PathType
from ..visiondatamodule import VisionDataModule

__all__ = ["BrainAgingPredictionDataModule", "plot_batch"]


def plot_batch(batch: dict,
               num_imgs: int = 5,
               slice_num: int = None,
               modalities: List[str] = [],
               labels: List[str] = [],
               exclude_keys: List[str] = [],
               train: bool = True) -> None:
    """plot images and labels from a batch of train images"""

    images = {}
    for key in batch.keys():
        if key not in exclude_keys:
            data = batch[key]["data"]
            batch_size = data.shape[0]
            images[key] = data

    num_imgs = min(num_imgs, batch_size)
    samples = random.sample(range(0, batch_size), num_imgs)

    num_images = len(images.keys())
    _, axs = plt.subplots(nrows=num_imgs, ncols=num_images, squeeze=False)
    for row_idx, img_idx in enumerate(samples):
        for idx, (key, data) in enumerate(images.items()):
            # Plot images
            axis = axs[row_idx, idx]
            axis.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            img = data[img_idx].permute(0, 3, 1, 2).numpy().squeeze(0)
            if slice_num is None:
                _slice = img.shape[0] // 2
            if key in modalities:
                if len(img.shape) == 3:
                    axis.imshow(img[_slice, :, :], cmap="gray")
                else:
                    axis.imshow(img, cmap="gray")
            # Plot label
            if key in labels and train:
                # axis.imshow(img[0, _slice, :, :], cmap="binary")
                if len(img.shape) == 3:
                    axis.imshow(img[_slice, :, :], cmap="binary")
                else:
                    axis.imshow(img, cmap="gray")

    for column, modality in enumerate(images.keys()):
        plt.sca(axs[0, column])
        plt.title(label=modality, size=15)
    plt.show()


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
        Default = ``"/media/cerebro/Studies"``.
    study : str, optional
        Study name. Default = ``"Brain_Aging_Prediction"``.
    data_dir : str, optional
        Subdirectory where the data is located.
        Default = ``"Public/data"``.
    step : str, optional
        Which processing step to use.
        Default = ``''step01_structural_processing''``.
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
        If ``True``, resample all images to ``T1w'``. Default = ``False``.
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
    modalities : List[str], optional
        Which modalilities to load. Default = ``['T1w']``.
    labels : List[str], optional
        Which labels to load. Default = ``[]``.
    dims : List[int], optional
        Max spatial dimensions across subjects' images.
        Default = ``[256, 256, 256]``.
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
        root: PathType = Path("/media/cerebro/Studies"),
        study: str = "Brain_Aging_Prediction",
        data_dir: str = "Public/data",
        step: str = "step01_structural_processing",
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
        modalities: List[str] = ["T1w"],
        labels: List[str] = [],
        dims: List[int] = [256, 256, 256],
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
        self.modalities = modalities
        self.labels = labels
        self.dims = dims
        self.use_augmentation = use_augmentation
        self.resample = resample

    def get_max_shape(self, subjects: List[tio.Subject]) -> List[int]:
        """
        Get max shape.

        Parameters
        ----------
        subjects : List[tio.Subject]
            List of TorchIO Subject objects.

        Returns
        -------
        _ : np.ndarray((1, 3), np.int_)
            Max height, width and depth across all subjects.
        """
        dataset = self.dataset_cls(subjects)
        shapes = np.array([
            image.spatial_shape for subject in dataset
            for image in subject.get_images()
        ])
        return shapes.max(axis=0).tolist()

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """Verify data directory exists."""
        if not is_dir_or_symlink(self.root):
            raise OSError("Study data directory not found!")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates train, validation and test collection of samplers.

        Parameters
        ----------
        stage: Optional[str]
            Either ``'fit``, ``'validate'``, ``'test'``, or ``'predict'``.
            If stage = None, set-up all stages. Default = None.
        """
        # dims = []
        if stage == "fit" or stage is None:
            train_transforms = self.default_transforms(
                stage="fit"
            ) if self.train_transforms is None else self.train_transforms

            val_transforms = self.default_transforms(
                stage="fit"
            ) if self.val_transforms is None else self.val_transforms

            train_subjects = self.get_subjects()
            # dims.append([self.get_max_shape(train_subjects)])
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
            # dims.append([self.get_max_shape(test_subjects)])
            test_dataset = self.dataset_cls(test_subjects,
                                            transform=test_transforms)
            self.test_datasets.append(test_dataset)
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

        # self.dims = np.concatenate(dims, axis=0).max(axis=0).tolist()

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
        modalities: List[str] = ['T1w'],
        labels: List[str] = [],
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

    def get_preprocessing_transforms(
        self,
        size: Optional[List[int]] = [256, 256, 256],
        resample: bool = False,
    ) -> Callable:
        """
        Get preprocessing transorms to apply to all subjects.

        Returns
        -------
        preprocess : tio.Compose
            All preprocessing steps that should be applied to all subjects.
        """
        preprocess_list = []

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
    def get_augmentation_transforms() -> Callable:
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
        preprocess = self.get_preprocessing_transforms(
            size=self.dims,
            resample=self.resample,
        )
        transforms.append(preprocess)
        if stage == "fit" or stage is None and self.use_augmentation:
            augment = self.get_augmentation_transforms()
            transforms.append(augment)

        return tio.Compose(transforms)
