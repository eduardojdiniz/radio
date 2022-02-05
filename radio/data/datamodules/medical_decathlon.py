#!/usr/bin/env python
# coding=utf-8
"""
Medical Decathlon Data Module
"""

from typing import Optional, Any, Callable, Union, Tuple, List
from pathlib import Path
import random
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import numpy.typing as npt
import torchio as tio  # type: ignore
from monai.apps import download_and_extract
from radio.settings.pathutils import (DATA_ROOT, is_dir_or_symlink,
                                      ensure_exists, PathType)
from ..visiondatamodule import VisionDataModule

__all__ = ["MedicalDecathlonDataModule", "plot_train_batch", "plot_test_batch"]


def plot_train_batch(batch: dict,
                     num_imgs: int = 5,
                     slice_num: int = 24) -> None:
    """plot images and labels from a batch of train images"""
    images, labels = (batch["image"]["data"], batch["label"]["data"])
    batch_size = images.shape[0]
    assert num_imgs <= batch_size
    samples = random.sample(range(0, batch_size), num_imgs)

    print(f"image shape: {images.shape[2:]}, label shape: {labels.shape[2:]}")
    _, axs = plt.subplots(nrows=num_imgs, ncols=2, squeeze=False)
    for row_idx, img_idx in enumerate(samples):
        # Plot image
        axis = axs[row_idx, 0]
        axis.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        img = images[img_idx].permute(0, 3, 1, 2).numpy().squeeze()
        axis.imshow(img[slice_num, :, :], cmap="gray")
        # Plot label
        axis = axs[row_idx, 1]
        axis.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        label = labels[img_idx].permute(0, 3, 1, 2).numpy().squeeze()
        axis.imshow(label[0, slice_num, :, :])

    plt.sca(axs[0, 0])
    plt.title(label="Images", size=15)
    plt.sca(axs[0, 1])
    plt.title(label="Labels", size=15)
    plt.show()


def plot_test_batch(batch: dict,
                    num_imgs: int = 5,
                    slice_num: int = 24) -> None:
    """plot images from a batch of test images"""
    images = batch["image"]["data"]
    batch_size = images.shape[0]
    assert num_imgs <= batch_size
    samples = random.sample(range(0, batch_size), num_imgs)

    print(f"image shape: {images.shape[2:]}")
    _, axs = plt.subplots(nrows=num_imgs, ncols=1, squeeze=False)
    for row_idx, img_idx in enumerate(samples):
        # Plot image
        axis = axs[row_idx, 0]
        axis.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        img = images[img_idx].permute(0, 3, 1, 2).numpy().squeeze()
        axis.imshow(img[slice_num, :, :], cmap="gray")

    plt.sca(axs[0, 0])
    plt.title(label="Images", size=15)
    plt.show()


class MedicalDecathlonDataModule(VisionDataModule):
    """
    Medical Decathlon Data Module.

    Typical Workflow
    ----------------
    medicaldecathlon = MedicalDecathlonDataModule()
    medicaldecathlon.prepare_data() # download
    medicaldecathlon.setup(stage) # process and split
    medicaldecathlon.teardown(stage) # clean-up

    Parameters
    ----------
    root : Path or str, optional
        Root directory of dataset. If None, a temporary directory will be used.
        Default = ``DATA_ROOT / 'medical_decathlon'``.
    task : str, optional
        Which task to download and execute. One of
        ``'Task01_BrainTumour'``,
        ``'Task02_Heart'``,
        ``'Task03_Liver'``,
        ``'Task04_Hippocampus'``,
        ``'Task05_Prostate'``,
        ``'Task06_Lung'``,
        ``'Task07_Pancreas'``,
        ``'Task08_HepaticVessel'``,
        ``'Task09_Spleen'``, and
        ``'Task10_Colon'``.
        Default = ``'Task04_Hippocampus'``.
        See = http://medicaldecathlon.com/#tasks.
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
    """
    name: str = "medicaldecathlon"
    dataset_cls = tio.SubjectsDataset

    def __init__(
        self,
        *args: Any,
        root: PathType = DATA_ROOT / 'medical_decathlon',
        task: str = 'Task04_Hippocampus',
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        use_augmentation: bool = True,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        num_folds: int = 2,
        val_split: Union[int, float] = 0.2,
        seed: int = 41,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args,
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
                         **kwargs)
        self.use_augmentation = use_augmentation
        self.task = task
        self.task_dir = self.root / task

    def get_max_shape(self,
                      subjects: List[tio.Subject]) -> npt.NDArray[np.int_]:
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
        shapes = np.array([subject.spatial_shape for subject in dataset])
        return shapes.max(axis=0)

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """Saves files to task data directory."""
        if not is_dir_or_symlink(self.task_dir):
            url = ('https://msd-for-monai.s3-us-west-2.amazonaws.com/' +
                   f'{self.task}.tar')
            output_dir = ensure_exists(self.root)
            download_and_extract(url=url, output_dir=output_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates train, validation and test collection of samplers.

        Parameters
        ----------
        stage: Optional[str]
            Either ``'fit``, ``'validate'``, ``'test'``, or ``'predict'``.
            If stage = None, set-up all stages. Default = None.
        """
        dims = []
        if stage == "fit" or stage is None:
            train_transforms = (self.default_transforms()
                                if self.train_transforms is None else
                                self.train_transforms)

            val_transforms = (self.default_transforms()
                              if self.val_transforms is None else
                              self.val_transforms)

            train_subjects = self.get_subjects()
            dims.append([self.get_max_shape(train_subjects)])
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
            test_transforms = (self.default_transforms()
                               if self.test_transforms is None else
                               self.test_transforms)
            test_subjects = self.get_subjects(train=False)
            dims.append([self.get_max_shape(test_subjects)])
            test_dataset = self.dataset_cls(test_subjects,
                                            transform=test_transforms)
            self.test_datasets.append(test_dataset)
            self.test_datasets.append(test_dataset)
            self.size_test = min([len(data) for data in self.test_datasets])

        if stage == "predict" or stage is None:
            predict_transforms = (self.default_transforms()
                                  if self.test_transforms is None else
                                  self.test_transforms)
            predict_subjects = self.get_subjects(train=False)
            predict_dataset = self.dataset_cls(predict_subjects,
                                               transform=predict_transforms)
            self.predict_datasets.append(predict_dataset)
            self.size_predict = min(
                [len(data) for data in self.predict_datasets])

        self.dims = np.concatenate(dims, axis=0).max(axis=0)

    @staticmethod
    def get_niis(directory: Path) -> List[Path]:
        """Get paths to nii files in the given directory."""
        return sorted(path for path in directory.glob('*.nii*')
                      if not path.name.startswith('.'))

    def get_train_paths(self) -> Tuple[List[Path], List[Path]]:
        """
        Get paths to nii files for train images and labels.

        Returns
        -------
        _ : Tuple[List[Path], List[Path]]
            Paths to train images and labels.
        """
        image_training_paths = self.get_niis(self.task_dir / 'imagesTr')
        label_training_paths = self.get_niis(self.task_dir / 'labelsTr')
        return image_training_paths, label_training_paths

    def get_test_paths(self) -> List[Path]:
        """
        Get paths to nii files for test images.

        Returns
        -------
        _ : List[Path]
            Paths to test images.
        """
        image_test_paths = self.get_niis(self.task_dir / 'imagesTs')
        return image_test_paths

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
            train_img_paths, train_label_paths = self.get_train_paths()
            train_subjects = []
            for image_path, label_path in zip(train_img_paths,
                                              train_label_paths):
                # 'image' and 'label' are arbitrary names for the images
                subject = tio.Subject(image=tio.ScalarImage(image_path),
                                      label=tio.LabelMap(label_path))
                train_subjects.append(subject)
            return train_subjects

        test_img_paths = self.get_test_paths()
        test_subjects = []
        for image_path in test_img_paths:
            subject = tio.Subject(image=tio.ScalarImage(image_path))
            test_subjects.append(subject)

        return test_subjects

    def get_preprocessing_transforms(self) -> Callable:
        """
        Get preprocessing transorms to apply to all subjects.

        Returns
        -------
        preprocess : tio.Compose
            All preprocessing steps that should be applied to all subjects.
        """
        train_subjects = self.get_subjects()
        test_subjects = self.get_subjects(train=False)
        preprocess = tio.Compose([
            tio.RescaleIntensity((-1, 1)),
            tio.CropOrPad(self.get_max_shape(train_subjects + test_subjects)),
            tio.EnsureShapeMultiple(8),  # for the U-Net
            tio.OneHot(),
        ])
        return preprocess

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
        preprocess = self.get_preprocessing_transforms()
        transforms.append(preprocess)
        if stage == "fit" or stage is None and self.use_augmentation:
            augment = self.get_augmentation_transforms()
            transforms.append(augment)

        return tio.Compose(transforms)
