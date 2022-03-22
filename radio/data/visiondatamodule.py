#!/usr/bin/env python
# coding=utf-8
"""
Based on LightningDataModule for managing data. A datamodule is a shareable,
reusable class that encapsulates all the steps needed to process data, i.e.,
decoupling datasets from models to allow building dataset-agnostic models. They
also allow you to share a full dataset without explaining how to download,
split, transform, and process the data.
"""

from abc import abstractmethod
from typing import Any, Callable, Optional, Tuple
import shutil
from torch.utils.data import DataLoader

from .validation import TrainDataLoaderType, EvalDataLoaderType
from .basedatamodule import BaseDataModule

__all__ = ["VisionDataModule"]


class VisionDataModule(BaseDataModule):
    """
    Base class For making datasets which are compatible with torchvision.

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
    datamodule = VisionDataModule()
    datamodule.prepare_data() # download
    datamodule.setup(stage) # process and split
    datamodule.teardown(stage) # clean-up

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
    """

    #: Extra arguments for dataset_cls instantiation.
    EXTRA_ARGS: dict = {}
    #: Dataset class to use. E.g., torchvision.datasets.MNIST
    dataset_cls: type
    #: A tuple describing the shape of the data
    dims: Optional[Tuple[int, int, int]]
    #: Dataset name
    name: str

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """Saves files to data root dir."""
        self.dataset_cls(self.root, train=True, download=True)
        self.dataset_cls(self.root, train=False, download=True)

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

            self.train_dataset = self.dataset_cls(self.root,
                                                  train=True,
                                                  transform=train_transforms,
                                                  **self.EXTRA_ARGS)
            self.val_dataset = self.dataset_cls(self.root,
                                                train=True,
                                                transform=val_transforms,
                                                **self.EXTRA_ARGS)

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
            self.has_validation = True
            self.size_train = self.validation.size_train
            self.size_val = self.validation.size_val

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms(
                stage="test"
            ) if self.test_transforms is None else self.test_transforms
            test_dataset = self.dataset_cls(self.root,
                                            train=False,
                                            transform=test_transforms,
                                            **self.EXTRA_ARGS)
            self.test_datasets.append(test_dataset)
            self.size_test = min([len(data) for data in self.test_datasets])

        if stage == "predict" or stage is None:
            predict_transforms = self.default_transforms(
                stage="predict"
            ) if self.test_transforms is None else self.test_transforms
            predict_dataset = self.dataset_cls(self.root,
                                               train=False,
                                               transform=predict_transforms,
                                               **self.EXTRA_ARGS)
            self.predict_datasets.append(predict_dataset)
            self.size_predict = min(
                [len(data) for data in self.predict_datasets])

    @abstractmethod
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
            that should be applied to the images.
        """

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
        dataloaders = []
        num_dataloaders = len(self.train_datasets)
        for idx in range(num_dataloaders):
            dataloaders.append(
                DataLoader(
                    dataset=self.train_datasets[idx],
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=self.drop_last,
                ))
        return dataloaders

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
        dataloaders = []
        num_dataloaders = len(self.val_datasets)
        for idx in range(num_dataloaders):
            dataloaders.append(
                DataLoader(
                    dataset=self.val_datasets[idx],
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=self.drop_last,
                ))
        return dataloaders

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
                    batch_size=self.batch_size,
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
                    batch_size=self.batch_size,
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
