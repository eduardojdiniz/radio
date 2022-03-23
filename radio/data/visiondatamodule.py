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
from typing import Any, Callable, Mapping, Optional, Sequence, Sized
import shutil
from torch.utils.data import DataLoader, IterableDataset
from .dataset import DatasetType
from .validation import TrainDataLoaderType, EvalDataLoaderType
from .basedatamodule import BaseDataModule
from .datatypes import TrainSizeType, EvalSizeType

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
    data = VisionDataModule()
    data.prepare_data() # download
    data.setup(stage) # process and split
    data.teardown(stage) # clean-up

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
            Either ``'fit``, ``'validate'``, or ``'test'``.
            If stage = None, set-up all stages. Default = None.
        """
        if stage in (None, "fit"):
            train_transforms = self.default_transforms(
                stage="fit"
            ) if self.train_transforms is None else self.train_transforms

            val_transforms = self.default_transforms(
                stage="fit"
            ) if self.val_transforms is None else self.val_transforms

            train_dataset = self.dataset_cls(self.root,
                                             train=True,
                                             transform=train_transforms,
                                             **self.EXTRA_ARGS)
            val_dataset = self.dataset_cls(self.root,
                                           train=True,
                                           transform=val_transforms,
                                           **self.EXTRA_ARGS)

            self.validation = self.val_cls(train_dataset=train_dataset,
                                           val_dataset=val_dataset,
                                           batch_size=self.batch_size,
                                           shuffle=self.shuffle,
                                           num_workers=self.num_workers,
                                           pin_memory=self.pin_memory,
                                           drop_last=self.drop_last,
                                           num_folds=self.num_folds,
                                           seed=self.seed)

            self.validation.setup(self.val_split)
            self.has_validation = True

            self.train_dataset = train_dataset
            self.size_train = self.size_train_dataset(
                self.validation.train_samplers)
            self.val_dataset = val_dataset
            self.size_val = self.size_eval_dataset(
                self.validation.val_samplers)

        if stage in (None, "test"):
            test_transforms = self.default_transforms(
                stage="test"
            ) if self.test_transforms is None else self.test_transforms
            self.test_dataset = self.dataset_cls(self.root,
                                                 train=False,
                                                 transform=test_transforms,
                                                 **self.EXTRA_ARGS)
            self.size_test = self.size_eval_dataset(self.test_dataset)

    @abstractmethod
    def default_transforms(self, stage: Optional[str] = None) -> Callable:
        """
        Default transforms and augmentations for the dataset.

        Parameters
        ----------
        stage: Optional[str]
            Either ``'fit``, ``'validate'``, or ``'test'``.
            If stage = None, set-up all stages. Default = None.

        Returns
        -------
        _: Callable
            All preprocessing steps (and if ``'fit'``, augmentation steps too)
            that should be applied to the images.
        """

    @staticmethod
    def size_train_dataset(train_dataset: Sized) -> TrainSizeType:
        """
        Compute the size of the train datasets.

        Parameters
        ----------
        train_dataset: TrainDatasetType
            Collection of train datasets.

        Returns
        -------
        _ : TrainSizeType
            Collection of train datasets' sizes.
        """

        def _handle_is_mapping(dataset):
            mapping = {}
            for key, dset in dataset.items():
                if isinstance(dset, Mapping):
                    mapping[key] = _handle_is_mapping(dset)
                if isinstance(dset, Sequence):
                    mapping[key] = _handle_is_sequence(dset)
                mapping[key] = len(dset)
            return mapping

        def _handle_is_sequence(dataset):
            sequence = []
            for dset in dataset:
                if isinstance(dset, Mapping):
                    sequence.append(_handle_is_mapping(dset))
                if isinstance(dset, Sequence):
                    sequence.append(_handle_is_sequence(dset))
                sequence.append(len(dset))
            return sequence

        if isinstance(train_dataset, Mapping):
            return _handle_is_mapping(train_dataset)
        if isinstance(train_dataset, Sequence):
            return _handle_is_sequence(train_dataset)
        return len(train_dataset)

    @staticmethod
    def size_eval_dataset(eval_dataset: Sized) -> EvalSizeType:
        """
        Compute the size of the test or validation datasets.

        Parameters
        ----------
        eval_dataset: EvalDatasetType
            Collection of test or validation datasets.

        Returns
        -------
        _ : EvalSizeType
            Collection of test or validation datasets' sizes.
        """
        if isinstance(eval_dataset, Sequence):
            if len(eval_dataset) == 1:
                return len(eval_dataset[0])
            return [len(ds) for ds in eval_dataset]
        return len(eval_dataset)

    def dataloader(
        self,
        dataset: DatasetType,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        drop_last: Optional[bool] = None,
    ) -> DataLoader:
        """
        Instantiate a DataLoader.

        Parameters
        ----------
        batch_size : int, optional
            How many samples per batch to load. Default = ``32``.
        shuffle : bool, optional
            Whether to shuffle the data at every epoch. Default = ``False``.
        num_workers : int, optional
            How many subprocesses to use for data loading. ``0`` means that the
            data will be loaded in the main process. Default: ``0``.
        pin_memory : bool, optional
            If ``True``, the data loader will copy Tensors into CUDA pinned
            memory before returning them.
        drop_last : bool, optional
            Set to ``True`` to drop the last incomplete batch, if the dataset
            size is not divisible by the batch size. If ``False`` and the size
            of dataset is not divisible by the batch size, then the last batch
            will be smaller. Default = ``False``.

        Returns
        -------
        _ : DataLoader
        """
        shuffle = shuffle if shuffle else self.shuffle
        shuffle &= not isinstance(dataset, IterableDataset)
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size if batch_size else self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers if num_workers else self.num_workers,
            pin_memory=pin_memory if pin_memory else self.pin_memory,
            drop_last=drop_last if drop_last else self.drop_last,
        )

    def train_dataloader(self, *args: Any,
                         **kwargs: Any) -> TrainDataLoaderType:
        """
        Generates one or multiple Pytorch DataLoaders for train.

        Returns
        -------
        _ : Collection of DataLoader
            Collection of train dataloaders specifying training samples.
        """
        loader_kwargs = {}
        loader_kwargs["batch_size"] = kwargs.get("batch_size", None)
        loader_kwargs["shuffle"] = kwargs.get("shuffle", None)
        loader_kwargs["num_workers"] = kwargs.get("num_workers", None)
        loader_kwargs["pin_memory"] = kwargs.get("pin_memory", None)
        loader_kwargs["drop_last"] = kwargs.get("drop_last", None)

        def _handle_is_mapping(dataset):
            mapping = {}
            for key, dset in dataset.items():
                if isinstance(dset, Mapping):
                    mapping[key] = _handle_is_mapping(dset)
                if isinstance(dset, Sequence):
                    mapping[key] = _handle_is_sequence(dset)
                mapping[key] = self.dataloader(dset, **loader_kwargs)
            return mapping

        def _handle_is_sequence(dataset):
            sequence = []
            for dset in dataset:
                if isinstance(dset, Mapping):
                    sequence.append(_handle_is_mapping(dset))
                if isinstance(dset, Sequence):
                    sequence.append(_handle_is_sequence(dset))
                sequence.append(self.dataloader(dset, **loader_kwargs))
            return sequence

        if self.has_validation:
            return self.validation.train_dataloader()

        if isinstance(self.train_dataset, Mapping):
            return _handle_is_mapping(self.train_dataset)
        if isinstance(self.train_dataset, Sequence):
            return _handle_is_sequence(self.train_dataset)
        return self.dataloader(self.train_dataset, **loader_kwargs)

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

        loader_kwargs = {}
        loader_kwargs["batch_size"] = kwargs.get("batch_size", None)
        loader_kwargs["shuffle"] = kwargs.get("shuffle", None)
        loader_kwargs["num_workers"] = kwargs.get("num_workers", None)
        loader_kwargs["pin_memory"] = kwargs.get("pin_memory", None)
        loader_kwargs["drop_last"] = kwargs.get("drop_last", None)

        if isinstance(self.val_dataset, Sequence):
            if len(self.val_dataset) == 1:
                return self.dataloader(self.val_dataset[0], **loader_kwargs)
            return [
                self.dataloader(ds, **loader_kwargs) for ds in self.val_dataset
            ]
        return self.dataloader(self.val_dataset, **loader_kwargs)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> EvalDataLoaderType:
        """
        Generates one or multiple Pytorch DataLoaders for test.

        Returns
        -------
        _ : Collection of DataLoaders
            Collection of test dataloaders specifying test samples.
        """
        loader_kwargs = {}
        loader_kwargs["batch_size"] = kwargs.get("batch_size", None)
        loader_kwargs["shuffle"] = kwargs.get("shuffle", None)
        loader_kwargs["num_workers"] = kwargs.get("num_workers", None)
        loader_kwargs["pin_memory"] = kwargs.get("pin_memory", None)
        loader_kwargs["drop_last"] = kwargs.get("drop_last", None)

        if isinstance(self.test_dataset, Sequence):
            if len(self.test_dataset) == 1:
                return self.dataloader(self.test_dataset[0], **loader_kwargs)
            return [
                self.dataloader(ds, **loader_kwargs)
                for ds in self.test_dataset
            ]
        return self.dataloader(self.test_dataset, **loader_kwargs)

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Called at the end of fit (train + validate), validate, test,
        or predict. Remove root directory if a temporary was used.

        Parameters
        ----------
        stage: Optional[str]
            Either ``'fit``, ``'validate'``, or ``'test'``.
            If stage = None, set-up all stages. Default = None.
        """
        if self.is_temp_dir:
            shutil.rmtree(self.root)
