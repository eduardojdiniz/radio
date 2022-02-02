#!/usr/bin/env python
# coding=utf-8
"""
Dataloaders are based on the PyTorch ``torch.utils.data.Dataloader`` data
primitive. They are wrappers around ``torch.utils.data.Dataset`` that enable
easy access to the dataset samples, i.e., they prepare your data for
training/testing. Specifically, dataloaders are iterables that abstracts the
complexity of retrieving "minibatches" from Datasets, reshuffling the data at
every epoch to reduce model overfitting, use Python's ``multiprocessing``
to speed up data retrieval, and automatic memory pinning, in an easy API.
"""

from typing import (Any, Callable, List, TypeVar, Iterator, Tuple, Union, Dict,
                    Sequence, Optional)
import numpy as np
from sklearn.model_selection import KFold  # type: ignore
import torch
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from radio.data.dataset import DatasetType

Type = TypeVar("Type")

GenericTrainDataLoaderType = (Union[(
    Type,
    Sequence[Type],
    Sequence[Sequence[Type]],
    Sequence[Dict[str, Type]],
    Dict[str, Type],
    Dict[str, Dict[str, Type]],
    Dict[str, Sequence[Type]],
)])
GenericEvalDataLoaderType = Union[Type, Sequence[Type]]

TrainDataLoaderType = GenericTrainDataLoaderType[DataLoader]
EvalDataLoaderType = GenericEvalDataLoaderType[DataLoader]

WorkerInitFnType = Callable[[int], None]

# Ideally we would parameterize `DataLoader` by the return type of
# `collate_fn`, but there is currently no way to have that type parameter set
# to a default value if the user doesn't pass in a custom 'collate_fn'.
# See https://github.com/python/mypy/issues/3737.
CollateFnType = Callable[[List[Type]], Any]

__all__ = [
    "KFoldValidation", "OneFoldValidation", "TrainDataLoaderType",
    "EvalDataLoaderType", "ValidationType"
]


class KFoldValidation:
    """
    Create train and validation dataloaders for K-Fold Cross-Validation.

    Parameters
    ----------
    train_dataset : DatasetType
        Dataset from which to load the train data.
    val_dataset : DatasetType or None
        Dataset from which to load the validation data. If None, load the
        validation data from the train_dataset. ``val_dataset`` must be of the
        same size as ``train_dataset``. Default = None.
    batch_size : int, optional
        How many samples per batch to load. Default = ``32``.
    shuffle : bool, optional
        Whether to shuffle the data before splitting into batches. Note that
        the samples within each split will not be shuffled.
        Default = ``False``.
    num_workers : int, optional
        How many subprocesses to use for data loading. ``0`` means that the
        data will be loaded in the main process. Default: ``0``.
    collate_fn : Callable, optional
        Merges a list of samples to form a mini-batch of Tensor(s). Used when
        using batched loading from a map-style dataset.
    pin_memory : bool, optional
        If ``True``, the data loader will copy Tensors into CUDA pinned memory
        before returning them.
    drop_last : bool, optional
        Set to ``True`` to drop the last incomplete batch, if the dataset size
        is not divisible by the batch size. If ``False`` and the size of
        dataset is not divisible by the batch size, then the last batch will be
        smaller. Default = ``False``.
    worker_init_fn : Callable, optional
        If not ``None``, this will be called on each worker subprocess with the
        worker id (an int in ``[0, num_workers - 1]``) as input, after seeding
        and before data loading. Default = ``None``.
    num_folds : int, optional
        Number of folds. Must be at least ``2``. Default = ``5``.
    seed : int, optional
        When `shuffle` is True, `seed` affects the ordering of the indices,
        which controls the randomness of each fold. It is also use to seed the
        RNG used by RandomSampler to generate random indexes and
        multiprocessing to generate `base_seed` for workers. Pass an int for
        reproducible output across multiple function calls. Default = ``41``.
    """

    def __init__(
        self,
        train_dataset: DatasetType,
        val_dataset: DatasetType = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        collate_fn: CollateFnType = None,
        pin_memory: bool = True,
        drop_last: bool = False,
        worker_init_fn: WorkerInitFnType = None,
        num_folds: int = 5,
        seed: int = 41,
    ) -> None:
        self.train_dataset = train_dataset
        if val_dataset:
            msg = "len of val_dataset must be the same len of train_dataset."
            assert len(train_dataset) == len(val_dataset), msg
        self.val_dataset = val_dataset if val_dataset else train_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.worker_init_fn = worker_init_fn
        self.generator = torch.Generator().manual_seed(seed)
        self.kfold = KFold(n_splits=num_folds,
                           shuffle=shuffle,
                           random_state=seed)
        self.fold: int
        self.kfold_split: Iterator[np.ndarray]
        self.train_samplers: List[SubsetRandomSampler] = []
        self.val_samplers: List[SubsetRandomSampler] = []
        self.size_train: Optional[int] = None
        self.size_val: Optional[int] = None

    def __iter__(self):
        self.fold = -1
        self.kfold_split = self.kfold.split(self.train_dataset)
        return self

    def __next__(self) -> Tuple[int, DataLoader, DataLoader]:
        train_sampler, val_sampler = self._get_samplers()
        self.train_samplers.append(train_sampler)
        self.val_samplers.append(val_sampler)
        self.fold += 1
        return (
            self.fold,
            self._get_dataloader(self.fold),
            self._get_dataloader(self.fold, train=False),
        )

    def setup(self, val_split: Union[int, float] = 0.2) -> None:
        """
        Creates train and validation collection of samplers.

        Parameters
        ----------
        val_split: int or float, optional
            WARNING: val_split is not used in K-Fold validation. Left here just
            for compatibility with `OneFoldValidation`. Specify how the
            train_dataset should be split into train/validation datasets.
            Default = ``0.2``.
        """
        if val_split != 0.2:
            print((
                'WARNING: val_split is not used in K-Fold validation',
                'Left here just for compatibilit with ``OneFoldValidation``.'))

        self.kfold_split = self.kfold.split(self.train_dataset)
        for train_idx, val_idx in self.kfold_split:
            train_sampler = SubsetRandomSampler(train_idx,
                                                generator=self.generator)
            val_sampler = SubsetRandomSampler(val_idx,
                                              generator=self.generator)
            self.train_samplers.append(train_sampler)
            self.val_samplers.append(val_sampler)
        self.size_train = len(self.train_samplers[0])
        self.size_val = len(self.val_samplers[0])

    def _get_samplers(self) -> Tuple[SubsetRandomSampler, SubsetRandomSampler]:
        """Splits the dataset into train and validation samplers."""
        train_idx, val_idx = self._get_indexes()
        train_sampler = SubsetRandomSampler(train_idx,
                                            generator=self.generator)
        val_sampler = SubsetRandomSampler(val_idx, generator=self.generator)
        return (train_sampler, val_sampler)

    def _get_indexes(self) -> Tuple[List[int], List[int]]:
        """Get train and validation sample indexes."""
        train_idx, val_idx = next(self.kfold_split)
        return (train_idx, val_idx)

    def train_dataloader(self) -> TrainDataLoaderType:
        """
        Generates one or multiple Pytorch DataLoaders for train.

        Parameters
        ----------
        sampler : Sampler
            Sampler for validation samples.

        Returns
        -------
        _ : Collection of DataLoader
            Collection of train dataloaders specifying training samples.
        """
        dataloaders = []
        num_dataloaders = len(self.train_samplers)
        for idx in range(num_dataloaders):
            dataloaders.append(self._get_dataloader(idx))
        return dataloaders

    def val_dataloader(self) -> EvalDataLoaderType:
        """
        Generates one or multiple Pytorch DataLoaders for validation.

        Parameters
        ----------
        sampler : Sampler
            Sampler for validation samples.

        Returns
        -------
        _ : Collection of DataLoader
            Collection of validation dataloaders specifying validation samples.
        """
        dataloaders = []
        num_dataloaders = len(self.val_samplers)
        for idx in range(num_dataloaders):
            dataloaders.append(self._get_dataloader(idx, train=False))
        return dataloaders

    def _get_dataloader(self,
                        dataloader_idx: int,
                        train: bool = True) -> DataLoader:
        """
        Get train or validation dataloader.

        Parameters
        ----------
        dataloader_idx: int
            Dataloader index.
        train : bool, optional
            If True, return a loader for the train dataset, else for the
            validation dataset. Default = ``True``.

        Returns
        -------
        _ : DataLoader
            Train or validation dataloader.
        """
        dataset = self.train_dataset if train else self.val_dataset
        sampler = (self.train_samplers[dataloader_idx]
                   if train else self.val_samplers[dataloader_idx])
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            worker_init_fn=self.worker_init_fn,
            generator=self.generator,
        )


class OneFoldValidation:
    """
    Random split dataset into train and validation dataloaders.

    Parameters
    ----------
    train_dataset : DatasetType
        Dataset from which to load the train data.
    val_dataset : DatasetType or None
        Dataset from which to load the validation data. If None, load the
        validation data from the train_dataset. ``val_dataset`` must be of the
        same size as ``train_dataset``. Default = None.
    batch_size : int, optional
        How many samples per batch to load. Default = ``32``.
    shuffle : bool, optional
        Whether to shuffle the data at every epoch. Default = ``False``.
    num_workers : int, optional
        How many subprocesses to use for data loading. ``0`` means that the
        data will be loaded in the main process. Default: ``0``.
    collate_fn : Callable, optional
        Merges a list of samples to form a mini-batch of Tensor(s). Used when
        using batched loading from a map-style dataset.
    pin_memory : bool, optional
        If ``True``, the data loader will copy Tensors into CUDA pinned memory
        before returning them.
    drop_last : bool, optional
        Set to ``True`` to drop the last incomplete batch, if the dataset size
        is not divisible by the batch size. If ``False`` and the size of
        dataset is not divisible by the batch size, then the last batch will be
        smaller. Default = ``False``.
    worker_init_fn : Callable, optional
        If not ``None``, this will be called on each worker subprocess with the
        worker id (an int in ``[0, num_workers - 1]``) as input, after seeding
        and before data loading. Default = ``None``.
    num_folds : int, optional
        WARNING: ``num_folds`` shouldn't be set, it is hard-coded to ``2``.
        Parameter was only added for compatibility with KFoldValidation.
        Default = ``2``.
    seed : int, optional
        When `shuffle` is True, `seed` affects the ordering of the indices,
        which controls the randomness of each fold. It is also use to seed the
        RNG used by RandomSampler to generate random indexes and
        multiprocessing to generate `base_seed` for workers. Pass an int for
        reproducible output across multiple function calls. Default = ``41``.
    """

    def __init__(
        self,
        train_dataset: DatasetType,
        val_dataset: DatasetType = None,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        collate_fn: CollateFnType = None,
        pin_memory: bool = True,
        drop_last: bool = False,
        worker_init_fn: WorkerInitFnType = None,
        num_folds: int = 2,
        seed: int = 41,
    ) -> None:
        self.train_dataset = train_dataset
        if val_dataset:
            msg = "len of val_dataset must be the same len of train_dataset."
            assert len(train_dataset) == len(val_dataset), msg
        if num_folds != 2:
            print(
                ("Warning: ``num_folds`` provided but will not be used. ",
                 "``num_folds`` is set to ``2``, i.e., train and val folds."))
        self.val_dataset = val_dataset if val_dataset else train_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.worker_init_fn = worker_init_fn
        self.generator = torch.Generator().manual_seed(seed)
        self.train_samplers: List[SubsetRandomSampler] = []
        self.val_samplers: List[SubsetRandomSampler] = []
        self.size_train: Optional[int] = None
        self.size_val: Optional[int] = None

    def __call__(
        self,
        val_split: Union[int, float] = 0.2
    ) -> Tuple[TrainDataLoaderType, EvalDataLoaderType]:
        """
        Returns train and validation dataloaders.

        Parameters
        ----------
        val_split: int or float, optional
            Specify how the train_dataset should be split into
            train/validation datasets. Default = ``0.2``.

        Returns
        -------
        _ : Tuple[TrainDataLoaderType, EvalDataLoaderType]
            Tuple where the first element is a collection of train dataloaders
            and the second element is a collection of validation dataloaders.
        """
        self.setup(val_split)
        return (self.train_dataloader(), self.val_dataloader())

    def setup(self, val_split: Union[int, float] = 0.2) -> None:
        """
        Creates train and validation collection of samplers.

        Parameters
        ----------
        val_split: int or float, optional
            Specify how the train_dataset should be split into
            train/validation datasets. Default = ``0.2``.
        """
        train_sampler, val_sampler = self._get_samplers(val_split)
        self.train_samplers.append(train_sampler)
        self.val_samplers.append(val_sampler)

    def _get_samplers(
        self,
        val_split: Union[int, float] = 0.2,
    ) -> Tuple[SubsetRandomSampler, SubsetRandomSampler]:
        """get train and validation samplers."""
        len_dataset = len(self.train_dataset)
        train_idx, val_idx = self._get_indexes(val_split,
                                               len_dataset,
                                               shuffle=self.shuffle)
        train_sampler = SubsetRandomSampler(train_idx,
                                            generator=self.generator)
        val_sampler = SubsetRandomSampler(val_idx, generator=self.generator)
        return (train_sampler, val_sampler)

    @staticmethod
    def _get_indexes(val_split: Union[int, float],
                     len_dataset: int,
                     shuffle: bool = True) -> Tuple[List[int], List[int]]:
        """Get train and validation sample indexes."""
        if isinstance(val_split, int):
            train_len = len_dataset - val_split
            splits = [train_len, val_split]
        elif isinstance(val_split, float):
            val_len = int(np.floor(val_split * len_dataset))
            train_len = len_dataset - val_len
            splits = [train_len, val_len]
        else:
            raise ValueError(f"Unsupported type {type(val_split)}")
        dataset_idx = list(range(len_dataset))
        if shuffle:
            np.random.shuffle(dataset_idx)
        train_idx, val_idx = dataset_idx[:splits[0]], dataset_idx[:splits[1]]

        return (train_idx, val_idx)

    def train_dataloader(self) -> TrainDataLoaderType:
        """
        Generates one or multiple Pytorch DataLoaders for train.

        Returns
        -------
        _ : Collection of DataLoader
            Collection of train dataloaders specifying training samples.
        """
        dataloaders = []
        num_dataloaders = len(self.train_samplers)
        for idx in range(num_dataloaders):
            dataloaders.append(self._get_dataloader(idx))
        return dataloaders

    def val_dataloader(self) -> EvalDataLoaderType:
        """
        Generates one or multiple Pytorch DataLoaders for validation.

        Returns
        -------
        _ : Collection of DataLoaders
            Collection of validation dataloaders specifying validation samples.
        """
        dataloaders = []
        num_dataloaders = len(self.val_samplers)
        for idx in range(num_dataloaders):
            dataloaders.append(self._get_dataloader(idx, train=False))
        return dataloaders

    def _get_dataloader(self,
                        dataloader_idx: int,
                        train: bool = True) -> DataLoader:
        """
        Get train or validation dataloader.

        Parameters
        ----------
        dataloader_idx: int
            Dataloader index.
        train : bool, optional
            If True, return a loader for the train dataset, else for the
            validation dataset. Default = ``True``.

        Returns
        -------
        _ : DataLoader
            Train or validation dataloader.
        """
        dataset = self.train_dataset if train else self.val_dataset
        sampler = (self.train_samplers[dataloader_idx]
                   if train else self.val_samplers[dataloader_idx])
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            worker_init_fn=self.worker_init_fn,
            generator=self.generator,
        )


ValidationType = Union[OneFoldValidation, KFoldValidation]
