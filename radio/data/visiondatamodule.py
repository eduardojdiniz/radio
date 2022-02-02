#!/usr/bin/env python
# coding=utf-8
"""
Based on LightningDataModule for managing data. A datamodule is a shareable,
reusable class that encapsulates all the steps needed to process data, i.e.,
decoupling datasets from models to allow building dataset-agnostic models. They
also allow you to share a full dataset without explaining how to download,
split, transform, and process the data.
"""

from abc import ABCMeta, abstractmethod
from typing import Optional, Callable, Any, Union, Type, List
from pathlib import Path
import shutil
import tempfile
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from radio.settings.pathutils import DATA_ROOT, PathType
from .validation import (TrainDataLoaderType, EvalDataLoaderType,
                         KFoldValidation, OneFoldValidation, ValidationType)
from .dataset import DatasetType


class BaseDataModule(pl.LightningDataModule, metaclass=ABCMeta):
    """
    Base class for making data modules.

    To create a subclass, you need to implement the following functions:

    A BaseDataModule needs to implement 6 key methods:
    <__init__>:
        (Optionally) Initialize the class, first call super.__init__().
    <prepare_data>:
        Things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode.
    <setup>:
        Things to do on every accelerator in distributed mode.
    <train_dataloader>:
        The training dataloader.
    <val_dataloader>:
        The validation dataloader(s).
    <test_dataloader>:
        The test dataloader(s).
    <predict_dataloader>:
        The prediction dataloader(s).
    <teardown>:
        Things to do on every accelerator in distributed mode when finished.

    Typical Workflow
    ----------------
    datamodule = BaseDataModule()
    datamodule.prepare_data() # download
    datamodule.setup(stage) # process and split
    datamodule.teardown(stage) # clean-up

    Parameters
    ----------
    root : Path or str, optional
        Root directory of dataset. If None, a temporary directory will be used.
        Default = ``DATA_ROOT / 'medical_decathlon'``.
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

    def __init__(
        self,
        *args: Any,
        root: PathType = DATA_ROOT,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
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

        super().__init__(*args, **kwargs)
        num_folds_msg = "``num_folds`` must be an integer of at least 2."
        assert isinstance(num_folds, int) and num_folds > 1, num_folds_msg
        self.root = Path(tempfile.mkdtemp()) if root is None else Path(root)
        self.is_temp_dir = bool(root is None)
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.num_folds = num_folds
        self.val_cls: Type[ValidationType] = (OneFoldValidation if num_folds
                                              == 2 else KFoldValidation)
        self.val_split = val_split
        self.seed = seed
        self.validation: ValidationType
        self.train_dataset: DatasetType
        self.val_dataset: DatasetType
        self.test_datasets: List[DatasetType] = []
        self.predict_datasets: List[DatasetType] = []
        self.size_train: Optional[int] = None
        self.size_val: Optional[int] = None
        self.size_test: Optional[int] = None
        self.size_predict: Optional[int] = None

    @abstractmethod
    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Data operations to be performed that need to be done only from a single
        process in distributed settings. For example, download and saving data.

        Warning
        -------
        DO NOT set state here (use ``'setup'`` instead) since this is NOT
        called on every device.

        Example
        -------
        def prepare_data(self, *args, **kwargs):
            # good
            download_data()
            tokenize()
            etc()

            # good
            torchvison.datasets.MNIST(self.root, train=True, download=True)
            torchvison.datasets.MNIST(self.root, train=False, download=True)

            # bad
            self.split = data_split
            self.some_state = some_other_state()
        """

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Data operations to be performed on every GPU. This is a good hook when
        you need to build models dynamically or adjust something about them
        based on the data atrributes. For example, count number of classes,
        build vocabulary, perform train/val/test splits, apply transforms.

        Parameters
        ----------
        stage: Optional[str]
            Either ``'fit``, ``'validate'``, ``'test'``, or ``'predict'``.
            If stage = None, set-up all stages. Default = None.

        Example
        -------
        def setup(self, stage):
            if stage in (None, "fit"):
                # train + validation set-up
                self.data_train, self.data_val = load_fit(...)
                self.dims = self.data_train[0][0].shape
                self.l1 = nn.Linear(28, self.data_train.num_classes)
            if stage in (None, "test"):
                # test set-up
                self.data_test = load_test(...)
                self.dims = self.data_test[0][0].shape
                self.l1 = nn.Linear(28, self.data_test.num_classes)
        """

    @abstractmethod
    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Called at the end of fit (train + validate), validate, test,
        or predict.

        Parameters
        ----------
        stage: Optional[str]
            Either ``'fit``, ``'validate'``, ``'test'``, or ``'predict'``.
            Default = None.
        """

    @abstractmethod
    def train_dataloader(self, *args: Any,
                         **kwargs: Any) -> TrainDataLoaderType:
        """
        Generates one or multiple Pytorch DataLoaders for training.

        Returns
        -------
        _ : Collection of DataLoader
            Collection of train dataloaders specifying training samples.

        Examples
        -------
        # single dataloader
        def train_dataloader(self):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (1.0,))
            ])
            dataset = MNIST(
                root='/path/to/mnist/',
                train=True,
                transform=transform,
                download=True
            )
            loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            return loader

        # multiple dataloaders, return as list
        def train_dataloader(self):
            mnist = MNIST(...)
            cifar = CIFAR(...)
            mnist_loader = torch.utils.data.DataLoader(
                dataset=mnist, batch_size=self.batch_size, shuffle=True
            )
            cifar_loader = torch.utils.data.DataLoader(
                dataset=cifar, batch_size=self.batch_size, shuffle=True
            )
            # each batch will be a list of tensors: [batch_mnist, batch_cifar]
            return [mnist_loader, cifar_loader]

        # multiple dataloader, return as dict
        def train_dataloader(self):
            mnist = MNIST(...)
            cifar = CIFAR(...)
            mnist_loader = torch.utils.data.DataLoader(
                dataset=mnist, batch_size=self.batch_size, shuffle=True
            )
            cifar_loader = torch.utils.data.DataLoader(
                dataset=cifar, batch_size=self.batch_size, shuffle=True
            )
            # each batch will be a dict of
            tensors: {'mnist': batch_mnist, 'cifar': batch_cifar}
            return {'mnist': mnist_loader, 'cifar': cifar_loader}
        """

    @abstractmethod
    def val_dataloader(self, *args: Any, **kwargs: Any) -> EvalDataLoaderType:
        """
        Generates one or multiple Pytorch DataLoaders for validation.

        Returns
        -------
        _ : Collection of DataLoader
            Collection of validation dataloaders specifying validation samples.

        Examples
        -------
        # single dataloader
        def val_dataloader(self):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (1.0,))
            ])
            dataset = MNIST(
                root='/path/to/mnist/',
                train=False,
                transform=transform,
                download=True
            )
            loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
            return loader

        # multiple dataloaders, return as list
        def val_dataloader(self):
            mnist = MNIST(...)
            cifar = CIFAR(...)
            mnist_loader = torch.utils.data.DataLoader(
                dataset=mnist, batch_size=self.batch_size, shuffle=False
            )
            cifar_loader = torch.utils.data.DataLoader(
                dataset=cifar, batch_size=self.batch_size, shuffle=False
            )
            # each batch will be a list of tensors: [batch_mnist, batch_cifar]
            return [mnist_loader, cifar_loader]

        # multiple dataloader, return as dict
        def val_dataloader(self):
            mnist = MNIST(...)
            cifar = CIFAR(...)
            mnist_loader = torch.utils.data.DataLoader(
                dataset=mnist, batch_size=self.batch_size, shuffle=False
            )
            cifar_loader = torch.utils.data.DataLoader(
                dataset=cifar, batch_size=self.batch_size, shuffle=False
            )
            # each batch will be a dict of
            tensors: {'mnist': batch_mnist, 'cifar': batch_cifar}
            return {'mnist': mnist_loader, 'cifar': cifar_loader}
        """

    @abstractmethod
    def test_dataloader(self, *args: Any, **kwargs: Any) -> EvalDataLoaderType:
        """
        Generates one or multiple Pytorch DataLoaders for testing.

        Returns
        -------
        _ : Collection of DataLoader
            Collection of test dataloaders specifying testing samples.

        Examples
        -------
        # single dataloader
        def test_dataloader(self):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (1.0,))
            ])
            dataset = MNIST(
                root='/path/to/mnist/',
                train=False,
                transform=transform,
                download=True
            )
            loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
            return loader

        # multiple dataloaders, return as list
        def test_dataloader(self):
            mnist = MNIST(...)
            cifar = CIFAR(...)
            mnist_loader = torch.utils.data.DataLoader(
                dataset=mnist, batch_size=self.batch_size, shuffle=False
            )
            cifar_loader = torch.utils.data.DataLoader(
                dataset=cifar, batch_size=self.batch_size, shuffle=False
            )
            # each batch will be a list of tensors: [batch_mnist, batch_cifar]
            return [mnist_loader, cifar_loader]

        # multiple dataloader, return as dict
        def test_dataloader(self):
            mnist = MNIST(...)
            cifar = CIFAR(...)
            mnist_loader = torch.utils.data.DataLoader(
                dataset=mnist, batch_size=self.batch_size, shuffle=False
            )
            cifar_loader = torch.utils.data.DataLoader(
                dataset=cifar, batch_size=self.batch_size, shuffle=False
            )
            # each batch will be a dict of
            tensors: {'mnist': batch_mnist, 'cifar': batch_cifar}
            return {'mnist': mnist_loader, 'cifar': cifar_loader}
        """

    @abstractmethod
    def predict_dataloader(self, *args: Any,
                           **kwargs: Any) -> EvalDataLoaderType:
        """
        Generates one or multiple Pytorch DataLoaders for prediction.

        Returns
        -------
        _ : Collection of DataLoader
            Collection of prediction dataloaders specifying prediction samples.

        Examples
        -------
        # single dataloader
        def predict_dataloader(self):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (1.0,))
            ])
            dataset = MNIST(
                root='/path/to/mnist/',
                train=False,
                transform=transform,
                download=True
            )
            loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=False
            )
            return loader

        # multiple dataloaders, return as list
        def predict_dataloader(self):
            mnist = MNIST(...)
            cifar = CIFAR(...)
            mnist_loader = torch.utils.data.DataLoader(
                dataset=mnist, batch_size=self.batch_size, shuffle=False
            )
            cifar_loader = torch.utils.data.DataLoader(
                dataset=cifar, batch_size=self.batch_size, shuffle=False
            )
            # each batch will be a list of tensors: [batch_mnist, batch_cifar]
            return [mnist_loader, cifar_loader]

        # multiple dataloader, return as dict
        def predict_dataloader(self):
            mnist = MNIST(...)
            cifar = CIFAR(...)
            mnist_loader = torch.utils.data.DataLoader(
                dataset=mnist, batch_size=self.batch_size, shuffle=False
            )
            cifar_loader = torch.utils.data.DataLoader(
                dataset=cifar, batch_size=self.batch_size, shuffle=False
            )
            # each batch will be a dict of
            tensors: {'mnist': batch_mnist, 'cifar': batch_cifar}
            return {'mnist': mnist_loader, 'cifar': cifar_loader}
        """


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
    visiondatamodule = VisionDataModule()
    visiondatamodule.prepare_data() # download
    visiondatamodule.setup(stage) # process and split
    visiondatamodule.teardown(stage) # clean-up

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
    dims: tuple
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
            ) if self.train_transforms is None else self.train_transforms

            val_transforms = self.default_transforms(
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
            self.size_train = self.validation.size_train
            self.size_val = self.validation.size_val

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms(
            ) if self.test_transforms is None else self.test_transforms
            test_dataset = self.dataset_cls(self.root,
                                            train=False,
                                            transform=test_transforms,
                                            **self.EXTRA_ARGS)
            self.test_datasets.append(test_dataset)
            self.size_test = min([len(data) for data in self.test_datasets])

        if stage == "predict" or stage is None:
            test_transforms = self.default_transforms(
            ) if self.test_transforms is None else self.test_transforms
            predict_dataset = self.dataset_cls(self.root,
                                               train=False,
                                               transform=test_transforms,
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
