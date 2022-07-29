#!/usr/bin/env python
# coding=utf-8
"""
3T to 7T Data Module
"""

from typing import Any, Callable, List, Optional, Tuple, Union, cast
from string import Template
from pathlib import Path
from operator import itemgetter
import torchio as tio  # type: ignore
from radio.settings.pathutils import is_dir_or_symlink, PathType
from ..unpaired_dataset import MRIUnpairedDataset
from ..visiondatamodule import VisionDataModule
from ..datatypes import SpatialShapeType

__all__ = ["MRI3T27TDataModule"]


class MRI3T27TDataModule(VisionDataModule):
    """
    3T to 7T Data Module.

    Typical Workflow
    ----------------
    data = MRI3T27TDataModule()
    data.prepare_data() # download
    data.setup(stage) # process and split
    data.teardown(stage) # clean-up

    Parameters
    ----------
    root : Path or str, optional
        Root to data root folder.
        Default = ``"/media/cerebro/Workspaces/Students/Eduardo_Diniz/Studies"``.
    study : str, optional
        Study name. Default = ``'MRI3T27T'``.
    subj_dir : str, optional
        Subdirectory where the subjects are located.
        Default = ``'radio/MPR/unprocessed'``.
    data_dir : str, optional
        Subdirectory where the subjects' data are located.
        Default = ````.
    domain_a : str, Optional
        Name of the domain A. Default = ``"3T_MPR"``.
    domain_b : str, Optional
        Name of the domain B. Default = ``"7T_MPR"``.
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
        If ``True``, resample all images to ``'7T_MPR'``. Default = ``False``.
    batch_size : int, optional
        How many samples per batch to load. Default = ``4``.
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
        Which modalities to load. Default = ``['3T_MPR', '7T_MPR']``.
    labels : List[str], optional
        Which labels to load. Default = ``[]``.
    dims : Tuple[int, int, int], optional
        Max spatial dimensions across subjects' images. If ``None``, compute
        dimensions from dataset. Default = ``(256, 320, 320)``.
    seed : int, optional
        When `shuffle` is True, `seed` affects the ordering of the indices,
        which controls the randomness of each fold. It is also use to seed the
        RNG used by RandomSampler to generate random indexes and
        multiprocessing to generate `base_seed` for workers. Pass an int for
        reproducible output across multiple function calls. Default = ``41``.
    verbose : bool, optional
        If ``True``, print debugging messages. Default = ``False``.
    """
    name: str = "MRI3T27T"
    dataset_cls = MRIUnpairedDataset
    img_template = Template('${subj_id}_-_${field}_-_${modality}.nii.gz')
    label_template: Template = Template("")

    def __init__(
        self,
        *args: Any,
        root: PathType = Path(
            "/media/cerebro/Workspaces/Students/Eduardo_Diniz/Studies"),
        study: str = 'MRI3T27T',
        subj_dir: str = 'radio/MPR/unprocessed',
        data_dir: str = '',
        domain_a: str = "3T_MPR",
        domain_b: str = "7T_MPR",
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
        modalities: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        dims: Tuple[int, int, int] = (368, 480, 384),
        seed: int = 41,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        root = Path(root) / study / subj_dir
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
        self.subj_dir = subj_dir
        self.data_dir = data_dir
        self.domain_a = domain_a
        self.domain_b = domain_b
        self.modalities = modalities if modalities else ['3T_MPR', '7T_MPR']
        self.labels = labels if labels else []
        self.dims = dims
        self.use_augmentation = use_augmentation
        self.use_preprocessing = use_preprocessing
        self.resample = resample
        self.verbose = verbose

    def check_if_data_split(self, data_dir: str = "") -> None:
        """Check if data is splitted in train, test and val folders"""
        has_train_a_folder = is_dir_or_symlink(self.root / data_dir /
                                               ("train_" + self.domain_a))
        has_train_b_folder = is_dir_or_symlink(self.root / data_dir /
                                               ("train_" + self.domain_b))
        has_train_folder = bool(has_train_a_folder and has_train_b_folder)

        has_val_a_folder = is_dir_or_symlink(self.root / data_dir /
                                             ("val_" + self.domain_a))
        has_val_b_folder = is_dir_or_symlink(self.root / data_dir /
                                             ("val_" + self.domain_b))
        has_val_folder = bool(has_val_a_folder and has_val_b_folder)

        has_test_a_folder = is_dir_or_symlink(self.root / data_dir /
                                              ("test_" + self.domain_a))
        has_test_b_folder = is_dir_or_symlink(self.root / data_dir /
                                              ("test_" + self.domain_b))
        has_test_folder = bool(has_test_a_folder and has_test_b_folder)

        self.has_train_test_split = bool(has_train_folder and has_test_folder)
        self.has_train_val_split = bool(has_train_folder and has_val_folder)

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """Verify data directory exists and if test/train/val splitted."""
        if not is_dir_or_symlink(self.root):
            raise OSError('Study data directory not found!')
        self.check_if_data_split(self.data_dir)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates train, validation and test collection of samplers.

        Parameters
        ----------
        stage: Optional[str]
            Either ``'fit``, ``'validate'``, or ``'test'``.
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
                train_dataset = self.get_dataset(fold="train",
                                                 transform=train_transforms)
                val_dataset = self.get_dataset(fold="train",
                                               transform=val_transforms)
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
                self.train_dataset = self.get_dataset(
                    fold="train", transform=train_transforms)
                self.size_train = self.size_train_dataset(self.train_dataset)

                self.val_dataset = self.get_dataset(fold="val",
                                                    transform=val_transforms)
                self.size_val = self.size_eval_dataset(self.val_dataset)

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms(
                stage="test"
            ) if self.test_transforms is None else self.test_transforms
            self.test_dataset = self.get_dataset(fold="test",
                                                 transform=test_transforms)
            self.size_test = self.size_eval_dataset(self.test_dataset)

    def get_dataset(
            self,
            fold: str = "train",
            transform: Optional[Callable] = None,
            add_sampling_map: bool = False,
            patch_size: SpatialShapeType = (96, 96, 1),
    ) -> MRIUnpairedDataset:
        """
        Get train, test, or val list of TorchIO Subjects.

        Parameters
        ----------
        fold : str, optional
            Identify which type of dataset, ``'train'``, ``'test'``, or
            ``'val'``. Default = ``'train'``.
        transform : Callable, optional
            A function/transform that takes in a sample and returns a
            transformed version, e.g, ``torchvision.transforms.RandomCrop``.

        Returns
        -------
        _ : tio.SubjectsDataset
            Train, test or val dataset of TorchIO Subjects.
        """

        dataset = self.dataset_cls(
            root=self.root,
            dataset_name='',
            domain_a=self.domain_a,
            domain_b=self.domain_b,
            transform=transform,
            stage=fold,
            add_sampling_map=add_sampling_map,
            patch_size=patch_size,
        )

        return dataset

    def get_preprocessing_transforms(
        self,
        shape: Optional[Tuple[int, int, int]] = None,
        resample: bool = False,
        resample_reference: str = '7T_MPR',
    ) -> tio.Transform:
        """
        Get preprocessing transorms to apply to all subjects.

        Parameters
        ----------
        shape : Tuple[int, int, int], Optional
            A tuple with the shape for the CropOrPad transform.
            Default = ``None``.
        resample : bool, Optional
            If True, resample to ``resample_reference``. Default = ``False``.
        resample_reference : str, Optional
            Name of the image to resample to. Default = ``"7T_MPR"```.

        Returns
        -------
        preprocess : tio.Transform
            All preprocessing steps that should be applied to all subjects.
        """
        preprocess_list: List[tio.transforms.Transform] = []

        # Use standard orientation for all images
        preprocess_list.append(tio.ToCanonical())

        # If true, resample to ``resample_reference``
        if resample:
            preprocess_list.append(tio.Resample(resample_reference))

        if shape is None:
            train_shape = self.get_dataset(fold="train").get_max_shape()
            test_shape = self.get_dataset(fold="test").get_max_shape()
            shapes = [train_shape, test_shape]
            max_shape = tuple((max(shapes, key=itemgetter(i))[i]
                               for i in range(len(train_shape))))
            shape = cast(Tuple[int, int, int], max_shape)
        else:
            shape = self.dims

        preprocess_list.extend([
            # tio.RescaleIntensity((-1, 1)),
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
