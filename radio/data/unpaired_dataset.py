#!/usr/bin/env python
# coding=utf-8
"""
This module implements the Unpaired Dataset class.
"""

import sys
from string import Template
from pathlib import Path
from multiprocessing import Manager
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
import numpy as np
import torch
import torchio as tio  # type: ignore
from radio.settings.pathutils import (DATA_ROOT, is_dir_or_symlink, PathType,
                                      MRI_EXTENSIONS)
from .dataset import FolderDataset
from .datautils import mri_image_loader, create_probability_map
from .datatypes import SpatialShapeType

Sample = Tuple[Path, int]
PairedSample = Dict[str, Tuple[Any, ...]]

__all__ = ["UnpairedDataset", "MRIUnpairedDataset", "MRISliceUnpairedDataset"]


class UnpairedDataset(FolderDataset):
    """
    This dataset class can load unpaired/unaligned datasets.

    It requires two directories to host training images from domain A
    '/path/to/data/train_A' and from domain B 'path/to/data/train_B',
    respectively.
    Similarly, you need to prepare two directories '/path/to/data/test_A' and
    '/path/to/data/test_B' during test time.

    This class inherits from :class:`FolderDataset` so the same methods can be
    overridden to customize the dataset.

    Attributes
    ----------
    classes : list
        List of the class names sorted alphabetically.
    num_classes : int
        Number of classes in the dataset.
    class_to_idx : dict
        Dict with items (class_name, class_index).
    samples : list
        List of (images, class_index) tuples
    targets : list
        The class_index value for each image in the dataset

    Parameters
    ----------
    root : Path or str, Optional
        Data root directory. Where to save/load the data.
        Default = ``DATA_ROOT``.
    dataset_name : str, Optional
        The name of the dataset. Default = ``a_dataset``.
    domain_a : str, Optional
        The name of the source domain. Default = ``"A"``.
    domain_b : str, Optional
        The name of the destination domain. Default = ``"B"``.
    transform : Optional[Callable]
        A function/transform that takes in an PIL image and returns a
        transformed version, e.g, ``torchvision.transforms.RandomCrop``.
    target_transform : Callable[Optional]
        Optional function/transform that takes in a target and returns a
        transformed version.
    return_paths : bool
        If True, calling the dataset returns `(sample, target, /path/to/sample,
        /path/to/target)` instead of returning `(sample, target)`.
    """
    folder_template = Template('${stage}_${domain}')

    def __init__(
        self,
        loader: Callable[[Path], Any],
        root: PathType = DATA_ROOT,
        dataset_name: str = "a_dataset",
        domain_a: str = "A",
        domain_b: str = "B",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[Path], bool]] = None,
        return_paths: bool = False,
        max_class_size: int = sys.maxsize,
        max_dataset_size: int = sys.maxsize,
        stage: str = "train",
    ) -> None:
        # Arguments parsing
        root = Path(root).expanduser() / dataset_name
        assert is_dir_or_symlink(
            root), f"Dataset not found. {root} is not a valid directory."

        super().__init__(
            loader=loader,
            root=root,
            transform=transform,
            target_transform=target_transform,
            extensions=extensions,
            is_valid_file=is_valid_file,
            return_paths=return_paths,
            max_class_size=max_class_size,
            max_dataset_size=max_dataset_size,
        )
        self.domain_a = domain_a
        self.domain_b = domain_b

        samples_dict = self.get_samples(stage)
        self._subjects_a = Manager().list(samples_dict[self.domain_a])
        self._subjects_b = Manager().list(samples_dict[self.domain_b])
        self.size_a = len(self._subjects_a)
        self.size_b = len(self._subjects_b)

    def dry_iter_a(self):
        """Return the internal list of subjects.

        This can be used to iterate over the subjects without loading the data
        and applying any transforms::

        >>> names = [subject.name for subject in dataset.dry_iter()]
        """
        return self._subjects_a

    def dry_iter_b(self):
        """Return the internal list of subjects.

        This can be used to iterate over the subjects without loading the data
        and applying any transforms::

        >>> names = [subject.name for subject in dataset.dry_iter()]
        """
        return self._subjects_b

    def get_samples(self, stage: str) -> Dict[str, List[Sample]]:
        """
        Construct a dictionary where the keys are the sample classes and the
        values are a list of (sample, class_idx).

        Parameters
        ----------
        stage: Optional[str]
            Either ``'train``, ``'val'``, or ``'test'``. Default = ``train``.

        Returns
        -------
        _: {class_label: [(sample, class_idx), ...]}
        """
        class_to_domain: Dict[str, str] = {}
        for domain in [self.domain_a, self.domain_b]:
            class_to_domain.update({
                self.folder_template.substitute(stage=stage, domain=domain):
                domain
            })

        idx_to_class = {
            v: k
            for k, v in self.class_to_idx.items() if k in class_to_domain
        }
        idx_to_domain = {
            k: class_to_domain[v]
            for k, v in idx_to_class.items()
        }

        samples = {}
        for i, domain in enumerate([self.domain_a, self.domain_b]):
            samples[domain] = [
                (mri_image_loader(s[0]), i) for s in self.samples
                if s[1] in idx_to_domain and idx_to_domain[int(s[1])] == domain
            ]

        return samples

    def __getitem__(self, idx: int) -> PairedSample:
        """
        Parameters
        ----------
        idx : int
            A (random) integer for data intexing.

        Returns
        -------
        _: Tuple[Any, ...]
            (sample, target) where target is class_index of the target class.
            (sample, target, /path/to/sample) if ``self.return_paths = True``.
        """
        try:
            idx = int(idx)
        except (RuntimeError, TypeError) as idx_not_int:
            message = (f'Index "{idx}" must be int or compatible dtype,'
                       f' but an object of type "{type(idx)}" was passed.')
            raise ValueError(message) from idx_not_int

        # Make sure indexes are within A and B ranges
        # sample_a, sample_b = self.loader(path_a), self.loader(path_b)
        sample_a, target_a = self._subjects_a[idx % self.size_a]
        sample_b, target_b = self._subjects_b[idx % self.size_b]

        if self.transform is not None:
            sample_a = self.transform(sample_a)
            sample_b = self.transform(sample_b)

        if self.target_transform is not None:
            target_a = self.target_transform(target_a)
            target_b = self.target_transform(target_b)

        if self.return_paths:
            return {
                self.domain_a: (sample_a, target_a, str(sample_a.mri.path)),
                self.domain_b: (sample_b, target_b, str(sample_b.mri.path)),
            }

        return {
            self.domain_a: (sample_a, target_a),
            self.domain_b: (sample_b, target_b),
        }

    def __len__(self) -> int:
        """
        As we have two datasets with potentially different number of images, we
        take a maximum of.
        """
        return max(self.size_a, self.size_b)


class MRIUnpairedDataset(UnpairedDataset):
    """
    This dataset class can load unpaired/unaligned 3D medical images.
    It is based on torch.utils.data.Dataset,
    torchvision.datasets.VisionDataset, and torchio.SubjectsDataset.

    It requires two directories to host training images from domain `A`
    '/path/to/data/train_A' and from domain `B` 'path/to/data/train_B',
    respectively.
    Similarly, you need to prepare two directories '/path/to/data/test_A' and
    '/path/to/data/test_B' during test time.


    It defaults to a generic MIR folder dataset where the images are arranged
    in this way by default:

    root/
    ├── train_A
    │   ├── xxx.nii
    │   ├── xxy.nii.gz
    │   └── ...
    │   └── xxz.nii.gz
    └── train_B
    │   ├── 123.nii.gz
    │   ├── nsdf3.nii
    │   └── ...
    │   └── asd932_.nii.gz
    └── val_A
    └── val_B
    └── test_A
    └── test_B

    .. tip:: To quickly iterate over the subjects without loading the images,
        use :meth:`dry_iter()`.

    Parameters
    ----------
    root : Path or str, Optional
        Data root directory. Where to save/load the data.
        Default = ``DATA_ROOT``.
    dataset_name : str, Optional
        The name of the dataset. Default = ``MRI3T27T``.
    domain_a : str, Optional
        The name of the source domain. Default = ``"3T_MPR"``.
    domain_b : str, Optional
        The name of the destination domain. Default = ``"7T_MPR"``.
    transform : Optional[Callable]
        A function/transform that takes in an PIL image and returns a
        transformed version, e.g, ``torchvision.transforms.RandomCrop``.
    target_transform : Callable[Optional]
        Optional function/transform that takes in a target and returns a
        transformed version.
    return_paths : bool
        If True, calling the dataset returns `(sample, target, /path/to/sample,
        /path/to/target)` instead of returning `(sample, target)`.
    load_getitem : bool, Optional
        Load all subject images before returning it in
        :meth:`__getitem__`. Set it to ``False`` if some of the images will
        not be needed during training. Default = ``True``.
    """

    def __init__(
            self,
            root: PathType = DATA_ROOT,
            dataset_name: str = "MRI3T27T",
            domain_a: str = "3T_MPR",
            domain_b: str = "7T_MPR",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[Path], bool]] = None,
            return_paths: bool = False,
            max_class_size: int = sys.maxsize,
            max_dataset_size: int = sys.maxsize,
            load_getitem: bool = False,
            stage: str = "train",
            add_sampling_map: bool = False,
            patch_size: SpatialShapeType = (96, 96, 1),
    ) -> None:
        super().__init__(
            loader=mri_image_loader,
            root=root,
            dataset_name=dataset_name,
            domain_a=domain_a,
            domain_b=domain_b,
            transform=transform,
            target_transform=target_transform,
            extensions=MRI_EXTENSIONS if is_valid_file is None else None,
            is_valid_file=is_valid_file,
            return_paths=return_paths,
            max_class_size=max_class_size,
            max_dataset_size=max_dataset_size,
            stage=stage,
        )
        self.load_getitem = load_getitem
        self.add_sampling_map = add_sampling_map
        self.patch_size = patch_size

    def __getitem__(self, idx: int) -> Tuple[tio.Subject, tio.Subject]:
        """
        Parameters
        ----------
        idx : int
            A (random) integer for data intexing.

        Returns
        -------
        _: Dict[str, Tuple[Any,...]]
        {
            domain_a: (sample_a, target_a),
            domain_b: (sample_b, target_b),
        }, where target_{a, b} is class_index of the respective target class.
        Or, if ``self.return_paths = True``, returns
        {domain_a: (sample_a, target_a, path_a),
         domain_b: (sample_b, target_b, path_b)}.
        """
        try:
            idx = int(idx)
        except (RuntimeError, TypeError) as idx_not_int:
            message = (f'Index "{idx}" must be int or compatible dtype,'
                       f' but an object of type "{type(idx)}" was passed.')
            raise ValueError(message) from idx_not_int

        # Make sure indexes are within A and B ranges
        # sample_a, sample_b = self.loader(path_a), self.loader(path_b)
        sample_a, target_a = self._subjects_a[idx % self.size_a]
        sample_b, target_b = self._subjects_b[idx % self.size_b]
        if self.add_sampling_map:
            sample_a = self.get_sampling_map(sample_a,
                                             patch_size=self.patch_size)
            sample_b = self.get_sampling_map(sample_b,
                                             patch_size=self.patch_size,
                                             offset=10)

        if self.load_getitem:
            sample_a.load()
            sample_b.load()

        if self.transform is not None:
            sample_a = self.transform(sample_a)
            sample_b = self.transform(sample_b)

        return sample_a, sample_b

    def get_max_shape(self) -> Tuple[int, int, int]:
        """
        Get max height, width, and depth accross all subjects.

        Returns
        -------
        shapes_tuple : Tuple[int, int, int]
            Max height, width and depth across all subjects.
        """
        shapes_a = [
            image.spatial_shape for subject in self._subjects_a
            for image in self.loader(subject[0]).get_images()
        ]

        shapes_b = [
            image.spatial_shape for subject in self._subjects_b
            for image in self.loader(subject[0]).get_images()
        ]
        shapes = np.array(shapes_a + shapes_b)
        shapes_tuple = tuple(map(int, shapes.max(axis=0).tolist()))
        return cast(Tuple[int, int, int], shapes_tuple)

    @staticmethod
    def get_sampling_map(
        subject: tio.Subject,
        image_reference: str = 'mri',
        sampling_map_reference: str = 'sampling_map',
        patch_size: SpatialShapeType = (96, 96, 1),
        offset: int = 0,
    ) -> tio.Subject:
        """
        Add sampling map to subject.

        Parameters
        ----------
        subject : tio.Subject
            A tio.Subject instance.
        image_reference : str, Optional
            Name of the image to base the sampling map. Default = ``"mri"```.
        sampling_map_reference : str, Optional
            Name of the sampling map. Default = ``"sampling_map"```.
        patch_size : Tuple[int, int, int], Optional
            Size of the patch. Default = ``(96, 96, 1)``.

        Returns
        -------
        subject : tio.Subject
            A tio.Subject instance with added sampling map.
        """
        probabilities = create_probability_map(subject,
                                               patch_size,
                                               offset=offset)
        sampling_map = tio.Image(tensor=probabilities,
                                 affine=subject[image_reference].affine,
                                 type=tio.SAMPLING_MAP)
        subject.add_image(sampling_map, sampling_map_reference)

        return subject


class MRISliceUnpairedDataset(UnpairedDataset):
    """
    This dataset class can load unpaired/unaligned 3D medical images.
    It is based on torch.utils.data.Dataset,
    torchvision.datasets.VisionDataset, and torchio.SubjectsDataset.

    It requires two directories to host training images from domain `A`
    '/path/to/data/train_A' and from domain `B` 'path/to/data/train_B',
    respectively.
    Similarly, you need to prepare two directories '/path/to/data/test_A' and
    '/path/to/data/test_B' during test time.


    It defaults to a generic MIR folder dataset where the images are arranged
    in this way by default:

    root/
    ├── train_A
    │   ├── xxx.nii
    │   ├── xxy.nii.gz
    │   └── ...
    │   └── xxz.nii.gz
    └── train_B
    │   ├── 123.nii.gz
    │   ├── nsdf3.nii
    │   └── ...
    │   └── asd932_.nii.gz
    └── val_A
    └── val_B
    └── test_A
    └── test_B

    .. tip:: To quickly iterate over the subjects without loading the images,
        use :meth:`dry_iter()`.

    Parameters
    ----------
    root : Path or str, Optional
        Data root directory. Where to save/load the data.
        Default = ``DATA_ROOT``.
    dataset_name : str, Optional
        The name of the dataset. Default = ``MRI3T27T``.
    domain_a : str, Optional
        The name of the source domain. Default = ``"3T_MPR"``.
    domain_b : str, Optional
        The name of the destination domain. Default = ``"7T_MPR"``.
    transform : Optional[Callable]
        A function/transform that takes in an PIL image and returns a
        transformed version, e.g, ``torchvision.transforms.RandomCrop``.
    target_transform : Callable[Optional]
        Optional function/transform that takes in a target and returns a
        transformed version.
    return_paths : bool
        If True, calling the dataset returns `(sample, target, /path/to/sample,
        /path/to/target)` instead of returning `(sample, target)`.
    load_getitem : bool, Optional
        Load all subject images before returning it in
        :meth:`__getitem__`. Set it to ``False`` if some of the images will
        not be needed during training. Default = ``True``.
    """

    def __init__(
            self,
            root: PathType = DATA_ROOT,
            dataset_name: str = "MRI3T27T",
            domain_a: str = "3T_MPR",
            domain_b: str = "7T_MPR",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[Path], bool]] = None,
            return_paths: bool = False,
            max_class_size: int = sys.maxsize,
            max_dataset_size: int = sys.maxsize,
            load_getitem: bool = True,
            stage: str = "train",
            add_sampling_map: bool = False,
            patch_size: SpatialShapeType = (96, 96, 1),
    ) -> None:
        super().__init__(
            loader=mri_image_loader,
            root=root,
            dataset_name=dataset_name,
            domain_a=domain_a,
            domain_b=domain_b,
            transform=transform,
            target_transform=target_transform,
            extensions=MRI_EXTENSIONS if is_valid_file is None else None,
            is_valid_file=is_valid_file,
            return_paths=return_paths,
            max_class_size=max_class_size,
            max_dataset_size=max_dataset_size,
            stage=stage,
        )
        self.load_getitem = load_getitem
        self.add_sampling_map = add_sampling_map
        self.patch_size = patch_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        idx : int
            A (random) integer for data intexing.

        Returns
        -------
        _: Dict[str, Tuple[Any,...]]
        {
            domain_a: (sample_a, target_a),
            domain_b: (sample_b, target_b),
        }, where target_{a, b} is class_index of the respective target class.
        Or, if ``self.return_paths = True``, returns
        {domain_a: (sample_a, target_a, path_a),
         domain_b: (sample_b, target_b, path_b)}.
        """
        try:
            idx = int(idx)
        except (RuntimeError, TypeError) as idx_not_int:
            message = (f'Index "{idx}" must be int or compatible dtype,'
                       f' but an object of type "{type(idx)}" was passed.')
            raise ValueError(message) from idx_not_int

        # Make sure indexes are within A and B ranges
        # sample_a, sample_b = self.loader(path_a), self.loader(path_b)
        sample_a, target_a = self._subjects_a[idx % self.size_a]
        sample_b, target_b = self._subjects_b[idx % self.size_b]

        if self.load_getitem:
            sample_a.load()
            sample_b.load()

        if self.transform is not None:
            sample_a = self.transform(sample_a)
            sample_b = self.transform(sample_b)

        sample_a = self.get_patch(sample_a)
        sample_b = self.get_patch(sample_b)

        return sample_a, sample_b

    def get_patch(
        self,
        subject,
        offset: int = 0,
    ) -> torch.Tensor:

        image = subject['mri']['data']
        data = image[-1]

        for idx, dim in enumerate(self.patch_size):
            if dim == 1:
                empty_dim = idx

        image_size = np.array((1, *subject.spatial_shape))
        slice_idx = image_size[empty_dim + 1] // 2 + offset

        if empty_dim == 0:  # Sagittal
            img_slice = data[slice_idx, :, :]
        elif empty_dim == 1:  # Coronal
            img_slice = data[:, slice_idx, :]
        else:  # Axial
            img_slice = data[:, :, slice_idx]
        return img_slice

    def get_max_shape(self) -> Tuple[int, int, int]:
        """
        Get max height, width, and depth accross all subjects.

        Returns
        -------
        shapes_tuple : Tuple[int, int, int]
            Max height, width and depth across all subjects.
        """
        shapes_a = [
            image.spatial_shape for subject in self._subjects_a
            for image in self.loader(subject[0]).get_images()
        ]

        shapes_b = [
            image.spatial_shape for subject in self._subjects_b
            for image in self.loader(subject[0]).get_images()
        ]
        shapes = np.array(shapes_a + shapes_b)
        shapes_tuple = tuple(map(int, shapes.max(axis=0).tolist()))
        return cast(Tuple[int, int, int], shapes_tuple)
