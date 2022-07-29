#!/usr/bin/env python
# coding=utf-8
"""
Data related utilities.
"""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, TypeVar, Union
import hashlib
import os.path
from os.path import join as pjoin
import traceback
import random
from pathlib import Path
from scipy import stats  # type: ignore
import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import torch
import torchio as tio
import torchvision.transforms as T  # type: ignore
from PIL import Image
from tqdm.auto import tqdm  # type: ignore
from torchio.data import ScalarImage, LabelMap, Subject
from .datatypes import Tensors, SeqSeqTensor, GenericTrainType
from . import constants
from ..settings import PathType

plt.rcParams["savefig.bbox"] = "tight"

Var = TypeVar("Var")
DirCollectionType = GenericTrainType[PathType]

__all__ = [
    "create_probability_map",
    "get_subjects_dataset",
    "plot_histogram",
    "compute_histogram",
    "get_historgram_standardization_transform",
    "get_intensity_normalization_transform",
    "get_first_batch",
    "default_image_loader",
    "mri_image_loader",
    "denormalize",
    "plot",
    "plot_batch",
    "load_standard_test_imgs",
    "check_integrity",
]


def create_probability_map(
    subject,
    patch_size,
    slice_range: Tuple[int, int] = None,
):

    probabilities = torch.zeros(1, *subject.spatial_shape)

    for idx, dim in enumerate(patch_size):
        if dim == 1:
            empty_dim = idx

    image_size = np.array((1, *subject.spatial_shape))
    if slice_range is None:
        leftmost = image_size[empty_dim + 1] // 2 - 5
        rightmost = image_size[empty_dim + 1] // 2 + 5
    else:
        leftmost, rightmost = slice_range

    probability_voxel_in = 1
    probability_voxel_out = 0

    if empty_dim == 0:  # Sagittal
        probabilities[:, :leftmost, :, :] = probability_voxel_out
        probabilities[:, leftmost:rightmost, :, :] = probability_voxel_in
        probabilities[:, rightmost:, :, :] = probability_voxel_out
    elif empty_dim == 1:  # Coronal
        probabilities[:, :, :leftmost, :] = probability_voxel_out
        probabilities[:, :, leftmost:rightmost, :] = probability_voxel_in
        probabilities[:, :, rightmost:, :] = probability_voxel_out
    else:  # Axial
        probabilities[:, :, :, :leftmost] = probability_voxel_out
        probabilities[:, :, :, leftmost:rightmost] = probability_voxel_in
        probabilities[:, :, :, rightmost:] = probability_voxel_out
    return probabilities


def flatten_dir_collection_type(
        dir_collection: DirCollectionType) -> List[Path]:
    """
    Flatten a directory collection, i.e., returns a collection of type
    List[Path].

    Parameters
    ----------
    dir_collection : DirCollectionType
        A collection of directories where the data are stored.

    Returns
    -------
    _ : List[Path]
    """

    def _handle_is_mapping(dir_collection):
        sequence = []
        for _, dset_dir in dir_collection.items():
            if isinstance(dset_dir, Mapping):
                sequence.extend(_handle_is_mapping(dset_dir))
            elif isinstance(dset_dir, List):
                sequence.extend(_handle_is_sequence(dset_dir))
            else:
                sequence.append(Path(dset_dir))
        return sequence

    def _handle_is_sequence(dir_collection):
        sequence = []
        for dset_dir in dir_collection:
            if isinstance(dset_dir, Mapping):
                sequence.extend(_handle_is_mapping(dset_dir))
            elif isinstance(dset_dir, List):
                sequence.extend(_handle_is_sequence(dset_dir))
            else:
                sequence.append(Path(dset_dir))
        return sequence

    if isinstance(dir_collection, Mapping):
        return _handle_is_mapping(dir_collection)
    if isinstance(dir_collection, List):
        if len(dir_collection) == 1:
            if isinstance(dir_collection[0], Mapping):
                return _handle_is_mapping(dir_collection[0])
            if isinstance(dir_collection[0], List):
                if len(dir_collection[0]) == 1:
                    return [Path(dir_collection[0][0])]
                return _handle_is_sequence(dir_collection[0])
            return [Path(dir_collection[0])]
        return _handle_is_sequence(dir_collection)
    return [Path(dir_collection)]


def get_subjects_dataset(
    dir_collection: DirCollectionType,
) -> Tuple[tio.SubjectsDataset, tio.SubjectsDataset, tio.SubjectsDataset]:
    """
    Create train/test/val tio.SubjectsDataset from a directory collection.

    Parameters
    ----------
    dir_collection : DirCollectionType
        A collection of directories where the data are stored.

    Returns
    -------
    _ : Tuple[tio.SubjectsDataset, tio.SubjectsDataset, tio.SubjectsDataset]
        Subjects Dataset for respectively, train, test, and validation images.
    """
    subjects: Dict[str, List[tio.Subject]] = {
        "train": [],
        "test": [],
        "val": [],
    }

    dir_list = flatten_dir_collection_type(dir_collection)

    for directory in dir_list:
        for fold in ["train", "test", "val"]:
            img_dir = directory / fold
            img_paths = [
                p.resolve() for p in sorted(img_dir.glob("*"))
                if p.suffixes in [['.nii', '.gz'], ['.nii']]
            ]
            for img_path in img_paths:
                subject = tio.Subject(mri=tio.ScalarImage(img_path))
                subjects[fold].append(subject)
    train_dataset = tio.SubjectsDataset(subjects["train"])
    test_dataset = tio.SubjectsDataset(subjects["test"])
    val_dataset = tio.SubjectsDataset(subjects["val"])
    return train_dataset, test_dataset, val_dataset


def plot_histogram(
    axis: matplotlib.axes.Axes,
    tensor: torch.Tensor,
    num_positions: int = 100,
    label: str = None,
    alpha: float = 0.05,
    color: str = 'black',
) -> None:
    """
    Plot Histogram.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Matplotlib Axis object.
    tensor : torch.Tensor
        Tensor with data.
    num_positions : int, Optional
        Number of positions on x-axis. Default = ``100``.
    label : str, Optional
        Plot label. Default = ``None``.
    alpha : float, Optional
        Plot alpha value. Default = ``0.05``.
    color : str, Optional
        Plot color. Default = ``"black"``.
    """
    values = tensor.numpy().ravel()
    kernel = stats.gaussian_kde(values)
    positions = np.linspace(values.min(), values.max(), num=num_positions)
    histogram = kernel(positions)
    kwargs = dict(linewidth=1, color=color, alpha=alpha)
    if label is not None:
        kwargs['label'] = label
    axis.plot(positions, histogram, **kwargs)


def compute_histogram(
    image_paths: List[Path] = None,
    dataset: tio.SubjectsDataset = None,
    transform: tio.Transform = None,
    xlim: Tuple[float, float] = (-100, 2000),
    ylim: Tuple[float, float] = (0, 0.004),
    title: str = None,
    xlabel: str = 'Intensity',
    name2colordict: Dict[str, str] = None,
) -> matplotlib.axes.Axes:
    """
    Compute and plot histogram.

    Parameters
    ----------
    image_paths : List[Path], Optional
        List with image paths. An alternative to providing ``dataset``. At
        least one of ``dataset`` or ``image_paths`` must be not ``None``.
        Default = ``None``.
    dataset : tio.SubjectsDataset, Optional
        Subjects Dataset. An alternative to providing ``image_paths``. If a
        dataset is provided, then it will be used instead of the image paths.
        Default = ``None``.
    transform : tio.Transform, Optional
        Transform to apply to samples prior to computing histogram.
        Default = ``None``.
    xlim : Tuple[float, float], Optional
        x-axis limits. Default = ``(-100, 2000)``.
    ylim : Tuple[float, float], Optional
        y-axis limits. Default = ``(0, 0.004)``.
    title : str, Optional
        Histogram title. Default = ``None``.
    xlabel : str, Optional
        x-axis label. Default = ``"Intensity"``.
    name2colordict : Dict[str, str], Optional
        Substring of image paths to be matched as keys and corresponding plot
        color as value. Default = ``{3T_-_T1w_MPR: blue, 7T_-_T1w_MPR: red}``.

    Returns
    -------
    _ : matplotlib.axes.Axes
        Returns the plot axis.
    """
    both_none = image_paths is None and dataset is None
    both_something = image_paths is not None and dataset is not None
    if both_none or both_something:
        both_none_or_something_msg = (
            "Both 'image_paths' and 'dataset' cannot be None ",
            "or not None at the same time",
        )
        raise ValueError(both_none_or_something_msg)

    name2colordict = name2colordict if name2colordict else {
        '3T_-_T1w_MPR': 'blue',
        '7T_-_T1w_MPR': 'red'
    }
    _, axis = plt.subplots(dpi=100)

    iterable = dataset if dataset else image_paths

    for sample in tqdm(iterable):
        sample = sample if dataset else tio.ScalarImage({'mri': sample})
        if transform:
            sample = transform(sample)
        tensor = sample.mri.data
        path = sample.mri.path
        for name, color in name2colordict.items():
            plot_color = color if name in path.name else 'black'
        plot_histogram(axis, tensor, color=plot_color)

    # axis.set_xlim(*xlim)
    # axis.set_ylim(*ylim)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.grid()

    return axis


def get_historgram_standardization_transform(
    image_paths: List[Path],
    output_histogram_landmarks_path: str = 'histogram_landmarks.npy',
) -> tio.Transform:
    """
    Trains a tio.HistogramStandardization transform.

    Parameters
    ----------
    image_paths : List[Path]
        List with image paths.
    output_histogram_landmarks_path : Path, Optional
        Path to file to store the histogram landmarks Numpy array.
        Default = ``'histogram_landmarks.npy'``.

    Returns
    -------
     : tio.Transform
       Trained tio.HistogramStandardization transform.
    """
    landmarks = tio.HistogramStandardization.train(
        image_paths, output_path=output_histogram_landmarks_path)
    landmarks_dict = {'mri': landmarks}
    return tio.HistogramStandardization(landmarks_dict)


def get_intensity_normalization_transform(
    image_paths: List[Path],
    output_histogram_landmarks_path: str = 'histogram_landmarks.npy',
    include_histogram_norm: bool = True,
    include_znorm: bool = True,
) -> tio.Transform:
    """
    Get a transform to normalize image intensities by first performing
    histogram standardization followed by Z-normalization.
    This transform ensures that intensities are similarly distributed and
    within similar ranges. Mean and variance for the Z-standardization are
    computed using only foreground values. Foreground is approximated as all
    values aboute the mean.

    Parameters
    ----------
    image_paths : List[Path]
        List with image paths.
    output_histogram_landmarks_path : Path, Optional
        Path to file to store the histogram landmarks Numpy array.
        Default = ``'histogram_landmarks.npy'``.

    Returns
    -------
     : tio.Transform
       Trained tio.HistogramStandardization transform followed by
       tio.ZNormalization transform.

    """
    both_false = include_znorm is False and include_histogram_norm is False
    if both_false:
        raise ValueError("At least one type of normalization must be set.")
    histogram_transform = get_historgram_standardization_transform(
        image_paths, output_histogram_landmarks_path)
    znorm_transform = tio.ZNormalization(
        masking_method=tio.ZNormalization.mean)

    transforms = []
    if include_histogram_norm:
        transforms.append(histogram_transform)
    if include_znorm:
        transforms.append(znorm_transform)

    return tio.Compose(transforms)


def get_first_batch(loader: Iterable,
                    default: Optional[Var] = None) -> Optional[Var]:
    """
    Returns the first item in the given iterable or `default` if empty,
    meaningful mostly with 'for' expressions.

    Parameters
    ----------
    loader : Iterable
        Dataloader to get the batch from.
    default: Any, optional
        Returned if ``loader`` is empty. Default = ``None``.

    Returns
    -------
    batch : Any
       First batch from the dataloader. Default = ``None``.

    """
    for batch in loader:
        return batch
    return default


def plot_batch(
    batch: Dict,
    num_samples: int = 5,
    intensities: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
) -> None:
    """plot images and labels from a batch of images"""
    # Create subjects dataset from batch
    batch_size = len(batch)
    samples_idx = random.sample(
        range(0, batch_size),
        min(num_samples, batch_size),
    )
    dataset = tio.SubjectsDataset.from_batch(batch)

    # Keep only samples_idx subjects in the dataset
    dataset._subjects = [
        subject for idx, subject in enumerate(dataset._subjects)
        if idx in samples_idx
    ]

    # Parse intensities, labels and exclude_keys
    exclude_keys = exclude_keys if exclude_keys else []
    # # Assumes all subjects hold the same tio.IMAGE's
    intensities_in_subj = list(
        dataset[0].get_images_dict(intensity_only=True).keys())
    labels_in_subj = list(dataset[0].get_images_dict(
        intensity_only=False, exclude=intensities_in_subj).keys())
    intensities_in_subj = [
        intensity for intensity in intensities_in_subj
        if intensity not in exclude_keys
    ]
    labels_in_subj = [
        label for label in labels_in_subj if label not in exclude_keys
    ]
    intensities = intensities if intensities else intensities_in_subj
    labels = labels if labels else labels_in_subj

    # Filter images from dataset
    for _, subject in enumerate(dataset):
        for image_name in subject.get_images_names():
            if image_name not in intensities or image_name not in labels:
                subject.remove_image(image_name)

    # Plot subjects
    for row_idx, subject in enumerate(dataset):
        print(f"Subject: {row_idx}")
        subject.plot()
        print("\n")


def get_batch_images_and_size(batch: Dict) -> Tuple[List[str], int]:
    """Get number of images and images names in a batch.
    Args:
        batch: Dictionary generated by a :class:`torch.utils.data.DataLoader`
        extracting data from a :class:`torchio.SubjectsDataset`.
    Raises:
        RuntimeError: If the batch does not seem to contain any dictionaries
        that seem to represent a :class:`torchio.Image`.
    """
    names = []
    for image_name, image_dict in batch.items():
        if isinstance(
                image_dict, Mapping
        ) and constants.DATA in image_dict:  # assume it is a TorchIO Image
            size = len(image_dict[constants.DATA])
            names.append(image_name)
    if not names:
        raise RuntimeError('The batch does not seem to contain any images')
    return names, size


def get_subjects_from_batch(batch: Dict) -> List:
    """Get list of subjects from collated batch.
    Args:
        batch: Dictionary generated by a :class:`torch.utils.data.DataLoader`
        extracting data from a :class:`torchio.SubjectsDataset`.
    """
    subjects = []
    image_names, batch_size = get_batch_images_and_size(batch)
    for i in range(batch_size):
        subject_dict = {}
        for image_name in image_names:
            image_dict = batch[image_name]
            data = image_dict[constants.DATA][i]
            affine = image_dict[constants.AFFINE][i]
            path = Path(image_dict[constants.PATH][i])
            is_label = image_dict[constants.TYPE][i] == constants.LABEL
            klass = LabelMap if is_label else ScalarImage
            image = klass(tensor=data, affine=affine, filename=path.name)
            subject_dict[image_name] = image
            if 'subj_id' in batch:
                subject_dict['subj_id'] = batch['subj_id'][i]
            if 'scan_id' in batch:
                subject_dict['scan_id'] = batch['scan_id'][i]
            if 'field' in batch:
                subject_dict['field'] = batch['field'][i]
        subject = Subject(subject_dict)
        if constants.HISTORY in batch:
            applied_transforms = batch[constants.HISTORY][i]
            for transform in applied_transforms:
                transform.add_transform_to_subject_history(subject)
        subjects.append(subject)
    return subjects


def default_image_loader(path: Path) -> Image.Image:
    """
    Load image file as RGB PIL Image

    Parameters
    ----------
    path : Path
        Image file path

    Returns
    -------
    return : Image.Image
       RGB PIL Image
    """

    # Open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with path.open(mode="rb") as fid:
        img = Image.open(fid)
        return img.convert("RGB")


def mri_image_loader(path: Union[PathType, List[PathType]],
                     image_name: Union[str, List[str]] = 'mri',
                     **kwargs) -> tio.Subject:
    """
    Load image file as torchio.Subject.

    Parameters
    ----------
    path : (Path or str) or List[(Path or str)]
        Image file path or list of Image file paths.

    image_name : str or List[str], Optional
        Name of the image or list with image names. Default = ``"mri"``.

    kwargs : Dict[str, Any], Optional
        Extra set of metadata to be added to the subjects.

    Returns
    -------
    return : torchio.Subject
       torchio.Subject object.
    """
    if not isinstance(path, List):
        path = [path]

    if not isinstance(image_name, List):
        image_name = [image_name]

    assert len(path) == len(
        image_name
    ), "Number of file paths must be equal to the number of image names."

    subject_dict = {}
    for name, file_path in zip(image_name, path):
        subject_dict.update({name: tio.ScalarImage(str(file_path))})
    for key, value in kwargs.items():
        subject_dict.update({key: value})

    return tio.Subject(subject_dict)


def denormalize(tensor: torch.Tensor,
                mean: Tuple[float, ...] = None,
                std: Tuple[float, ...] = None):
    """
    Undoes mean/standard deviation normalization, zero to one scaling, and
    channel rearrangement for a batch of images.

    Parameters
    ----------
    tensor : torch.Tensor
        A (CHANNELS x HEIGHT x WIDTH) tensor
    mean: Tuple[float, ...]
        A tuple of mean values to be subtracted from each image channel.
    std: Tuple[float, ...]
        A tuple of standard deviation values to be devided from each image
        channel.

    Returns
    ----------
    array : numpy.ndarray[float]
        A (HEIGHT x WIDTH x CHANNELS) numpy array of floats
    """
    if not mean:
        if tensor.shape[0] == 1:
            mean = (-0.5 / 0.5, )
        else:
            mean = (-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5)
    if not std:
        if tensor.shape[0] == 1:
            std = (1 / 0.5, )
        else:
            std = (1 / 0.5, 1 / 0.5, 1 / 0.5)
    inverse_normalize = T.Normalize(mean=mean, std=std)
    denormalized_tensor = (inverse_normalize(tensor) * 255.0).type(torch.uint8)
    array = denormalized_tensor.permute(1, 2, 0).numpy().squeeze()
    return array


def plot(imgs: Tensors,
         baseline_imgs: Tensors = None,
         row_titles: List[str] = None,
         fig_title: str = None,
         **imshow_kwargs) -> None:
    """
    Plot images in a 2D grid.

    Arguments
    ---------
    imgs : Samples
        Collection of images to be plotted. Each element of ``imgs`` holds a
        row of the image grid to be plotted.
    baseline_imgs : Samples, Optional
        Collection of baseline images. If not ``None``, the first column of the
        grid will be filled with the baseline images. ``baseline_imgs`` is
        either a single image, or a collection of images of the same length of
        an element of ``imgs``. Default = ``None``.
    row_titles : List[str], Optional
        List of row titles. If not ``None``, ``len(row_title)`` must be equal
        to ``len(imgs)``. Default = ``None``.
    fig_title : str, Optional
        Figure title.  Default = ``None``.
    """
    # Make a 2d grid even if there's just 1 row
    if isinstance(imgs, SeqSeqTensor):
        local_imgs = imgs
    else:
        local_imgs = SeqSeqTensor(imgs)

    num_rows = len(local_imgs)
    num_cols = len(local_imgs[0])

    if not baseline_imgs:
        with_baseline = False
    else:
        if not isinstance(baseline_imgs, list):
            baseline_imgs = [baseline_imgs for i in range(0, num_rows)]
        else:
            if len(baseline_imgs) == 1:
                baseline_imgs = [baseline_imgs[0] for i in range(0, num_rows)]
            elif len(baseline_imgs) != num_rows:
                msg = (
                    "Number of elements in `baseline_imgs` ",
                    "must match the number of elements in `imgs[0]`",
                )
                raise ValueError(msg)
            if isinstance(baseline_imgs[0], list):
                msg = (
                    "Elements of `baseline_imgs` must be PIL Images ",
                    "or Torch Tensors",
                )
                raise TypeError(msg)
        with_baseline = True
        num_cols += 1  # First column is now the baseline images
    if row_titles:
        if len(row_titles) != num_rows:
            msg = (
                "Number of elements in `row_title` ",
                "must match the number of elements in `imgs`",
            )
            raise ValueError(msg)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [baseline_imgs[row_idx]] + row if with_baseline else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            if isinstance(img, torch.Tensor):
                img = denormalize(img)
            else:
                img = np.asarray(img)
            if len(img.shape) == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(img, **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_baseline:
        plt.sca(axs[0, 0])
        plt.title(label="Baseline images", size=15)

    if row_titles is not None:
        for row_idx in range(num_rows):
            plt.sca(axs[row_idx, 0])
            plt.ylabel(row_titles[row_idx], rotation=0, labelpad=50, size=15)
            plt.tight_layout()

    if fig_title:
        fig.suptitle(t=fig_title, size=16)

    fig.tight_layout()
    return fig


def load_standard_test_imgs(directory: Path = Path().cwd() / "imgs"):
    directory = directory.expanduser()
    test_imgs = []
    names = []
    for root, _, fnames in sorted(os.walk(directory, followlinks=True)):
        for fname in sorted(fnames):
            if is_valid_image(Path(fname)):
                path = pjoin(root, fname)
                test_imgs.extend([Image.open(path)])
                names.append(Path(path).stem)
    return test_imgs, names


def calculate_md5_dir(dirpath: Path,
                      chunk_size: int = 1024 * 1024,
                      verbose: bool = False) -> str:
    md5 = hashlib.md5()
    try:
        for root, _, files in sorted(os.walk(dirpath)):
            for name in files:
                if verbose:
                    print("Hashing", name)
                fpath = Path(root) / name
                with fpath.open(mode="rb") as fid:
                    for chunk in iter(lambda: fid.read(chunk_size), b""):
                        md5.update(chunk)

    except BaseException:
        # Print the stack traceback
        traceback.print_exc()
        return -2

    return md5.hexdigest()


def calculate_md5_file(fpath: Path, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5()
    try:
        with fpath.open(mode="rb") as fid:
            for chunk in iter(lambda: fid.read(chunk_size), b""):
                md5.update(chunk)
    except BaseException:
        # Print the stack traceback
        traceback.print_exc()
        return -2
    return md5.hexdigest()


def check_md5(path: Path, md5: str, **kwargs: Any) -> bool:
    if path.is_dir():
        return md5 == calculate_md5_dir(path, **kwargs)
    return md5 == calculate_md5_file(path, **kwargs)


def check_integrity(path: Path, md5: Optional[str] = None) -> bool:
    if not os.path.exists(path):
        return False
    if md5 is None:
        return True
    return check_md5(path, md5)
