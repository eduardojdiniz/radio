#!/usr/bin/env python
# coding=utf-8
"""
Data related utilities.
"""

from typing import Dict, List, Optional, Any
import random
import numpy as np
import torchio as tio
from torchio.data.subject import Subject
from torchio.data.image import Image, LabelMap
from torchio.transforms.preprocessing.spatial.to_canonical import ToCanonical
from .datautils import get_subjects_from_batch

__all__ = ["plot_batch", "plot_subjects", "plot_dataset"]


def import_mpl_plt():
    try:
        import matplotlib as mpl  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as error:
        raise ImportError('Install matplotlib for plotting support') from error
    plt.rcParams["savefig.bbox"] = "tight"
    return mpl, plt


def rotate(image, radiological=True, n=-1):
    # Rotate for visualization purposes
    image = np.rot90(image, n)
    if radiological:
        image = np.fliplr(image)
    return image


def color_labels(arrays, cmap_dict):
    results = []
    for array in arrays:
        shape_i, shape_j = array.shape
        rgb = np.zeros((shape_i, shape_j, 3), dtype=np.uint8)
        for label, color in cmap_dict.items():
            if isinstance(color, str):
                mpl, _ = import_mpl_plt()
                color = mpl.colors.to_rgb(color)
                color = [255 * n for n in color]
            rgb[array == label] = color
        results.append(rgb)
    return results


def plot_subjects(
    subjects: List[Subject],
    num_samples: int = 5,
    random_samples: bool = True,
    modalities: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
) -> None:
    """plot images and labels from a batch of images"""
    # Create subjects dataset from list of subjects
    num_subjects = len(subjects)
    if random_samples:
        samples_idx = random.sample(
            range(0, num_subjects),
            min(num_samples, num_subjects),
        )
    else:
        samples_idx = list(range(0, min(num_samples, num_subjects)))

    dataset = tio.SubjectsDataset(subjects)
    # Keep only samples_idx subjects in the dataset
    dataset._subjects = [
        subject for idx, subject in enumerate(dataset._subjects)
        if idx in samples_idx
    ]

    plot_dataset(dataset,
                 modalities=modalities,
                 labels=labels,
                 exclude_keys=exclude_keys)


def plot_batch(
    batch: Dict[str, Any],
    num_samples: int = 5,
    random_samples: bool = True,
    modalities: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
) -> None:
    """plot images and labels from a batch of images"""
    # Parse exclude_keys
    exclude_keys = exclude_keys if exclude_keys else ['sampling_map']
    # Create subjects dataset from batch
    batch_size = len(batch)
    if random_samples:
        samples_idx = random.sample(
            range(0, batch_size),
            min(num_samples, batch_size),
        )
    else:
        samples_idx = list(range(0, min(num_samples, batch_size)))

    subjects = get_subjects_from_batch(batch)
    dataset = tio.SubjectsDataset(subjects)

    # Keep only samples_idx subjects in the dataset
    dataset._subjects = [
        subject for idx, subject in enumerate(dataset._subjects)
        if idx in samples_idx
    ]
    plot_dataset(dataset,
                 modalities=modalities,
                 labels=labels,
                 exclude_keys=exclude_keys)


def plot_dataset(
    dataset: tio.SubjectsDataset,
    modalities: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    exclude_keys: Optional[List[str]] = None,
) -> None:
    """plot images and labels from a dataset of subjects"""

    # Parse modalities, labels and exclude_keys
    exclude_keys = exclude_keys if exclude_keys else ['sampling_map']
    # # Assumes all subjects hold the same tio.IMAGE's
    modalities_in_subj = list(
        dataset[0].get_images_dict(intensity_only=True).keys())
    labels_in_subj = list(dataset[0].get_images_dict(
        intensity_only=False, exclude=modalities_in_subj).keys())
    modalities_in_subj = [
        modality for modality in modalities_in_subj
        if modality not in exclude_keys
    ]
    labels_in_subj = [
        label for label in labels_in_subj if label not in exclude_keys
    ]
    modalities = modalities if modalities else modalities_in_subj
    labels = labels if labels else labels_in_subj

    # Filter images from dataset
    filtered_subjects = []
    for subject in dataset:
        for image_name in subject.get_images_names():
            if (image_name not in modalities) and (image_name not in labels):
                subject.remove_image(image_name)
        filtered_subjects.append(subject)

    # Plot subjects
    for row_idx, subject in enumerate(filtered_subjects):
        print(f"Subject: {row_idx + 1}")
        plot_subject(subject)
        print("\n")


def plot_subject(
    subject: Subject,
    cmap_dict=None,
    show=True,
    output_path=None,
    figsize=None,
    clear_axes=True,
    **kwargs,
):
    _, plt = import_mpl_plt()
    num_images = len(subject)
    many_images = num_images > 2
    subplots_kwargs = {'figsize': figsize}
    try:
        if clear_axes:
            subject.check_consistent_spatial_shape()
            subplots_kwargs['sharex'] = 'row' if many_images else 'col'
            subplots_kwargs['sharey'] = 'row' if many_images else 'col'
    except RuntimeError:  # different shapes in subject
        pass

    if subject.is_2d():
        args = (1, num_images) if many_images else (num_images, 1)
        fig, axes = plt.subplots(*args, **subplots_kwargs)
        # The array of axes must be 2D so that it can be indexed correctly
        # within the plot_slice() function
        axes = axes.T if many_images else np.reshape(axes, (-1, 1))
    else:
        args = (3, num_images) if many_images else (num_images, 3)
        fig, axes = plt.subplots(*args, **subplots_kwargs)
        # The array of axes must be 2D so that it can be indexed correctly
        # within the plot_volume() function
        axes = axes.T if many_images else np.reshape(axes, (-1, 3))
        axes_names = ('sagittal', 'coronal', 'axial')

    iterable = enumerate(subject.get_images_dict(intensity_only=False).items())
    for image_index, (name, image) in iterable:
        image_axes = axes[image_index]
        cmap = None
        if cmap_dict is not None and name in cmap_dict:
            cmap = cmap_dict[name]
        last_row = image_index == len(axes) - 1
        if subject.is_2d():
            axis_name = plot_slice(
                image,
                axes=image_axes,
                show=False,
                cmap=cmap,
                xlabels=last_row,
                **kwargs,
            )
            for axis, axis_name in zip(image_axes, axis_name):
                axis.set_title(f'{name} ({axis_name})')
        else:
            plot_volume(
                image,
                axes=image_axes,
                show=False,
                cmap=cmap,
                xlabels=last_row,
                **kwargs,
            )
            for axis, axis_name in zip(image_axes, axes_names):
                axis.set_title(f'{name} ({axis_name})')
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path)
    if show:
        plt.show()


def plot_volume(
        image: Image,
        radiological=True,
        channel=-1,  # default to foreground for binary maps
        axes=None,
        cmap=None,
        output_path=None,
        show=True,
        xlabels=True,
        percentiles=(0.5, 99.5),
        figsize=None,
        reorient=True,
        indices=None,
):
    _, plt = import_mpl_plt()
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    sag_axis, cor_axis, axi_axis = axes

    if reorient:
        image = ToCanonical()(image)
    data = image.data[channel]
    if indices is None:
        indices = np.array(data.shape) // 2
    i, j, k = indices
    slice_x = rotate(data[i, :, :], radiological=radiological)
    slice_y = rotate(data[:, j, :], radiological=radiological)
    slice_z = rotate(data[:, :, k], radiological=radiological)
    kwargs = {}
    is_label = isinstance(image, LabelMap)
    if isinstance(cmap, dict):
        slices = slice_x, slice_y, slice_z
        slice_x, slice_y, slice_z = color_labels(slices, cmap)
    else:
        if cmap is None:
            cmap = 'cubehelix' if is_label else 'gray'
        kwargs['cmap'] = cmap
    if is_label:
        kwargs['interpolation'] = 'none'

    spacing_r, spacing_a, spacing_s = image.spacing
    kwargs['origin'] = 'lower'

    if percentiles is not None and not is_label:
        percentile_1, percentile_2 = np.percentile(data, percentiles)
        kwargs['vmin'] = percentile_1
        kwargs['vmax'] = percentile_2

    sag_aspect = spacing_s / spacing_a
    sag_axis.imshow(slice_x, aspect=sag_aspect, **kwargs)
    if xlabels:
        sag_axis.set_xlabel('A')
    sag_axis.set_ylabel('S')
    sag_axis.invert_xaxis()
    sag_axis.set_title('Sagittal')

    cor_aspect = spacing_s / spacing_r
    cor_axis.imshow(slice_y, aspect=cor_aspect, **kwargs)
    if xlabels:
        cor_axis.set_xlabel('R')
    cor_axis.set_ylabel('S')
    cor_axis.invert_xaxis()
    cor_axis.set_title('Coronal')

    axi_aspect = spacing_a / spacing_r
    axi_axis.imshow(slice_z, aspect=axi_aspect, **kwargs)
    if xlabels:
        axi_axis.set_xlabel('R')
    axi_axis.set_ylabel('A')
    axi_axis.invert_xaxis()
    axi_axis.set_title('Axial')

    plt.tight_layout()
    if output_path is not None and fig is not None:
        fig.savefig(output_path)
    if show:
        plt.show()


def plot_slice(
        image: Image,
        radiological=True,
        channel=-1,  # default to foreground for binary maps
        axes=None,
        cmap=None,
        output_path=None,
        show=True,
        xlabels=True,
        percentiles=(0.5, 99.5),
        figsize=None,
        reorient=True,
) -> str:
    _, plt = import_mpl_plt()
    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 1, figsize=figsize)
    axis = axes[0]
    if reorient:
        image = ToCanonical()(image)
    data = image.data[channel]

    for idx, dim in enumerate(data.shape):
        if dim == 1:
            empty_dim = idx
    spacing_r, spacing_a, spacing_s = image.spacing

    if empty_dim == 0:
        img_slice = rotate(data[0, :, :], radiological=radiological)
        axis_aspect = spacing_s / spacing_a
        x_label = 'A'
        y_label = 'S'
        title = 'Sagittal'
    elif empty_dim == 1:
        img_slice = rotate(data[:, 0, :], radiological=radiological)
        axis_aspect = spacing_s / spacing_r
        x_label = 'R'
        y_label = 'S'
        title = 'Coronal'
    else:
        img_slice = rotate(data[:, :, 0], radiological=radiological)
        axis_aspect = spacing_a / spacing_r
        x_label = 'R'
        y_label = 'A'
        title = 'Axial'

    kwargs = {}
    is_label = isinstance(image, LabelMap)
    if isinstance(cmap, dict):
        img_slice = color_labels(img_slice, cmap)
    else:
        if cmap is None:
            cmap = 'cubehelix' if is_label else 'gray'
        kwargs['cmap'] = cmap
    if is_label:
        kwargs['interpolation'] = 'none'

    kwargs['origin'] = 'lower'

    if percentiles is not None and not is_label:
        percentile_1, percentile_2 = np.percentile(data, percentiles)
        kwargs['vmin'] = percentile_1
        kwargs['vmax'] = percentile_2

    axis.imshow(img_slice, aspect=axis_aspect, **kwargs)
    if xlabels:
        axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.invert_xaxis()
    axis.set_title(title)

    plt.tight_layout()
    if output_path is not None and fig is not None:
        fig.savefig(output_path)
    if show:
        plt.show()

    return title
