#!/usr/bin/env python
# coding=utf-8
"""
Based on BaseDataModule for managing data. A vision datamodule that is
shareable, reusable class that encapsulates all the steps needed to process
data, i.e., decoupling datasets from models to allow building dataset-agnostic
models. They also allow you to share a full dataset without explaining how to
download, split, transform, and process the data.
"""

from typing import List, Optional, Union, cast
import copy
from torch.utils.data import DataLoader, Dataset
import torchio as tio  # type: ignore
import torch
from ..datatypes import SpatialShapeType

__all__ = ["PatchBasedInference"]


class PatchBasedInference:
    """
    Dense Patch-based inference.

    Typical Workflow
    ----------------
    in_dataset: tio.SubjectsDataset
    model: torch.nn.Module
    modalities: List[str]
    out_dataset: tio.SubjectsDataset

    inferencemodule = PatchBasedInference(
        patch_size,
        patch_overlap,
        padding_mode,
        overlap_mode,
    )

    out_dataset = inferencemodule(in_dataset, model, modalities)

    Parameters
    ----------
    patch_size : int or (int, int, int)
        Tuple of integers ``(w, h, d)`` to generate patches of size ``w x h x
        d``. If a single number ``n`` is provided, ``w = h = d = n``.
    patch_overlap : int or (int, int, int), optional
        Tuple of even integers ``(w_o, h_o, d_o)`` specifying the overlap
        between patches for dense inference. If a single number ``n`` is
        provided, ``w_o = h_o = d_o = n``. Default = ``(0, 0, 0)``.
    patch_batch_size : int, optional
        How many patches per batch to load. Default = ``32``.
    padding_mode : str or float or None, optional
        If ``None``, the volume will not be padded before sampling and patches
        at the border will not be cropped by the aggregator. Otherwise, the
        volume will be padded with ``w_o/2, h_o/2, d_o/2`` on each side before
        sampling and it will cropped by the aggregator to its original size.
        Possible padding modes: ``('empty', 'edge', 'wrap', 'constant',
        'linear_ramp', 'maximum', 'mean', 'median', 'minimum', 'reflect',
        'symmetric')``. If it is a number,  the mode will be set to
        ``'constant'``. Default = ``None``.
    overlap_mode : str, optional
        If ``'crop'``, the overlapping predictions will be cropped. If
        ``'average'``, the predictions in the overlapping areas will be
        averaged with equal weights. Default = ``'crop'``.
    num_workers : int, optional
        How many subprocesses to use for data loading. ``0`` means that the
        data will be loaded in the main process. Default: ``0``.
    pin_memory : bool, optional
        If ``True``, the data loader will copy Tensors into CUDA pinned memory
        before returning them. Default = ``True``.
    verbose : bool, optional
        If ``True``, print debugging messages. Default = ``False``.
    """

    def __init__(
        self,
        patch_size: SpatialShapeType = 96,
        patch_overlap: SpatialShapeType = (0, 0, 0),
        patch_batch_size: int = 32,
        padding_mode: Union[str, float, None] = None,
        overlap_mode: str = 'crop',
        num_workers: int = 0,
        pin_memory: bool = True,
        verbose: bool = False,
    ) -> None:
        # Init Dataloader Parameters
        self.patch_batch_size = patch_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Init Sampler Parameters
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.padding_mode = padding_mode

        # Init Aggregator Parameters
        self.overlap_mode = overlap_mode

        self.verbose = verbose

    def _inference(
        self,
        subject: tio.Subject,
        model: torch.nn.Module,
        modality: str = None,
    ) -> torch.Tensor:
        modality = modality if modality else 't1'
        sampler = tio.data.GridSampler(
            subject=subject,
            patch_size=self.patch_size,
            patch_overlap=self.patch_overlap,
            padding_mode=self.padding_mode,
        )
        aggregator = tio.data.GridAggregator(
            sampler,
            overlap_mode=self.overlap_mode,
        )
        patch_loader = DataLoader(
            cast(Dataset, sampler),
            batch_size=self.patch_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        model.eval()
        with torch.no_grad():
            for patches_batch in patch_loader:
                insor = patches_batch[modality][tio.DATA]
                locations = patches_batch[tio.LOCATION]
                outsor = model(insor)
                aggregator.add_batch(outsor, locations)
        return aggregator.get_output_tensor()

    def __call__(
        self,
        in_dataset: tio.SubjectsDataset,
        model: torch.nn.Module,
        modalities: Optional[List[str]] = None,
    ) -> tio.SubjectsDataset:
        """
        Parameters
        ----------
        in_dataset : tio.SubjectsDataset
            Test subjects dataset.
        model : torch.nn.Module
            Model to use for inference.
        modalities : List[str], optional
            In which modalilities to perform inference. Default = ``['t1']``.

        Returns
        -------
        out_dataset : tio.SubjectsDataset
            Test subjects dataset with inference results on given modalities.
        """
        modalities = modalities if modalities is not None else ['t1']
        subjects_list = []
        for _, subject in enumerate(in_dataset):
            # Create a copy of subject and remove images
            subject_copy = copy.copy(subject)
            for image_name in subject_copy.get_images_names():
                subject_copy.remove_image(image_name)
            # Inference on each of the required modalities
            for modality in modalities:
                outsor = self._inference(subject, model, modality)
                subject_copy.add_image(outsor, image_name=modality)
            subjects_list.append(subject_copy)
        out_dataset = tio.SubjectsDataset(subjects_list)
        return out_dataset
