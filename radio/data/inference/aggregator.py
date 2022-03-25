#!/usr/bin/env python
# coding=utf-8
"""
Based on BaseDataModule for managing data. A vision datamodule that is
shareable, reusable class that encapsulates all the steps needed to process
data, i.e., decoupling datasets from models to allow building dataset-agnostic
models. They also allow you to share a full dataset without explaining how to
download, split, transform, and process the data.
"""

from typing import List, Optional, Union, cast, Dict, Any
import copy
from torch.utils.data import DataLoader, Dataset
import torchio as tio  # type: ignore
import torch
from ..datatypes import SpatialShapeType
from ..datautils import get_subjects_from_batch

__all__ = ["PatchBasedInference"]


class PatchBasedInference:
    """
    Dense Patch-based inference.

    Typical Workflow
    ----------------
    in_dataset: tio.SubjectsDataset
    model: torch.nn.Module
    intensities: List[str]
    out_dataset: tio.SubjectsDataset

    inferencemodule = PatchBasedInference(
        patch_size,
        patch_overlap,
        padding_mode,
        overlap_mode,
    )

    out_dataset = inferencemodule(in_dataset, model, intensities)

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
        intensity: str,
    ) -> torch.Tensor:
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
                insor = patches_batch[intensity][tio.DATA]
                locations = patches_batch[tio.LOCATION]
                outsor = model(insor)
                aggregator.add_batch(outsor, locations)
        return aggregator.get_output_tensor()

    def __call__(
        self,
        batch: Dict[str, Any],
        model: torch.nn.Module,
        intensities: Optional[List[str]] = None,
    ) -> List[tio.Subject]:
        """
        Parameters
        ----------
        batch : Dict[str, Any]
            Dataloader batch.
        model : torch.nn.Module
            Model to use for inference.
        intensities : List[str], optional
            In which modalilities to perform inference. Default = ``['T1']``.

        Returns
        -------
        subjects_list : List[tio.Subject]
            List of test subjects with inference results on given intensities.
        """
        intensities = intensities if intensities else ['T1']
        subjects = get_subjects_from_batch(batch)
        subjects_list = []
        for subject in subjects:
            # Create a copy of subject and remove images
            subject_copy = copy.copy(subject)
            for image_name in subject_copy.get_images_names():
                subject_copy.remove_image(image_name)
            # Inference on each of the required intensities
            for intensity in intensities:
                outsor = self._inference(subject, model, intensity)
                subject_copy.add_image(tio.ScalarImage(tensor=outsor),
                                       image_name=intensity)
            subjects_list.append(subject_copy)
        return subjects_list
