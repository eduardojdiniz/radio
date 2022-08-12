#!/usr/bin/env python
# coding=utf-8
"""
Brain Aging Prediction Data Module
"""

from typing import Any, Dict, List, Optional, Tuple, Union, cast
from pathlib import Path
import re
from string import Template
from collections import OrderedDict
import numpy as np
import torchio as tio  # type: ignore
from torch.utils.data import DataLoader
from radio.settings.pathutils import PathType, ensure_exists
from ..cerebrodatamodule import CerebroDataModule
from ..datautils import get_subjects_from_batch
from ..datatypes import SubjPathType, SubjDictType

__all__ = ["BrainAgingPredictionDataModule"]


class BrainAgingPredictionDataModule(CerebroDataModule):
    """
    Brain Aging Prediction Data Module.

    Typical Workflow
    ----------------
    data = BrainAgingPredictionDataModule()
    data.prepare_data() # download
    data.setup(stage) # process and split
    data.teardown(stage) # clean-up

    Parameters
    ----------
    root : Path or str, optional
        Root to GPN's CEREBRO Studies folder.
        Default = ``'/media/cerebro/Studies'``.
    study : str, optional
        Study name. Default = ``'Brain_Aging_Prediction'``.
    subj_dir : str, optional
        Subdirectory where the subjects are located.
        Default = ``'Public/data'``.
    data_dir : str, optional
        Subdirectory where the subjects' data are located.
        Default = ``'step01_structural_processing'``.
    modalities : List[str], optional
        Which modalities to load. Default = ``['T1']``.
    labels : List[str], optional
        Which labels to load. Default = ``[]``.
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
    resample : str, optional
        If an intensity name is provided, resample all images to specified
        intensity. Default = ``None``.
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
    val_split : int or float, optional
        If ``num_folds = 2``, then ``val_split`` specify how the
        train_dataset should be split into train/validation datasets. If
        ``num_folds > 2``, then it is not used. Default = ``0.2``.
    dims : Tuple[int, int, int], optional
        Max spatial dimensions across subjects' images. If ``None``, compute
        dimensions from dataset. Default = ``(160, 192, 160)``.
    seed : int, optional
        When `shuffle` is True, `seed` affects the ordering of the indices,
        which controls the randomness of each fold. It is also use to seed the
        RNG used by RandomSampler to generate random indexes and
        multiprocessing to generate `base_seed` for workers. Pass an int for
        reproducible output across multiple function calls. Default = ``41``.
    verbose : bool, optional
        If ``True``, print debugging messages. Default = ``False``.
    """
    name: str = "brain_aging_prediction"
    intensity2template = {
        "T1": Template('wstrip_m${subj_id}_${scan_id}_T1.nii'),
        "FLAIR": Template('wstrip_mr${subj_id}_${scan_id}_FLAIR.nii'),
    }
    label2template: Dict[str, Template] = {}

    def get_subjects(self, fold: str = "train") -> List[tio.Subject]:
        """
        Get train, test, or val list of TorchIO Subjects.

        Parameters
        ----------
        fold : str, optional
            Identify which type of dataset, ``'train'``, ``'test'``, or
            ``'val'``. Default = ``'train'``.

        Returns
        -------
        _ : List[tio.Subject]
            Train, test or val list of TorchIO Subjects.
        """
        train_subjs, test_subjs, val_subjs = self.get_subjects_dicts(
            modalities=self.modalities, labels=self.labels)
        if fold == "train":
            subjs_dict = train_subjs
        elif fold == "test":
            subjs_dict = test_subjs
        else:
            subjs_dict = val_subjs

        subjects = []
        for _, subject_dict in subjs_dict.items():
            subject = tio.Subject(subject_dict)
            subjects.append(subject)
        return subjects

    def get_subjects_dicts(
        self,
        modalities: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
    ) -> Tuple[SubjDictType, SubjDictType, SubjDictType]:
        """
        Get paths to nii files for train/test/val images and labels.

        Returns
        -------
        _ : {(str, str): Dict}, {(str, str): Dict}, {(str, str): Dict}
            Paths to, respectively, train, test, and val images and labels.
        """

        def _get_dict(
            paths_dict: SubjPathType,
            modalities: List[str],
            labels: List[str],
            train: bool = True,
        ) -> SubjDictType:
            subjects_dict: SubjDictType = OrderedDict()
            for (subj_id, scan_id), path in paths_dict.items():
                subjects_dict[(subj_id, scan_id)] = OrderedDict({
                    "subj_id":
                    subj_id,
                    "scan_id":
                    scan_id
                })
                for modality in modalities:
                    modality_path = path / self.intensity2template[
                        modality].substitute(subj_id=subj_id, scan_id=scan_id)
                    if modality_path.is_file():
                        subjects_dict[(subj_id, scan_id)].update(
                            {modality: tio.ScalarImage(modality_path)})
                    else:
                        subjects_dict.pop((subj_id, scan_id), None)

                if train:
                    for label in labels:
                        label_path = path / self.label2template[
                            label].substitute(subj_id=subj_id, scan_id=scan_id)
                        if label_path.is_file():
                            subjects_dict[(subj_id, scan_id)].update(
                                {label: tio.LabelMap(label_path)})
                        else:
                            subjects_dict.pop((subj_id, scan_id), None)
            return subjects_dict

        modalities = modalities if modalities else ['T1', 'FLAIR']
        labels = labels if labels else []

        for modality in modalities:
            assert modality in self.intensity2template

        for label in labels:
            assert label in self.label2template

        subj_train_paths, subj_test_paths, subj_val_paths = self.get_paths(
            self.root,
            stem=self.data_dir,
            has_train_test_split=self.has_train_test_split,
            has_train_val_split=self.has_train_val_split,
            shuffle=self.shuffle,
            seed=self.seed,
        )

        subj_train_dict = _get_dict(subj_train_paths,
                                    modalities,
                                    labels,
                                    train=True)
        subj_val_dict = _get_dict(subj_val_paths,
                                  modalities,
                                  labels,
                                  train=True)
        subj_test_dict = _get_dict(subj_test_paths,
                                   modalities,
                                   labels,
                                   train=False)

        return subj_train_dict, subj_test_dict, subj_val_dict

    @staticmethod
    def get_paths(
        data_root: PathType,
        stem: str = 'step01_structural_processing',
        has_train_test_split: bool = False,
        has_train_val_split: bool = False,
        test_split: Union[int, float] = 0.2,
        shuffle: bool = True,
        seed: int = 41,
    ) -> Tuple[SubjPathType, SubjPathType, SubjPathType]:
        """
        Get subject and scan IDs and the respective paths from the study data
        directory.

        Returns
        -------
        _ : {(str, str): Path}, {(str, str): Path}, {(str, str): Path}
            Paths for respectively, train, test and images and labels.
        """

        def _split_subj_train_paths(
                paths: OrderedDict) -> Tuple[OrderedDict, OrderedDict]:
            """Split dictionary into two proportially to `test_split`."""
            len_paths = len(paths)
            if isinstance(test_split, int):
                train_len = len_paths - test_split
                splits = [train_len, test_split]
            elif isinstance(test_split, float):
                test_len = int(np.floor(test_split * len_paths))
                train_len = len_paths - test_len
                splits = [train_len, test_len]
            else:
                raise ValueError(f"Unsupported type {type(test_split)}")
            indexes = list(range(len_paths))
            if shuffle:
                np.random.seed(seed)
                np.random.shuffle(indexes)
            train_idx, test_idx = indexes[:splits[0]], indexes[:splits[1]]

            paths_list = list(paths.items())

            subj_train_paths = OrderedDict([
                value for idx, value in enumerate(paths_list)
                if idx in train_idx
            ])
            subj_test_paths = OrderedDict([
                value for idx, value in enumerate(paths_list)
                if idx in test_idx
            ])
            return subj_train_paths, subj_test_paths

        data_root = Path(data_root)
        subj_pattern = r"[a-zA-z]{3}_[a-zA-Z]{2}_\d{4}"
        scan_pattern = r"[a-zA-z]{4}\d{3}"
        no_split_regex = re.compile("(" + subj_pattern + ")" + "/" +
                                    subj_pattern + "_" + "(" + scan_pattern +
                                    ")")
        has_split_regex = re.compile("(" + subj_pattern + ")" + "_" + "(" +
                                     scan_pattern + ")")

        def _get_subj_paths(data_root, regex):
            subj_paths = OrderedDict()
            for item in data_root.glob("*"):
                if not item.is_dir() and not item.name.startswith('.'):
                    match = regex.search(str(item))
                    if match is not None:
                        subj_id, scan_id = match.groups()
                        subj_paths[(subj_id, scan_id)] = data_root
            return subj_paths

        if not has_train_test_split:
            paths = OrderedDict()
            for item in data_root.glob("*/*"):
                if item.is_dir() and not item.name.startswith('.'):
                    match = no_split_regex.search(str(item))
                    if match is not None:
                        subj_id, scan_id = match.groups()
                        paths[(subj_id, scan_id)] = data_root / subj_id / (
                            subj_id + "_" + scan_id) / stem
            subj_train_paths, subj_test_paths = _split_subj_train_paths(paths)
        else:
            train_root = data_root / stem / "train"
            subj_train_paths = _get_subj_paths(train_root, has_split_regex)
            test_root = data_root / stem / "test"
            subj_test_paths = _get_subj_paths(test_root, has_split_regex)

        val_root = data_root / stem / "val"
        subj_val_paths = _get_subj_paths(
            val_root,
            has_split_regex) if has_train_val_split else OrderedDict()

        return subj_train_paths, subj_test_paths, subj_val_paths

    def default_preprocessing_transforms(self,
                                         **kwargs: Any) -> List[tio.Transform]:
        """
        List with preprocessing transforms to apply to all subjects.

        Returns
        -------
        _ : List[tio.Transform]
            Preprocessing transforms that should be applied to all subjects.
        """
        shape = kwargs.get("shape", (256, 256, 256))
        return [
            tio.RescaleIntensity((-1, 1)),
            tio.CropOrPad(shape),
            tio.EnsureShapeMultiple(8),  # better suited for U-Net type Nets
            tio.OneHot()  # for labels
        ]

    def default_augmentation_transforms(self,
                                        **kwargs: Any) -> List[tio.Transform]:
        """
        List with augmentation transforms to apply to training subjects.

        Returns
        -------
        _ : List[tio.Transform]
            Augmentation transforms to apply to training subjects.
        """
        random_gamma_p = kwargs.get("random_gamma_p", 0.5)
        random_noise_p = kwargs.get("random_noise_p", 0.5)
        random_motion_p = kwargs.get("random_motion_p", 0.1)
        random_bias_field_p = kwargs.get("random_bias_field_p", 0.25)
        return [
            tio.RandomAffine(),
            tio.RandomGamma(p=random_gamma_p),
            tio.RandomNoise(p=random_noise_p),
            tio.RandomMotion(p=random_motion_p),
            tio.RandomBiasField(p=random_bias_field_p),
        ]

    def save(self,
             dataloader: DataLoader,
             root: PathType = Path(
                 "/media/cerebro/Workspaces/Students/Eduardo_Diniz/Studies"),
             data_dir: str = 'processed_data',
             step: str = 'step01_structural_processing',
             fold: str = "train") -> None:
        """
        Arguments
        ---------
        root : Path or str, optional
            Root where to save data. Default = ``'~/LocalCerebro'``.
        """
        save_root = ensure_exists(
            Path(root).expanduser() / self.study / data_dir / step / fold)

        for batch in dataloader:
            subjects = get_subjects_from_batch(cast(Dict[str, Any], batch))
            for subject in subjects:
                subj_id = subject["subj_id"]
                scan_id = subject["scan_id"]
                for image_name in subject.get_images_names():
                    filename = self.intensity2template[image_name].substitute(
                        subj_id=subj_id, scan_id=scan_id)
                    subject[image_name].save(save_root / filename)
