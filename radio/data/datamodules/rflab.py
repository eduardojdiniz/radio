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
from radio.settings.pathutils import is_dir_or_symlink, PathType, ensure_exists
from ..visiondatamodule import VisionDataModule
from ..datautils import get_subjects_from_batch
from ..datatypes import SubjDictType

__all__ = ["RFLabDataModule"]


class RFLabDataModule(VisionDataModule):
    """
    RFLab Data Module.

    Typical Workflow
    ----------------
    data = RFLabDataModule()
    data.prepare_data() # download
    data.setup(stage) # process and split
    data.teardown(stage) # clean-up

    Parameters
    ----------
    root : Path or str, optional
        Root to data root folder.
        Default = ``'/data'``.
    study : str, optional
        Study name. Default = ``'RFLab'``.
    subj_dir : str, optional
        Subdirectory where the subjects are located.
        Default = ``'NII/unprocessed'``.
    data_dir : str, optional
        Subdirectory where the subjects' data are located.
        Default = ``''``.
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
        If ``True``, resample all images to ``'T1'``. Default = ``False``.
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
    modalities : List[str], optional
        Which modalities to load. Default = ``['7T_MPR']``.
    labels : List[str], optional
        Which labels to load. Default = ``[]``.
    dims : Tuple[int, int, int], optional
        Max spatial dimensions across subjects' images. If ``None``, compute
        dimensions from dataset. Default = ``(368, 480, 384)``.
    seed : int, optional
        When `shuffle` is True, `seed` affects the ordering of the indices,
        which controls the randomness of each fold. It is also use to seed the
        RNG used by RandomSampler to generate random indexes and
        multiprocessing to generate `base_seed` for workers. Pass an int for
        reproducible output across multiple function calls. Default = ``41``.
    verbose : bool, optional
        If ``True``, print debugging messages. Default = ``False``.
    """
    name: str = "RFLab"
    dataset_cls = tio.SubjectsDataset
    img_template = Template(
        '${modality}/${subj_id}_-_${field}_-_${modality}.nii.gz')
    img_template_radio = Template('${subj_id}_-_${field}_-_${modality}.nii.gz')
    label_template: Template = Template("")
    label_template_radio: Template = Template("")

    def __init__(
        self,
        *args: Any,
        root: PathType = Path('/data'),
        study: str = 'RFLab',
        subj_dir: str = 'NII/unprocessed',
        data_dir: str = '',
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
        dims: Optional[Tuple[int, int, int]] = (368, 480, 384),
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
        self.modalities = modalities if modalities else ['7T_MPR']
        self.labels = labels if labels else []
        self.dims = dims
        self.use_augmentation = use_augmentation
        self.use_preprocessing = use_preprocessing
        self.resample = resample
        self.verbose = verbose

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
                train_subjects = self.get_subjects(fold="train")
                train_dataset = self.dataset_cls(
                    train_subjects,
                    transform=train_transforms,
                )
                val_dataset = self.dataset_cls(
                    train_subjects,
                    transform=val_transforms,
                )
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
                train_subjects = self.get_subjects(fold="train")
                self.train_dataset = self.dataset_cls(
                    train_subjects, transform=train_transforms)
                self.size_train = self.size_train_dataset(self.train_dataset)

                val_subjects = self.get_subjects(fold="val")
                self.val_dataset = self.dataset_cls(val_subjects,
                                                    transform=val_transforms)
                self.size_val = self.size_eval_dataset(self.val_dataset)

        if stage == "test" or stage is None:
            test_transforms = self.default_transforms(
                stage="test"
            ) if self.test_transforms is None else self.test_transforms
            test_subjects = self.get_subjects(fold="test")
            self.test_dataset = self.dataset_cls(test_subjects,
                                                 transform=test_transforms)
            self.size_test = self.size_eval_dataset(self.test_dataset)

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
        if not self.has_train_test_split:
            train_subjs, test_subjs, val_subjs = self.get_subjects_dicts(
                modalities=self.modalities, labels=self.labels)
        else:
            train_subjs, test_subjs, val_subjs = self.get_subjects_dicts_radio(
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

    def get_subjects_dicts_radio(
        self,
        modalities: List[str],
        labels: List[str],
    ) -> Tuple[SubjDictType, SubjDictType, SubjDictType]:
        """
        Get subject IDs and the respective paths from the study data
        directory if formated by radio.

        Returns
        -------
        _ : {(str, ...): Path}, {(str, ...): Path}, {(str, ...): Path}
            Paths for respectively, train, test and images and labels.
        """
        modalities = modalities if modalities else ['7T_MPR']
        labels = labels if labels else []

        subj_pattern = r"([0-9A-Za-z\-\_]+-\d{14})"
        field_pattern = r"(\d{1}T)"
        fold_pattern = r"\b(train|test|val)\b"
        modality_pattern = r"([0-9A-Za-z]+_[0-9A-Za-z]+)"
        regex = re.compile(fold_pattern + "/" + subj_pattern + "_-_" +
                           field_pattern + "_-_" + modality_pattern)

        train_subjects_dict: SubjDictType = OrderedDict()
        test_subjects_dict: SubjDictType = OrderedDict()
        val_subjects_dict: SubjDictType = OrderedDict()

        dicts = {
            "train": train_subjects_dict,
            "test": test_subjects_dict,
            "val": val_subjects_dict,
        }
        for item in (self.root / self.data_dir).glob("**/*"):
            if item.is_file() and not item.name.startswith('.'):
                match = regex.search(str(item))
                if match is not None:
                    label = ""
                    fold, subj_id, field, modality = match.groups()
                    img_basename = self.img_template_radio.substitute(
                        subj_id=subj_id, field=field, modality=modality)
                    img_path = self.root / self.data_dir / fold / img_basename
                    label_basename = self.label_template_radio.substitute(
                        subj_id=subj_id, field=field, modality=modality)
                    label_path = self.root / self.data_dir / fold / label_basename

                    if 'SPC' in modality:
                        modality = field + '_' + 'SPC'
                    elif 'MPR' in modality:
                        modality = field + '_' + 'MPR'
                    elif 'FLR' in modality:
                        modality = field + '_' + 'FLR'
                    elif 'LABEL' in modality:
                        label = field + '_' + modality

                    if subj_id not in dicts[fold]:
                        dicts[fold][subj_id] = OrderedDict({
                            "subj_id": subj_id,
                            "field": field,
                        })

                    if modality in modalities and img_path.is_file():
                        dicts[fold][subj_id].update(
                            {modality: tio.ScalarImage(img_path)})

                    if labels:
                        if label in labels and label_path.is_file():
                            dicts[fold][subj_id].update(
                                {label: tio.LabelMap(label_path)})

        # Remove labels from test subjects
        for _, subject in dicts["test"].items():
            for label in labels:
                subject.pop(label, None)

        # Remove subjects that don't have all modalities
        for fold in ["train", "test", "val"]:
            to_remove = []
            for subj_id, subject in dicts[fold].items():
                if not all(mod in subject.keys() for mod in modalities):
                    to_remove.append(subj_id)
            for subj_id in to_remove:
                dicts[fold].pop(subj_id, None)

        return train_subjects_dict, test_subjects_dict, val_subjects_dict

    def get_subjects_dicts(
        self,
        modalities: List[str],
        labels: List[str],
        test_split: Union[int, float] = 0.2,
    ) -> Tuple[SubjDictType, SubjDictType, SubjDictType]:
        """
        Get subject IDs and the respective paths from the study data
        directory.

        Returns
        -------
        _ : {(str, ...): Path}, {(str, ...): Path}, {(str, ...): Path}
            Paths for respectively, train, test and images and labels.
        """

        def _split_subjects_train_test(
                subjects: SubjDictType) -> Tuple[SubjDictType, SubjDictType]:
            """
            Split dictionary into two, proportially to the `test_split` ratio.
            """
            num_subjects = len(subjects)
            if isinstance(test_split, int):
                train_len = num_subjects - test_split
                splits = [train_len, test_split]
            elif isinstance(test_split, float):
                test_len = int(np.floor(test_split * num_subjects))
                train_len = num_subjects - test_len
                splits = [train_len, test_len]
            else:
                raise ValueError(f"Unsupported type {type(test_split)}")
            indexes = list(range(num_subjects))
            if self.shuffle:
                np.random.seed(self.seed)
                np.random.shuffle(indexes)

            train_idx, test_idx = indexes[:splits[0]], indexes[-splits[1]:]

            subjects_list = list(subjects.items())

            train_subjects_dict = OrderedDict([
                value for idx, value in enumerate(subjects_list)
                if idx in train_idx
            ])
            test_subjects_dict = OrderedDict([
                value for idx, value in enumerate(subjects_list)
                if idx in test_idx
            ])
            return train_subjects_dict, test_subjects_dict

        modalities = modalities if modalities else ['7T_MPR']
        labels = labels if labels else []

        subj_pattern = r"([0-9A-Za-z\-\_]+-\d{14})"
        field_pattern = r"(\d{1}T)"
        modality_pattern = r"([0-9A-Za-z]+_[0-9A-Za-z]+)"
        regex = re.compile(subj_pattern + "/" + field_pattern + "/" +
                           modality_pattern)

        subjects_dict: SubjDictType = OrderedDict()
        val_subjects_dict: SubjDictType = OrderedDict()

        for item in self.root.glob("**/*"):
            if item.is_dir() and not item.name.startswith('.'):
                match = regex.search(str(item))
                if match is not None:
                    label = ""
                    subj_id, field, modality = match.groups()
                    img_basename = self.img_template.substitute(
                        subj_id=subj_id, field=field, modality=modality)
                    img_path = self.root / subj_id / self.data_dir / field / img_basename
                    label_basename = self.label_template.substitute(
                        subj_id=subj_id, field=field, modality=modality)
                    label_path = self.root / subj_id / self.data_dir / field / label_basename

                    if 'SPC' in modality:
                        modality = field + '_' + 'SPC'
                    elif 'MPR' in modality:
                        modality = field + '_' + 'MPR'
                    elif 'FLR' in modality:
                        modality = field + '_' + 'FLR'
                    elif 'LABEL' in modality:
                        label = field + '_' + modality

                    if subj_id not in subjects_dict:
                        subjects_dict[subj_id] = OrderedDict({
                            "subj_id": subj_id,
                            "field": field,
                        })

                    if modality in modalities and img_path.is_file():
                        subjects_dict[subj_id].update(
                            {modality: tio.ScalarImage(img_path)})

                    if labels:
                        if label in labels and label_path.is_file():
                            subjects_dict[subj_id].update(
                                {label: tio.LabelMap(label_path)})

        train_subjects_dict, test_subjects_dict = _split_subjects_train_test(
            subjects_dict)

        # Remove labels from test subjects
        for _, subject in test_subjects_dict.items():
            for label in labels:
                subject.pop(label, None)

        # Remove subjects that don't have all modalities
        for subj_dict in [train_subjects_dict, test_subjects_dict]:
            to_remove = []
            for subj_id, subject in subj_dict.items():
                if not all(mod in subject.keys() for mod in modalities):
                    to_remove.append(subj_id)
            for subj_id in to_remove:
                subj_dict.pop(subj_id, None)

        return train_subjects_dict, test_subjects_dict, val_subjects_dict

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
            train_subjects = self.get_subjects(fold="train")
            test_subjects = self.get_subjects(fold="test")
            val_subjects = self.get_subjects(fold="val")
            shape = self.get_max_shape(train_subjects + test_subjects +
                                       val_subjects)
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

    def save(self,
             dataloader: DataLoader,
             root: PathType = Path(
                 "/media/cerebro/Workspaces/Students/Eduardo_Diniz/Studies"),
             subj_dir: str = 'radio_7T_MPR/unprocessed',
             data_dir: str = '',
             fold: str = "train") -> None:
        """
        Arguments
        ---------
        root : Path or str, optional
            Root where to save data. Default = ``'~/LocalCerebro'``.
        """
        save_root = ensure_exists(
            Path(root).expanduser() / self.study / subj_dir / fold / data_dir)

        image_name2modality = {
            "3T_MPR": "T1w_MPR1",
            "7T_MPR": "T1w_MPR1",
            "3T_SPC": "T2w_SPC1",
            "7T_SPC": "T2w_SPC1",
            "3T_FLR": "T2w_FLR1",
            "7T_FLR": "T2w_FLR1",
        }

        for batch in dataloader:
            subjects = get_subjects_from_batch(cast(Dict[str, Any], batch))
            for subject in subjects:
                subj_id = subject["subj_id"]
                field = subject["field"]
                for image_name in subject.get_images_names():
                    filename = self.img_template_radio.substitute(
                        subj_id=subj_id,
                        field=field,
                        modality=image_name2modality[image_name])
                    subject[image_name].save(save_root / Path(filename).name)
