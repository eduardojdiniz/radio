Module radio.data.validation
============================
Dataloaders are based on the PyTorch ``torch.utils.data.Dataloader`` data
primitive. They are wrappers around ``torch.utils.data.Dataset`` that enable
easy access to the dataset samples, i.e., they prepare your data for
training/testing. Specifically, dataloaders are iterables that abstracts the
complexity of retrieving "minibatches" from Datasets, reshuffling the data at
every epoch to reduce model overfitting, use Python's ``multiprocessing``
to speed up data retrieval, and automatic memory pinning, in an easy API.

Classes
-------

`KFoldValidation(train_dataset: Union[torch.utils.data.dataset.Dataset, radio.data.dataset.BaseVisionDataset], val_dataset: Union[torch.utils.data.dataset.Dataset, radio.data.dataset.BaseVisionDataset] = None, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0, collate_fn: Callable[[List[~Type]], Any] = None, pin_memory: bool = True, drop_last: bool = False, worker_init_fn: Callable[[int], None] = None, num_folds: int = 5, seed: int = 41)`
:   Create train and validation dataloaders for K-Fold Cross-Validation.
    
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

    ### Methods

    `setup(self, val_split: Union[int, float] = 0.2) ‑> None`
    :   Creates train and validation collection of samplers.
        
        Parameters
        ----------
        val_split: int or float, optional
            WARNING: val_split is not used in K-Fold validation. Left here just
            for compatibility with `OneFoldValidation`. Specify how the
            train_dataset should be split into train/validation datasets.
            Default = ``0.2``.

    `train_dataloader(self) ‑> Union[torch.utils.data.dataloader.DataLoader, Sequence[torch.utils.data.dataloader.DataLoader], Sequence[Sequence[torch.utils.data.dataloader.DataLoader]], Sequence[Dict[str, torch.utils.data.dataloader.DataLoader]], Dict[str, torch.utils.data.dataloader.DataLoader], Dict[str, Dict[str, torch.utils.data.dataloader.DataLoader]], Dict[str, Sequence[torch.utils.data.dataloader.DataLoader]]]`
    :   Generates one or multiple Pytorch DataLoaders for train.
        
        Parameters
        ----------
        sampler : Sampler
            Sampler for validation samples.
        
        Returns
        -------
        _ : Collection of DataLoader
            Collection of train dataloaders specifying training samples.

    `val_dataloader(self) ‑> Union[torch.utils.data.dataloader.DataLoader, Sequence[torch.utils.data.dataloader.DataLoader]]`
    :   Generates one or multiple Pytorch DataLoaders for validation.
        
        Parameters
        ----------
        sampler : Sampler
            Sampler for validation samples.
        
        Returns
        -------
        _ : Collection of DataLoader
            Collection of validation dataloaders specifying validation samples.

`OneFoldValidation(train_dataset: Union[torch.utils.data.dataset.Dataset, radio.data.dataset.BaseVisionDataset], val_dataset: Union[torch.utils.data.dataset.Dataset, radio.data.dataset.BaseVisionDataset] = None, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0, collate_fn: Callable[[List[~Type]], Any] = None, pin_memory: bool = True, drop_last: bool = False, worker_init_fn: Callable[[int], None] = None, num_folds: int = 2, seed: int = 41)`
:   Random split dataset into train and validation dataloaders.
    
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

    ### Methods

    `setup(self, val_split: Union[int, float] = 0.2) ‑> None`
    :   Creates train and validation collection of samplers.
        
        Parameters
        ----------
        val_split: int or float, optional
            Specify how the train_dataset should be split into
            train/validation datasets. Default = ``0.2``.

    `train_dataloader(self) ‑> Union[torch.utils.data.dataloader.DataLoader, Sequence[torch.utils.data.dataloader.DataLoader], Sequence[Sequence[torch.utils.data.dataloader.DataLoader]], Sequence[Dict[str, torch.utils.data.dataloader.DataLoader]], Dict[str, torch.utils.data.dataloader.DataLoader], Dict[str, Dict[str, torch.utils.data.dataloader.DataLoader]], Dict[str, Sequence[torch.utils.data.dataloader.DataLoader]]]`
    :   Generates one or multiple Pytorch DataLoaders for train.
        
        Returns
        -------
        _ : Collection of DataLoader
            Collection of train dataloaders specifying training samples.

    `val_dataloader(self) ‑> Union[torch.utils.data.dataloader.DataLoader, Sequence[torch.utils.data.dataloader.DataLoader]]`
    :   Generates one or multiple Pytorch DataLoaders for validation.
        
        Returns
        -------
        _ : Collection of DataLoaders
            Collection of validation dataloaders specifying validation samples.