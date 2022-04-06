Module radio.data.inference.aggregator
======================================
Based on BaseDataModule for managing data. A vision datamodule that is
shareable, reusable class that encapsulates all the steps needed to process
data, i.e., decoupling datasets from models to allow building dataset-agnostic
models. They also allow you to share a full dataset without explaining how to
download, split, transform, and process the data.

Classes
-------

`PatchBasedInference(patch_size: Union[int, Tuple[int, int, int]] = 96, patch_overlap: Union[int, Tuple[int, int, int]] = (0, 0, 0), patch_batch_size: int = 32, padding_mode: Union[str, float, ForwardRef(None)] = None, overlap_mode: str = 'crop', num_workers: int = 0, pin_memory: bool = True, verbose: bool = False)`
:   Dense Patch-based inference.
    
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