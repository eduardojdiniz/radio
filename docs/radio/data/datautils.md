Module radio.data.datautils
===========================
Data related utilities.

Functions
---------

    
`check_integrity(path: pathlib.Path, md5: Optional[str] = None) ‑> bool`
:   

    
`default_image_loader(path: pathlib.Path) ‑> PIL.Image.Image`
:   Load image file as RGB PIL Image
    
    Parameters
    ----------
    path : Path
        Image file path
    
    Returns
    -------
    return : Image.Image
       RGB PIL Image

    
`denormalize(tensor: torch.Tensor, mean: Tuple[float, ...] = None, std: Tuple[float, ...] = None)`
:   Undoes mean/standard deviation normalization, zero to one scaling, and
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

    
`get_first_batch(loader: Iterable, default: Optional[~Var] = None) ‑> Optional[~Var]`
:   Returns the first item in the given iterable or `default` if empty,
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

    
`load_standard_test_imgs(directory: pathlib.Path = WindowsPath('C:/Users/LIW82/Lab Work/radio/imgs'))`
:   

    
`plot(imgs: Union[torch.Tensor, radio.data.datatypes.Tensor, Sequence[Union[torch.Tensor, radio.data.datatypes.Tensor]], radio.data.datatypes.SeqTensor, Sequence[Union[torch.Tensor, radio.data.datatypes.Tensor, Sequence[Union[torch.Tensor, radio.data.datatypes.Tensor]], radio.data.datatypes.SeqTensor]], radio.data.datatypes.SeqSeqTensor], baseline_imgs: Union[torch.Tensor, radio.data.datatypes.Tensor, Sequence[Union[torch.Tensor, radio.data.datatypes.Tensor]], radio.data.datatypes.SeqTensor, Sequence[Union[torch.Tensor, radio.data.datatypes.Tensor, Sequence[Union[torch.Tensor, radio.data.datatypes.Tensor]], radio.data.datatypes.SeqTensor]], radio.data.datatypes.SeqSeqTensor] = None, row_titles: List[str] = None, fig_title: str = None, **imshow_kwargs) ‑> None`
:   Plot images in a 2D grid.
    
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

    
`plot_batch(batch: Dict, num_samples: int = 5, intensities: Optional[List[str]] = None, labels: Optional[List[str]] = None, exclude_keys: Optional[List[str]] = None) ‑> None`
:   plot images and labels from a batch of images