Module radio.data.dataset
=========================
Datasets are based on the PyTorch ``torch.utils.data.Dataset`` data
primitive. They store the samples and their corresponding labels. Pytorch
domain libraries (e.g., vision, text, audio) provide pre-loaded datasets (e.g.,
MNIST) that subclass ``torch.utils.data.Dataset`` and implement functions
specific to the particular data. They can be used to prototype and benchmark
your model. You can find them at
[Image Datasets](https://pytorch.org/vision/stable/datasets.html),
[Text Datasets](https://pytorch.org/text/stable/datasets.html), and
[Audio Datasets](https://pytorch.org/audio/stable/datasets.html).

This module implements an abstract base class `BaseVisionDataset` for vision
datasets. It also replicates the official PyTorch image folder
(https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so it can inherent from `BaseVisionDataset` and have extended functionality.

Classes
-------

`BaseVisionDataset(root: Union[str, pathlib.Path] = WindowsPath('C:/Users/LIW82/Lab Work/radio/dataset'), transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, max_class_size: int = 9223372036854775807, max_dataset_size: int = 9223372036854775807)`
:   Base Class For making datasets which are compatible with torchvision.
    It is necessary to override the ``__getitem__`` and ``__len__`` method.
    
    To create a subclass, you need to implement the following functions:
    
    <__init__>:
        (Optionally) Initialize the class, first call super.__init__(root,
        train, transform, target_transform, **kwargs).
    <__len__>:
        Return the number of samples in the dataset.
    <__getitem__>:
        Get a data point.
    
    Parameters
    ----------
    root : Path or str
        Data root directory. Where to save/load the data.
    transform : Optional[Callable]
        A function/transform that takes in an PIL image and returns a
        transformed version, e.g, ``torchvision.transforms.RandomCrop``.
    target_transform : Optional[Callable]
        A function/transform that takes in the target and transforms it.

    ### Ancestors (in MRO)

    * torchvision.datasets.vision.VisionDataset
    * torch.utils.data.dataset.Dataset
    * typing.Generic

    ### Descendants

    * radio.data.dataset.FolderDataset

    ### Class variables

    `functions: Dict[str, Callable]`
    :

`FolderDataset(root: Union[str, pathlib.Path], loader: Callable[[pathlib.Path], Any], transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, extensions: Optional[Tuple[str, ...]] = None, is_valid_file: Optional[Callable[[pathlib.Path], bool]] = None, return_paths: bool = False, max_class_size: int = 9223372036854775807, max_dataset_size: int = 9223372036854775807)`
:   A generic folder dataset.
    
    This default directory structure can be customized by overriding the
    :meth:`find_classes` and :meth:`make_dataset` methods.
    
    Attributes
    ----------
    classes : list
        List of the class names sorted alphabetically.
    num_classes : int
        Number of classes in the dataset.
    class_to_idx : dict
        Dict with items (class_name, class_index).
    samples : list
        List of (sample, class_index) tuples.
    targets : list
        The class_index value for each image in the dataset.
    
    Parameters
    ----------
    root : Path or str
        Data root directory. Where to save/load the data.
    loader : Callable
        A function to load a sample given its path.
    transform : Optional[Callable]
        A function/transform that takes in a sample and returns a
        transformed version, e.g, ``torchvision.transforms.RandomCrop`` for
        images.
    target_transform : Callable[Optional]
        Optional function/transform that takes in a target and returns a
        transformed version.
    extensions : Tuple[str]
        A list of allowed extensions.
    is_valid_file :  Optional[Callable[[Path], bool]]
        A function that takes path of a file and check if the file is a
        valid file (used to check of corrupt files).
    return_paths : bool
        If True, calling the dataset returns `(sample, target), target,
        sample path` instead of returning `(sample, target), target`.
    
    Notes
    -----
    Both `extensions` and `is_valid_file` cannot be None or not None at the
    same time.

    ### Ancestors (in MRO)

    * radio.data.dataset.BaseVisionDataset
    * torchvision.datasets.vision.VisionDataset
    * torch.utils.data.dataset.Dataset
    * typing.Generic

    ### Descendants

    * radio.data.dataset.ImageFolder

    ### Class variables

    `functions: Dict[str, Callable]`
    :

    ### Static methods

    `find_classes(directory: pathlib.Path) ‑> Tuple[List[str], Dict[str, int]]`
    :   Find the class folders in a image dataset structured as follows:
        
        directory/
        ├── class_x
        │   ├── xxx.ext
        │   ├── xxy.ext
        │   └── ...
        │   └── xxz.ext
        └── class_y
            ├── 123.ext
            ├── nsdf3.ext
            └── ...
            └── asd932_.ext
        
        
        This method can be overridden to only consider a subset of classes,
        or to adapt to a different dataset directory structure.
        
        Arguments
        ---------
        directory : Path
            Root directory path, corresponding to ``self.root``.
        
        Raises
        ------
        FileNotFoundError: If ``directory`` has no class folders.
        
        Returns
        -------
        _: Tuple[List[str], Dict[str, int]]
            List of all classes and dictionary mapping each class to an index.

    `make_dataset(directory: pathlib.Path, class_to_idx: Dict[str, int], extensions: Optional[Tuple[str, ...]] = None, is_valid_file: Optional[Callable[[pathlib.Path], bool]] = None, max_class_size: int = 9223372036854775807, max_dataset_size: int = 9223372036854775807) ‑> List[Tuple[pathlib.Path, int]]`
    :   Generates a list of images of a form (path_to_sample, class).
        This can be overridden to e.g. read files from a compressed zip file
        instead of from the disk.
        
        Parameters
        ----------
        directory : Path
            root dataset directory, corresponding to ``self.root``.
        class_to_idx : Dict[str, int]
            Dictionary mapping class name to class index.
        extensions : Tuple[str]
            A list of allowed extensions.
        is_valid_file :  Optional[Callable[[Path], bool]]
            A function that takes path of a file and check if the file is a
            valid file (used to check of corrupt files).
        max_dataset_size : int
            Maximum number of samples allowed in the dataset.
        max_class_size : int
            Maximum number of samples allowed per class.
        
        Raises
        ------
        ValueError: In case ``class_to_idx`` is empty.
        FileNotFoundError: In case no valid file was found for any class.
        
        Returns
        -------
        _: Sample
            Samples of a form (path_to_sample, class).
        
        Notes
        -----
        Both `extensions` and `is_valid_file` cannot be None or not None at the
        same time.

`ImageFolder(root: Union[str, pathlib.Path], loader: Callable[[pathlib.Path], Any] = <function default_image_loader>, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, is_valid_file: Optional[Callable[[pathlib.Path], bool]] = None, return_paths: bool = False, max_class_size: int = 9223372036854775807, max_dataset_size: int = 9223372036854775807)`
:   A generic image folder dataset where the images are arranged in this way by
    default:
    
    root/
    ├── dog
    │   ├── xxx.png
    │   ├── xxy.png
    │   └── ...
    │   └── xxz.png
    └── cat
        ├── 123.png
        ├── nsdf3.png
        └── ...
        └── asd932_.png
    
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
    root : Path or str
        Data root directory. Where to save/load the data.
    loader : Optional[Callable]
        A function to load a image given its path.
    transform : Optional[Callable]
        A function/transform that takes in an PIL image and returns a
        transformed version, e.g, ``torchvision.transforms.RandomCrop``.
    target_transform : Callable[Optional]
        Optional function/transform that takes in a target and returns a
        transformed version.
    is_valid_file : Optional[Callable[[Path], bool]]
        A function that takes path of an image file and check if the file
        is a valid image file (used to check of corrupt files).
    return_paths : bool
        If True, calling the dataset returns `(img, label), label, image
        path` instead of returning `(img, label), label`.

    ### Ancestors (in MRO)

    * radio.data.dataset.FolderDataset
    * radio.data.dataset.BaseVisionDataset
    * torchvision.datasets.vision.VisionDataset
    * torch.utils.data.dataset.Dataset
    * typing.Generic

    ### Class variables

    `functions: Dict[str, Callable]`
    :