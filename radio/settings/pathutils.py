#!/usr/bin/env python
# coding=utf-8
"""
Manages Paths for Saving Models and Logs.
"""

from pathlib import Path
from typing import Tuple, Union
import radio as rio

__all__ = [
    "PathType",
    "is_valid_extension",
    "is_valid_image",
    "SRC",
    "ROOT",
    "DATA_ROOT",
    "SAVE_ROOT",
    "CONF_ROOT",
    "is_dir_or_symlink",
    "ensure_exists",
]

PathType = Union[str, Path]

module_path = Path(rio.__file__)
SRC = module_path.parents[0].absolute()
ROOT = module_path.parents[1].absolute()
SAVE_ROOT = ROOT / "trials"
DATA_ROOT = ROOT / "data"
CONF_ROOT = ROOT / "conf"

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".pgm",
)


def is_valid_extension(filename: PathType, extensions: Tuple[str,
                                                             ...]) -> bool:
    """
    Verifies if given file name has a valid extension.

    Parameters
    ----------
    filename : str or Path
        Path to a file.
    extensions : Tuple[str, ...]
        Extensions to consider (lowercase).

    Returns
    -------
    return : bool
        True if the filename ends with one of given extensions.
    """
    fname = str(filename)
    return any(fname.lower().endswith(ext) for ext in extensions)


def is_valid_image(filename: PathType) -> bool:
    """
    Verifies if given file name has a valid image extension.

    Parameters
    ----------
    filename : str or Path
        Path to a file.

    Returns
    -------
    return : bool
        True if the filename ends with one of the valid image extensions.
    """
    return is_valid_extension(filename, IMG_EXTENSIONS)


def is_dir_or_symlink(path: PathType) -> bool:
    """
    Check if the path is a directory or a symlink to a directory.
    """
    path = Path(path)
    path = path.expanduser()
    return bool(path.is_dir() or path.is_symlink())


def ensure_exists(path: PathType) -> Path:
    """
    Enforce the directory existance.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
