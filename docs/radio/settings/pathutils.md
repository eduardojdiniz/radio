Module radio.settings.pathutils
===============================
Manages Paths for Saving Models and Logs.

Functions
---------

    
`ensure_exists(path: Union[str, pathlib.Path]) ‑> pathlib.Path`
:   Enforce the directory existence.

    
`is_dir_or_symlink(path: Union[str, pathlib.Path]) ‑> bool`
:   Check if the path is a directory or a symlink to a directory.

    
`is_valid_extension(filename: Union[str, pathlib.Path], extensions: Tuple[str, ...]) ‑> bool`
:   Verifies if given file name has a valid extension.
    
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

    
`is_valid_image(filename: Union[str, pathlib.Path]) ‑> bool`
:   Verifies if given file name has a valid image extension.
    
    Parameters
    ----------
    filename : str or Path
        Path to a file.
    
    Returns
    -------
    return : bool
        True if the filename ends with one of the valid image extensions.