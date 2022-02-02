#!/usr/bin/env python
# coding=utf-8
"""
Decorators for Sample parameters validation.
"""

import functools
from typing import Type, Callable
from .datatypes import Tensors


def assert_type(*dynamic_args,
                _func: Callable = None,
                test_type=Tensors,
                **dynamic_kwargs) -> Callable:
    """
    Assert ```*dynamic_args`` and ``**dynamic_kwargs`` are of ``test_type``.

    Parameters
    ----------
    _func : Callable, optional
    test_type: Tensors, optional
        Type to test ``dynamic_args`` and ``dynamic_kwargs`` aganst it.
        Default = ``Tensors``.

    Returns
    -------
    _ : Callable
        Decorated function.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            test_args = [arg for arg in args if arg in dynamic_args]
            test_kwargs = {
                key: val
                for key, val in kwargs.items()
                if key not in dynamic_kwargs or val != dynamic_kwargs[key]
            }
            assert all(isinstance(arg, Type[test_type]) for arg in test_args)

            assert all(
                isinstance(val, Type[test_type])
                for key, val in test_kwargs.items())
            value = func(*args, **kwargs)
            return value

        return wrapper

    if _func is None:
        return decorator
    return decorator(_func)
