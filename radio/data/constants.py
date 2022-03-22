#!/usr/bin/env python
# coding=utf-8
"""
Constants.
"""

import torch

__all__ = [
    "INTENSITY", "LABEL", "SAMPLING_MAP", "PATH", "TYPE", "STEM", "DATA",
    "AFFINE", "IMAGE", "LOCATION", "HISTORY", "CHANNELS_DIMENSION",
    "MIN_FLOAT_32"
]

# Image types
INTENSITY = 'intensity'
LABEL = 'label'
SAMPLING_MAP = 'sampling_map'

# Keys for dataset samples
PATH = 'path'
TYPE = 'type'
STEM = 'stem'
DATA = 'data'
AFFINE = 'affine'

# For aggregator
IMAGE = 'image'
LOCATION = 'location'

# For special collate function
HISTORY = 'history'

# In PyTorch convention
CHANNELS_DIMENSION = 1

# Floating point error
MIN_FLOAT_32 = torch.finfo(torch.float32).eps
