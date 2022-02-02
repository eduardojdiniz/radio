#!/usr/bin/env python
# coding=utf-8
"""
Sample Variables and Types definition.
"""

from __future__ import annotations
from typing import (Dict, Generic, Mapping, Sequence, TypeVar, Union, cast,
                    List)
from dataclasses import dataclass, astuple, field, InitVar

import matplotlib.pyplot as plt  # type: ignore
import torch
from PIL import Image

plt.rcParams["savefig.bbox"] = "tight"

KeyVar = TypeVar("KeyVar", bound=str)

SAMPLES = ["Tensor", "MutableTensor", "Img", "MutableImg"]
NAMED_SAMPLES = [
    "NamedTensor", "MutableNamedTensor", "NamedImg", "MutableNamedImg"
]

# ############
# Sample Types
# ############

SingletonVar = TypeVar("SingletonVar", Image.Image, torch.Tensor)


# Unbounded Sample Types
@dataclass(frozen=True, repr=False)
class Sample(Generic[SingletonVar]):
    """Immutable Sample Type."""

    #: This stores the data like a tuple
    __slots__ = ("data", )
    data: SingletonVar
    sample_type: InitVar[str] = "Sample"

    def __repr__(self):
        return f"radio.{self.__class__.__qualname__}({self.data})"

    def __iter__(self):
        """Unpack like a tuple."""
        yield from astuple(self)


@dataclass(repr=False)
class MutableSample(Generic[SingletonVar]):
    """Mutable Sample Type."""

    #: This stores the data like a tuple
    __slots__ = ("data", )
    data: SingletonVar
    sample_type: InitVar[str] = "MutableSample"

    def __repr__(self):
        return f"radio.{self.__class__.__qualname__}({self.data})"

    def __iter__(self):
        """Unpack like a tuple."""
        yield from astuple(self)


# Bounded Sample Types
# Can also declare the bounded types using ``NewType``. E.g.,
# Tensor = NewType("Tensor", Sample[torch.Tensor])
@dataclass(frozen=True, repr=False)
class Tensor(Sample[torch.Tensor]):
    """Immutable Tensor Type."""
    sample_type: InitVar[str] = "TorchTensor"


@dataclass(frozen=True)
class Img(Sample[Image.Image]):
    """Immutable Image Type."""
    sample_type: InitVar[str] = "PILImage"


@dataclass(repr=False)
class MutableTensor(MutableSample[torch.Tensor]):
    """Mutable Tensor Type."""
    sample_type: InitVar[str] = "TorchTensor"


@dataclass(repr=False)
class MutableImg(MutableSample[Image.Image]):
    """Mutable Image Type."""
    sample_type: InitVar[str] = "PILImage"


# ##################
# Named Sample Types
# ##################


# Unbounded Named Sample Types
@dataclass(frozen=True, repr=False)
class NamedSample(Generic[KeyVar, SingletonVar]):
    """Immutable Named Sample Type."""

    #: This stores the data like a tuple
    __slots__ = ("name", "data")
    #: data given name
    name: KeyVar
    data: SingletonVar
    sample_type: InitVar[str] = "Sample"

    def __repr__(self):
        dictstring = f"{{{self.name}: {self.data}}}"
        return f"radio.{self.__class__.__qualname__}({dictstring})"

    def __iter__(self):
        """Unpack like a tuple."""
        yield from astuple(self)


@dataclass(repr=False)
class MutableNamedSample(Generic[KeyVar, SingletonVar]):
    """Mutable Named Sample Type."""

    #: This stores the data like a tuple
    __slots__ = ("name", "data")
    #: data given name
    name: KeyVar
    data: SingletonVar
    sample_type: InitVar[str] = "MutableSample"

    def __repr__(self):
        dictstring = f"{{{self.name}: {self.data}}}"
        return f"radio.{self.__class__.__qualname__}({dictstring})"

    def __iter__(self):
        """Unpack like a tuple."""
        yield from astuple(self)


# Bounded Named Sample Types
@dataclass(frozen=True, repr=False)
class NamedTensor(NamedSample[str, torch.Tensor]):
    """Immutable Named Tensor Type."""
    sample_type: InitVar[str] = "TorchTensor"


@dataclass(frozen=True, repr=False)
class NamedImg(NamedSample[str, Image.Image]):
    """Immutable Named Image Type."""
    sample_type: InitVar[str] = "PILImage"


@dataclass(repr=False)
class MutableNamedTensor(MutableNamedSample[str, torch.Tensor]):
    """Mutable Named Tensor Type."""
    sample_type: InitVar[str] = "TorchTensor"


@dataclass(repr=False)
class MutableNamedImg(MutableNamedSample[str, Image.Image]):
    """Mutable Named Image Type."""
    sample_type: InitVar[str] = "PILImage"


# Generic Sequene of Samples Types
SampleVar = TypeVar("SampleVar", Tensor, Img, NamedTensor, NamedImg,
                    MutableTensor, MutableImg, MutableNamedTensor,
                    MutableNamedImg)
SampleType = Union[Tensor, Img, NamedTensor, NamedImg, MutableTensor,
                   MutableImg, MutableNamedTensor, MutableNamedImg]

# ####################
# Hybrid Samples Types
# ####################

HybridSample = Union[SingletonVar, SampleVar]
HybridNamedSample = Union[Dict[str, SingletonVar], Dict[str, SampleVar]]

# Tensors
HybridTensor = HybridSample[torch.Tensor, Tensor]
HybridNamedTensor = HybridNamedSample[torch.Tensor, NamedTensor]
HybridMutableTensor = HybridSample[torch.Tensor, MutableTensor]
HybridMutableNamedTensor = HybridNamedSample[torch.Tensor, MutableNamedTensor]

# Imgs
HybridImg = HybridSample[Image.Image, Img]
HybridNamedImg = HybridNamedSample[Image.Image, NamedImg]
HybridMutableImg = HybridSample[Image.Image, MutableImg]
HybridMutableNamedImg = HybridNamedSample[Image.Image, MutableNamedImg]

HybridSampleVar = TypeVar("HybridSampleVar", HybridTensor, HybridNamedTensor,
                          HybridMutableTensor, HybridMutableNamedTensor,
                          HybridImg, HybridNamedImg, HybridMutableImg,
                          HybridMutableNamedImg)

# #########################
# Sequence of Samples Types
# #########################

Seq = Union[HybridSampleVar, Sequence[HybridSampleVar]]
SeqVar = TypeVar("SeqVar", Seq[HybridTensor], Seq[HybridImg])
MutableSeqVar = TypeVar("MutableSeqVar", Seq[HybridMutableTensor],
                        Seq[HybridMutableImg])


def _from_samples(class_name: str = "Tensor",
                  samples: Seq = None) -> Sequence[SampleVar]:
    """
    Gets a hybrid sequence of hybrid samples and outputs a sequence of samples.
    """
    sample_list: List[SampleVar] = []
    if samples is None:
        return sample_list

    assert class_name in SAMPLES, f"{class_name} not a valid Sample Type."

    sample_type: type
    if "Tensor" in class_name:
        sample_type = torch.Tensor
    else:
        sample_type = Image.Image

    if not isinstance(samples, Sequence):
        samples = [samples]

    for sample in samples:
        if isinstance(sample, sample_type):
            sample = globals()[class_name](sample)
            sample_list.append(sample)
        else:
            sample_list.append(sample)

    return sample_list


# Unbounded Sequence of Samples Types
@dataclass(frozen=True, repr=False)
class SeqSample(Generic[SeqVar]):
    """Immutable Sequence of Samples Type."""

    #: This stores the data like a tuple
    __slots__ = ("data", )
    data: SeqVar
    sample_type: InitVar[str] = "Sample"

    def __post_init__(self, sample_type):
        object.__setattr__(self, 'data', _from_samples(sample_type, self.data))

    def __repr__(self):
        return f"radio.{self.__class__.__qualname__}({self.data})"

    def __iter__(self):
        """Unpack like a tuple."""
        yield from astuple(self)

    def __len__(self):
        return len(self.data)


@dataclass(repr=False)
class MutableSeqSample(Generic[MutableSeqVar]):
    """Mutable Sequence of Samples Type."""

    #: This stores the data like a tuple
    __slots__ = ("data", )
    data: MutableSeqVar
    sample_type: InitVar[str] = "MutableSample"

    def __post_init__(self, sample_type):
        object.__setattr__(self, 'data', _from_samples(sample_type, self.data))

    def __repr__(self):
        return f"radio.{self.__class__.__qualname__}({self.data})"

    def __iter__(self):
        """Unpack like a tuple."""
        yield from astuple(self)

    def __len__(self):
        return len(self.data)


# Bounded Sequence of Samples Types
@dataclass(frozen=True, repr=False)
class SeqTensor(SeqSample[Seq[HybridTensor]]):
    """Immutable Sequence of Tensors Type."""
    sample_type: InitVar[str] = "Tensor"


@dataclass(frozen=True, repr=False)
class SeqImg(SeqSample[Seq[HybridImg]]):
    """Immutable Sequence of Images Type."""
    sample_type: InitVar[str] = "Img"


@dataclass(repr=False)
class MutableSeqTensor(MutableSeqSample[Seq[HybridMutableTensor]]):
    """Mutable Sequence of Tensors Type."""
    sample_type: InitVar[str] = "MutableTensor"


@dataclass(repr=False)
class MutableSeqImg(MutableSeqSample[Seq[HybridMutableImg]]):
    """Mutable Sequence of Images Type."""
    sample_type: InitVar[str] = "MutableImg"


# ###############################
# Sequence of Named Samples Types
# ###############################
SeqNamedVar = TypeVar("SeqNamedVar", Seq[HybridNamedTensor],
                      Seq[HybridNamedImg])
MutableSeqNamedVar = TypeVar("MutableSeqNamedVar",
                             Seq[HybridMutableNamedTensor],
                             Seq[HybridMutableNamedImg])


def _from_named_samples(  # noqa
        class_name: str = "NamedTensor",
        samples: Seq = None) -> Sequence[SampleVar]:
    sample_list: List[SampleVar] = []
    if not samples:
        return sample_list
    msg = f"{class_name} not a valid named sample type."

    assert class_name in NAMED_SAMPLES, msg
    sample_type: type
    if "Tensor" in class_name:
        sample_type = torch.Tensor
    else:
        sample_type = Image.Image

    if not isinstance(samples, Sequence):
        samples = [samples]

    def _parse_data(hybrid_sample: HybridSampleVar,
                    name: str = None) -> SampleVar:
        if not isinstance(hybrid_sample, sample_type):
            sample = cast(SampleVar, hybrid_sample)
        if name:
            sample = globals()[class_name](name, hybrid_sample)
        else:
            sample = globals()[class_name](hybrid_sample)
        return sample

    for sample in samples:
        if isinstance(sample, Mapping):
            for name, subsample in sample.items():
                sample_list.append(_parse_data(subsample, name))
        else:
            for subsample in samples:
                sample_list.append(_parse_data(subsample))

    return sample_list


# Unbounded Sequence of Named Samples Types
@dataclass(frozen=True, repr=False)
class SeqNamedSample(Generic[KeyVar, SeqNamedVar]):
    """Immutable Sequence of Named Samples Type."""

    #: This stores the data like a tuple
    __slots__ = ("name", "data")
    name: KeyVar
    data: SeqNamedVar
    sample_type: InitVar[str] = "NamedSample"

    def __post_init__(self, sample_type):
        object.__setattr__(self, 'data',
                           _from_named_seq(sample_type, self.data))

    def __repr__(self):
        dictstring = f"{{{self.name}: {self.data}}}"
        return f"radio.{self.__class__.__qualname__}({dictstring})"

    def __iter__(self):
        """Unpack like a tuple."""
        yield from astuple(self)


@dataclass(repr=False)
class MutableSeqNamedSample(Generic[KeyVar, MutableSeqNamedVar]):
    """Mutable Sequence of Named Samples Type."""

    #: This stores the data like a tuple
    __slots__ = ("name", "data")
    name: KeyVar
    data: MutableSeqNamedVar
    sample_type: InitVar[str] = "MutableNamedSample"

    def __post_init__(self, sample_type):
        object.__setattr__(self, 'data',
                           _from_named_seq(sample_type, self.data))

    def __repr__(self):
        dictstring = f"{{{self.name}: {self.data}}}"
        return f"radio.{self.__class__.__qualname__}({dictstring})"

    def __iter__(self):
        """Unpack like a tuple."""
        yield from astuple(self)

    def __len__(self):
        return len(self.data)


# Bounded Sequence of Named Samples Types
@dataclass(frozen=True, repr=False)
class SeqNamedTensor(SeqNamedSample[str, Seq[HybridNamedTensor]]):
    """Immutable Sequence of Named Tensors Type."""
    sample_type: InitVar[str] = "NamedTensor"


@dataclass(frozen=True, repr=False)
class SeqNamedImg(SeqNamedSample[str, Seq[HybridNamedImg]]):
    """Immutable Sequence of Named Images Type."""
    sample_type: InitVar[str] = "NamedImg"


@dataclass(repr=False)
class MutableSeqNamedTensor(
        MutableSeqNamedSample[str, Seq[HybridMutableNamedTensor]]):
    """Mutable Sequence of NamedTensors Type."""
    sample_type: InitVar[str] = "MutableNamedTensor"


@dataclass(repr=False)
class MutableSeqNamedImg(MutableSeqNamedSample[str,
                                               Seq[HybridMutableNamedImg]]):
    """Mutable Sequence of Named Images Type."""
    sample_type: InitVar[str] = "MutableNamedImg"


# Generic Sequence-based Types
SeqSampleVar = TypeVar("SeqSampleVar", SeqTensor, SeqImg, SeqNamedTensor,
                       SeqNamedImg, MutableSeqTensor, MutableSeqImg,
                       MutableSeqNamedTensor, MutableSeqNamedImg)

# ################################
# Hybrid Sequence of Samples Types
# ################################

HybridSeqSample = Union[HybridSampleVar, Sequence[HybridSampleVar],
                        SeqSampleVar]
HybridNamedSeqSample = Union[Dict[str, HybridSampleVar],
                             Dict[str, Sequence[HybridSampleVar]],
                             Dict[str, SeqSampleVar]]

# Tensors
HybridSeqTensor = HybridSeqSample[HybridTensor, SeqTensor]
HybridNamedSeqTensor = HybridNamedSeqSample[HybridNamedTensor, SeqNamedTensor]
HybridMutableSeqTensor = HybridSeqSample[HybridMutableTensor, MutableSeqTensor]
HybridMutableNamedSeqTensor = HybridNamedSeqSample[HybridMutableNamedTensor,
                                                   MutableSeqNamedTensor]

# Imgs
HybridSeqImg = HybridSeqSample[HybridImg, SeqImg]
HybridNamedSeqImg = HybridNamedSeqSample[HybridNamedImg, SeqNamedImg]
HybridMutableSeqImg = HybridSeqSample[HybridMutableImg, MutableSeqNamedImg]
HybridMutableNamedSeqImg = HybridNamedSeqSample[HybridMutableNamedImg,
                                                MutableSeqNamedImg]

HybridSeqSampleVar = TypeVar("HybridSeqSampleVar", HybridSeqTensor,
                             HybridNamedSeqTensor, HybridMutableSeqTensor,
                             HybridMutableNamedSeqTensor, HybridSeqImg,
                             HybridNamedSeqImg, HybridMutableSeqImg,
                             HybridMutableNamedSeqImg)

# #####################################
# Sequence of Sequence of Samples Types
# #####################################

SeqSeq = Union[HybridSeqSampleVar, Sequence[HybridSeqSampleVar]]
SeqSeqVar = TypeVar("SeqSeqVar", SeqSeq[HybridSeqTensor], SeqSeq[HybridSeqImg])
MutableSeqSeqVar = TypeVar("MutableSeqSeqVar", SeqSeq[HybridMutableSeqTensor],
                           SeqSeq[HybridMutableSeqImg])


def _from_seq(class_name: str = "Tensor",
              samples: SeqSeq = None) -> Sequence[SeqSampleVar]:
    seq_list: List[SeqSampleVar] = []
    if not samples:
        return seq_list

    assert class_name in SAMPLES, f"{class_name} not a valid Sample Type."

    if not isinstance(samples, Sequence):
        samples = [samples]

    for sample in samples:
        if isinstance(sample, type(Seq)):
            sample = globals()["Seq" + class_name](sample)
            seq_list.append(sample)
        else:
            seq_list.append(sample)

    return seq_list


@dataclass(frozen=True)
class SeqSeqSample(Generic[SeqSeqVar]):
    """Immutable Nested Sequence of Samples Type."""

    #: This stores the data like a tuple
    __slots__ = ("data", )
    data: SeqSeqVar
    sample_type: str = field(default="SeqSample", init=False)

    def __post_init__(self):
        self.data = _from_seq(self.sample_type, self.data)

    def __iter__(self):
        """Unpack like a tuple."""
        yield from astuple(self)

    def __len__(self):
        return len(self.data)


@dataclass
class MutableSeqSeqSample(Generic[MutableSeqSeqVar]):
    """Mutable Nested Sequence of Samples Type."""

    #: This stores the data like a tuple
    __slots__ = ("data", )
    data: MutableSeqSeqVar
    sample_type: str = field(default="MutableSeqSample", init=False)

    def __post_init__(self):
        self.data = _from_seq(self.sample_type, self.data)

    def __iter__(self):
        """Unpack like a tuple."""
        yield from astuple(self)

    def __len__(self):
        return len(self.data)


# Bounded Nested Sequence of Samples Types
@dataclass(frozen=True)
class SeqSeqTensor(SeqSeqSample[SeqSeq[HybridSeqTensor]]):
    """Immutable Nested Sequence of Tensors Type."""
    sample_type: str = field(default="SeqTensor", init=False)


@dataclass(frozen=True)
class SeqSeqImg(SeqSeqSample[SeqSeq[HybridSeqImg]]):
    """Immutable Nested Sequence of Images Type."""
    sample_type: str = field(default="SeqImg", init=False)


@dataclass
class MutableSeqSeqTensor(MutableSeqSeqSample[SeqSeq[HybridMutableSeqTensor]]):
    """Mutable Nested Sequence of Tensors Type."""
    sample_type: str = field(default="MutableSeqTensor", init=False)


@dataclass
class MutableSeqSeqImg(MutableSeqSeqSample[SeqSeq[HybridMutableSeqImg]]):
    """Mutable Nested Sequence of Images Type."""
    sample_type: str = field(default="MutableSeqImg", init=False)


# ###########################################
# Sequence of Sequence of Named Samples Types
# ###########################################

SeqSeqNamedVar = TypeVar("SeqSeqNamedVar", SeqSeq[HybridNamedSeqTensor],
                         SeqSeq[HybridNamedSeqImg])
MutableSeqSeqNamedVar = TypeVar("MutableSeqSeqNamedVar",
                                SeqSeq[HybridMutableNamedSeqTensor],
                                SeqSeq[HybridMutableNamedSeqImg])


def _from_named_seq(class_name: str = "NamedTensor",
                    samples: SeqSeq = None) -> Sequence[SeqSampleVar]:
    seq_list: List[SeqSampleVar] = []
    if not samples:
        return seq_list
    msg = f"{class_name} not a valid named sample type."
    assert class_name in NAMED_SAMPLES, msg

    if not isinstance(samples, Sequence):
        samples = [samples]

    for sample in samples:
        if isinstance(sample, type(Seq)):
            sample = globals()["Seq" + class_name](sample)
            seq_list.append(sample)
        else:
            seq_list.append(sample)

    return seq_list


@dataclass(frozen=True)
class SeqSeqNamedSample(Generic[SeqSeqNamedVar]):
    """Immutable Nested Sequence of Named Samples Type."""

    #: This stores the data like a tuple
    __slots__ = ("data", )
    sample_type: str = field(default="SeqNamedSample", init=False)
    data: SeqSeqNamedVar

    def __post_init__(self):
        self.data = _from_named_seq(self.sample_type, self.data)

    def __iter__(self):
        """Unpack like a tuple."""
        yield from astuple(self)

    def __len__(self):
        return len(self.data)


@dataclass
class MutableSeqSeqNamedSample(Generic[MutableSeqSeqNamedVar]):
    """Mutable Nested Sequence of Named Samples Type."""

    #: This stores the data like a tuple
    __slots__ = ("data", )
    sample_type: str = field(default="MutableSeqNamedSample", init=False)
    data: MutableSeqSeqNamedVar

    def __post_init__(self):
        self.data = _from_named_seq(self.sample_type, self.data)

    def __iter__(self):
        """Unpack like a tuple."""
        yield from astuple(self)

    def __len__(self):
        return len(self.data)


# Bounded Nested Sequence of Samples Types
@dataclass(frozen=True)
class SeqSeqNamedTensor(SeqSeqNamedSample[SeqSeq[HybridNamedSeqTensor]]):
    """Immutable Nested Sequence of Named Tensors Type."""
    sample_type: str = field(default="SeqNamedTensor", init=False)


@dataclass(frozen=True)
class SeqSeqNamedImg(SeqSeqNamedSample[SeqSeq[HybridNamedSeqImg]]):
    """Immutable Nested Sequence of Named Images Type."""
    sample_type: str = field(default="SeqNamedImg", init=False)


@dataclass
class MutableSeqSeqNamedTensor(
        MutableSeqSeqNamedSample[SeqSeq[HybridMutableNamedSeqTensor]]):
    """Mutable Nested Sequence of Named Tensors Type."""
    sample_type: str = field(default="MutableSeqNamedTensor", init=False)


@dataclass
class MutableSeqSeqNamedImg(
        MutableSeqSeqNamedSample[SeqSeq[HybridMutableNamedSeqImg]]):
    """Mutable Nested Sequence of Named Images Type."""
    sample_type: str = field(default="MutableSeqNamedImg", init=False)


# Generic Nested Sequence-based Types
SeqSeqSampleVar = TypeVar("SeqSeqSampleVar", SeqSeqTensor, SeqSeqImg,
                          SeqSeqNamedTensor, SeqSeqNamedImg,
                          MutableSeqSeqTensor, MutableSeqSeqImg,
                          MutableSeqSeqNamedTensor, MutableSeqSeqNamedImg)

# #######################################
# Hybrid Nested Sequence of Samples Types
# #######################################
HybridSeqSeqSample = Union[HybridSeqSampleVar, Sequence[HybridSeqSampleVar],
                           SeqSeqSampleVar]
HybridNamedSeqSeqSample = Union[Dict[str, HybridSeqSampleVar],
                                Dict[str, Sequence[HybridSeqSampleVar]],
                                Dict[str, SeqSeqSampleVar]]

# Tensors
HybridSeqSeqTensor = HybridSeqSeqSample[HybridSeqTensor, SeqSeqTensor]
HybridNamedSeqSeqTensor = HybridNamedSeqSeqSample[HybridNamedSeqTensor,
                                                  SeqSeqNamedTensor]
HybridMutableSeqSeqTensor = HybridSeqSeqSample[HybridMutableSeqTensor,
                                               MutableSeqSeqTensor]
HybridMutableNamedSeqSeqTensor = HybridNamedSeqSeqSample[
    HybridMutableNamedSeqTensor, MutableSeqSeqNamedTensor]

# Imgs
HybridSeqSeqImg = HybridSeqSeqSample[HybridSeqImg, SeqSeqImg]
HybridNamedSeqSeqImg = HybridNamedSeqSeqSample[HybridNamedSeqImg,
                                               SeqSeqNamedImg]
HybridMutableSeqSeqImg = HybridSeqSeqSample[HybridMutableSeqImg,
                                            MutableSeqSeqNamedImg]
HybridMutableNamedSeqSeqImg = HybridNamedSeqSeqSample[HybridMutableNamedSeqImg,
                                                      MutableSeqSeqNamedImg]

HybridSeqSeqSampleVar = TypeVar(
    "HybridSeqSeqSampleVar", HybridSeqSeqTensor, HybridNamedSeqSeqTensor,
    HybridMutableSeqSeqTensor, HybridMutableNamedSeqSeqTensor, HybridSeqSeqImg,
    HybridNamedSeqSeqImg, HybridMutableSeqSeqImg, HybridMutableNamedSeqSeqImg)

# ##############
# Umbrella Types
# ##############

# Umbrella bounded Sample Types
MultiImg = Union[HybridImg, HybridSeqImg, HybridSeqSeqImg]
MutableMultiImg = Union[HybridMutableImg, HybridMutableSeqImg,
                        HybridMutableSeqSeqImg]
MultiTensor = Union[HybridTensor, HybridSeqTensor, HybridSeqSeqTensor]
MutableMultiTensor = Union[HybridMutableTensor, HybridMutableSeqTensor,
                           HybridMutableSeqSeqTensor]

# Aliases
Imgs = MultiImg
MutableImgs = MutableMultiImg
Tensors = MultiTensor
MutableTensors = MutableMultiTensor

# ###########################
# Umbrella Named Sample Types
# ###########################

# Umbrella Unbounded Sample Types

# Umbrella bounded Sample Types
MultiNamedImg = Union[HybridNamedImg, HybridNamedSeqImg, HybridNamedSeqSeqImg]
MutableMultiNamedImg = Union[HybridMutableNamedImg, HybridMutableNamedSeqImg,
                             HybridMutableNamedSeqSeqImg]
MultiNamedTensor = Union[HybridNamedTensor, HybridNamedSeqTensor,
                         HybridNamedSeqSeqTensor]
MutableMultiNamedTensor = Union[HybridMutableNamedTensor,
                                HybridMutableNamedSeqTensor,
                                HybridMutableNamedSeqSeqTensor]

# Aliases
NamedImgs = MultiNamedImg
MutableNamedImgs = MutableMultiNamedImg
NamedTensors = MultiNamedTensor
MutableNamedTensors = MutableMultiNamedTensor
