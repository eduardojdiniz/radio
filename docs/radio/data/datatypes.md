Module radio.data.datatypes
===========================
Sample  Variables and Types definition.

Functions
---------

    
`get_sample_type(sample_name: str) ‑> type`
:   Infer sample type from sample name.
    
    Parameters
    ----------
    sample_name: str
        Sample name, e.g., ``'Tensor'`.
    
    Returns
    -------
    sample_type: torch.Tensor or Image.Image
        Sample type, e.g., ``torch.Tensor``.

Classes
-------

`Img(data: SingletonVar, sample_type: InitVar[str] = 'PILImage')`
:   Immutable Image Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.Sample
    * typing.Generic

    ### Class variables

    `sample_type: InitVar[str]`
    :

`MutableImg(_data: SingletonVar, sample_type: InitVar[str] = 'PILImage')`
:   Mutable Image Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.MutableSample
    * typing.Generic

    ### Class variables

    `sample_type: InitVar[str]`
    :

`MutableNamedImg(_name: KeyVar, _data: SingletonVar, sample_type: InitVar[str] = 'PILImage')`
:   Mutable Named Image Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.MutableNamedSample
    * typing.Generic

    ### Class variables

    `sample_type: InitVar[str]`
    :

`MutableNamedSample(_name: KeyVar, _data: SingletonVar, sample_type: InitVar[str] = 'MutableSample')`
:   Mutable Named Sample Type.

    ### Ancestors (in MRO)

    * typing.Generic

    ### Descendants

    * radio.data.datatypes.MutableNamedImg
    * radio.data.datatypes.MutableNamedTensor

    ### Class variables

    `sample_type: InitVar[str]`
    :

    ### Instance variables

    `data: ~SingletonVar`
    :   data attribute

    `name: ~KeyVar`
    :   name attribute

`MutableNamedTensor(_name: KeyVar, _data: SingletonVar, sample_type: InitVar[str] = 'TorchTensor')`
:   Mutable Named Tensor Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.MutableNamedSample
    * typing.Generic

    ### Class variables

    `sample_type: InitVar[str]`
    :

`MutableSample(_data: SingletonVar, sample_type: InitVar[str] = 'MutableSample')`
:   Mutable Sample Type.

    ### Ancestors (in MRO)

    * typing.Generic

    ### Descendants

    * radio.data.datatypes.MutableImg
    * radio.data.datatypes.MutableTensor

    ### Class variables

    `sample_type: InitVar[str]`
    :

    ### Instance variables

    `data: ~SingletonVar`
    :   data attribute

`MutableSeqImg(data: MutableSeqVar, sample_type: InitVar[str] = 'MutableImg')`
:   Mutable Sequence of Images Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.MutableSeqSample
    * typing.Generic

    ### Class variables

    `sample_type: InitVar[str]`
    :

`MutableSeqNamedImg(name: KeyVar, data: MutableSeqNamedVar, sample_type: InitVar[str] = 'MutableNamedImg')`
:   Mutable Sequence of Named Images Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.MutableSeqNamedSample
    * typing.Generic

    ### Class variables

    `sample_type: InitVar[str]`
    :

`MutableSeqNamedSample(name: KeyVar, data: MutableSeqNamedVar, sample_type: InitVar[str] = 'MutableNamedSample')`
:   Mutable Sequence of Named Samples Type.

    ### Ancestors (in MRO)

    * typing.Generic

    ### Descendants

    * radio.data.datatypes.MutableSeqNamedImg
    * radio.data.datatypes.MutableSeqNamedTensor

    ### Class variables

    `sample_type: InitVar[str]`
    :

    ### Instance variables

    `data`
    :   Return an attribute of instance, which is of type owner.

    `name`
    :   Return an attribute of instance, which is of type owner.

`MutableSeqNamedTensor(name: KeyVar, data: MutableSeqNamedVar, sample_type: InitVar[str] = 'MutableNamedTensor')`
:   Mutable Sequence of NamedTensors Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.MutableSeqNamedSample
    * typing.Generic

    ### Class variables

    `sample_type: InitVar[str]`
    :

`MutableSeqSample(data: MutableSeqVar, sample_type: InitVar[str] = 'MutableSample')`
:   Mutable Sequence of Samples Type.

    ### Ancestors (in MRO)

    * typing.Generic

    ### Descendants

    * radio.data.datatypes.MutableSeqImg
    * radio.data.datatypes.MutableSeqTensor

    ### Class variables

    `sample_type: InitVar[str]`
    :

    ### Instance variables

    `data`
    :   Return an attribute of instance, which is of type owner.

`MutableSeqSeqImg(data: MutableSeqSeqVar)`
:   Mutable Nested Sequence of Images Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.MutableSeqSeqSample
    * typing.Generic

    ### Class variables

    `sample_type: str`
    :

`MutableSeqSeqNamedImg(data: MutableSeqSeqNamedVar)`
:   Mutable Nested Sequence of Named Images Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.MutableSeqSeqNamedSample
    * typing.Generic

    ### Class variables

    `sample_type: str`
    :

`MutableSeqSeqNamedSample(data: MutableSeqSeqNamedVar)`
:   Mutable Nested Sequence of Named Samples Type.

    ### Ancestors (in MRO)

    * typing.Generic

    ### Descendants

    * radio.data.datatypes.MutableSeqSeqNamedImg
    * radio.data.datatypes.MutableSeqSeqNamedTensor

    ### Class variables

    `sample_type: str`
    :

    ### Instance variables

    `data: ~MutableSeqSeqNamedVar`
    :   Return an attribute of instance, which is of type owner.

`MutableSeqSeqNamedTensor(data: MutableSeqSeqNamedVar)`
:   Mutable Nested Sequence of Named Tensors Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.MutableSeqSeqNamedSample
    * typing.Generic

    ### Class variables

    `sample_type: str`
    :

`MutableSeqSeqSample(data: MutableSeqSeqVar)`
:   Mutable Nested Sequence of Samples Type.

    ### Ancestors (in MRO)

    * typing.Generic

    ### Descendants

    * radio.data.datatypes.MutableSeqSeqImg
    * radio.data.datatypes.MutableSeqSeqTensor

    ### Class variables

    `sample_type: str`
    :

    ### Instance variables

    `data: ~MutableSeqSeqVar`
    :   Return an attribute of instance, which is of type owner.

`MutableSeqSeqTensor(data: MutableSeqSeqVar)`
:   Mutable Nested Sequence of Tensors Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.MutableSeqSeqSample
    * typing.Generic

    ### Class variables

    `sample_type: str`
    :

`MutableSeqTensor(data: MutableSeqVar, sample_type: InitVar[str] = 'MutableTensor')`
:   Mutable Sequence of Tensors Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.MutableSeqSample
    * typing.Generic

    ### Class variables

    `sample_type: InitVar[str]`
    :

`MutableTensor(_data: SingletonVar, sample_type: InitVar[str] = 'TorchTensor')`
:   Mutable Tensor Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.MutableSample
    * typing.Generic

    ### Class variables

    `sample_type: InitVar[str]`
    :

`NamedImg(name: KeyVar, data: SingletonVar, sample_type: InitVar[str] = 'PILImage')`
:   Immutable Named Image Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.NamedSample
    * typing.Generic

    ### Class variables

    `sample_type: InitVar[str]`
    :

`NamedSample(name: KeyVar, data: SingletonVar, sample_type: InitVar[str] = 'Sample')`
:   Immutable Named Sample Type.

    ### Ancestors (in MRO)

    * typing.Generic

    ### Descendants

    * radio.data.datatypes.NamedImg
    * radio.data.datatypes.NamedTensor

    ### Class variables

    `sample_type: InitVar[str]`
    :

    ### Instance variables

    `data`
    :   Return an attribute of instance, which is of type owner.

    `name`
    :   data given name

`NamedTensor(name: KeyVar, data: SingletonVar, sample_type: InitVar[str] = 'TorchTensor')`
:   Immutable Named Tensor Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.NamedSample
    * typing.Generic

    ### Class variables

    `sample_type: InitVar[str]`
    :

`Sample(data: SingletonVar, sample_type: InitVar[str] = 'Sample')`
:   Immutable Sample Type.

    ### Ancestors (in MRO)

    * typing.Generic

    ### Descendants

    * radio.data.datatypes.Img
    * radio.data.datatypes.Tensor

    ### Class variables

    `sample_type: InitVar[str]`
    :

    ### Instance variables

    `data`
    :   Return an attribute of instance, which is of type owner.

`SeqImg(data: SeqVar, sample_type: InitVar[str] = 'Img')`
:   Immutable Sequence of Images Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.SeqSample
    * typing.Generic

    ### Class variables

    `sample_type: InitVar[str]`
    :

`SeqNamedImg(name: KeyVar, data: SeqNamedVar, sample_type: InitVar[str] = 'NamedImg')`
:   Immutable Sequence of Named Images Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.SeqNamedSample
    * typing.Generic

    ### Class variables

    `sample_type: InitVar[str]`
    :

`SeqNamedSample(name: KeyVar, data: SeqNamedVar, sample_type: InitVar[str] = 'NamedSample')`
:   Immutable Sequence of Named Samples Type.

    ### Ancestors (in MRO)

    * typing.Generic

    ### Descendants

    * radio.data.datatypes.SeqNamedImg
    * radio.data.datatypes.SeqNamedTensor

    ### Class variables

    `sample_type: InitVar[str]`
    :

    ### Instance variables

    `data`
    :   Return an attribute of instance, which is of type owner.

    `name`
    :   Return an attribute of instance, which is of type owner.

`SeqNamedTensor(name: KeyVar, data: SeqNamedVar, sample_type: InitVar[str] = 'NamedTensor')`
:   Immutable Sequence of Named Tensors Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.SeqNamedSample
    * typing.Generic

    ### Class variables

    `sample_type: InitVar[str]`
    :

`SeqSample(data: SeqVar, sample_type: InitVar[str] = 'Sample')`
:   Immutable Sequence of Samples Type.

    ### Ancestors (in MRO)

    * typing.Generic

    ### Descendants

    * radio.data.datatypes.SeqImg
    * radio.data.datatypes.SeqTensor

    ### Class variables

    `sample_type: InitVar[str]`
    :

    ### Instance variables

    `data`
    :   Return an attribute of instance, which is of type owner.

`SeqSeqImg(data: SeqSeqVar)`
:   Immutable Nested Sequence of Images Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.SeqSeqSample
    * typing.Generic

    ### Class variables

    `sample_type: str`
    :

`SeqSeqNamedImg(data: SeqSeqNamedVar)`
:   Immutable Nested Sequence of Named Images Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.SeqSeqNamedSample
    * typing.Generic

    ### Class variables

    `sample_type: str`
    :

`SeqSeqNamedSample(data: SeqSeqNamedVar)`
:   Immutable Nested Sequence of Named Samples Type.

    ### Ancestors (in MRO)

    * typing.Generic

    ### Descendants

    * radio.data.datatypes.SeqSeqNamedImg
    * radio.data.datatypes.SeqSeqNamedTensor

    ### Class variables

    `sample_type: str`
    :

    ### Instance variables

    `data: ~SeqSeqNamedVar`
    :   Return an attribute of instance, which is of type owner.

`SeqSeqNamedTensor(data: SeqSeqNamedVar)`
:   Immutable Nested Sequence of Named Tensors Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.SeqSeqNamedSample
    * typing.Generic

    ### Class variables

    `sample_type: str`
    :

`SeqSeqSample(data: SeqSeqVar)`
:   Immutable Nested Sequence of Samples Type.

    ### Ancestors (in MRO)

    * typing.Generic

    ### Descendants

    * radio.data.datatypes.SeqSeqImg
    * radio.data.datatypes.SeqSeqTensor

    ### Class variables

    `sample_type: str`
    :

    ### Instance variables

    `data: ~SeqSeqVar`
    :   Return an attribute of instance, which is of type owner.

`SeqSeqTensor(data: SeqSeqVar)`
:   Immutable Nested Sequence of Tensors Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.SeqSeqSample
    * typing.Generic

    ### Class variables

    `sample_type: str`
    :

`SeqTensor(data: SeqVar, sample_type: InitVar[str] = 'Tensor')`
:   Immutable Sequence of Tensors Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.SeqSample
    * typing.Generic

    ### Class variables

    `sample_type: InitVar[str]`
    :

`Tensor(data: SingletonVar, sample_type: InitVar[str] = 'TorchTensor')`
:   Immutable Tensor Type.

    ### Ancestors (in MRO)

    * radio.data.datatypes.Sample
    * typing.Generic

    ### Class variables

    `sample_type: InitVar[str]`
    :