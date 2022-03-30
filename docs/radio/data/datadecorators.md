Module radio.data.datadecorators
================================
Decorators for Sample parameters validation.

Functions
---------

    
`assert_type(*dynamic_args, test_type=typing.Union[torch.Tensor, radio.data.datatypes.Tensor, typing.Sequence[typing.Union[torch.Tensor, radio.data.datatypes.Tensor]], radio.data.datatypes.SeqTensor, typing.Sequence[typing.Union[torch.Tensor, radio.data.datatypes.Tensor, typing.Sequence[typing.Union[torch.Tensor, radio.data.datatypes.Tensor]], radio.data.datatypes.SeqTensor]], radio.data.datatypes.SeqSeqTensor], **dynamic_kwargs) ‑> Callable`
:   Assert ```*dynamic_args`` and ``**dynamic_kwargs`` are of ``test_type``.
    
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