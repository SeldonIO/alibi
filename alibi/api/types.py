import numpy as np
from typing import Any


class Array(np.ndarray):

    def __class_getitem__(cls, params):
        try:
            dtype, shape = params
            if not isinstance(shape, tuple):
                raise ValueError('Shape must be a tuple of integers.')
        except TypeError:
            dtype = params
            shape = tuple()
        return type('Array', (Array,), {'__dtype__': dtype, '__shape__': shape})

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, val):
        dtype = getattr(cls, '__dtype__', None)
        if dtype == Any:
            dtype = None
        shape = getattr(cls, '__shape__', tuple())

        result = np.array(val, dtype=dtype, copy=False, ndmin=len(shape))
        assert not shape or len(shape) == len(result.shape)  # ndmin guarantees this

        # TODO: this can result in unexpected shapes if shape is misspecified
        if any((shape[i] != -1 and shape[i] != result.shape[i]) for i in range(len(shape))):
            result = result.reshape(shape)
        return result
