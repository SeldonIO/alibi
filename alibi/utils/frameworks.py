from enum import Enum


class Framework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"

    @staticmethod
    def from_str(name: str):
        return {
            'pytorch': Framework.PYTORCH,
            'tensorflow': Framework.TENSORFLOW
        }[name]


try:
    import tensorflow as tf  # noqa
    has_tensorflow = True
except ImportError:
    has_tensorflow = False

try:
    import torch  # noqa
    has_pytorch = True
except ImportError:
    has_pytorch = False
