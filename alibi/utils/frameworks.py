from enum import Enum


class Framework(str, Enum):
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"


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
