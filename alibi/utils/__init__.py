# flake8: noqa: F401
from alibi.utils.frameworks import has_tensorflow, has_pytorch

# run decorators in the TensorFlow module
if has_tensorflow:
    import alibi.utils.tensorflow

# run decorators in the PyTorch module
if has_pytorch:
    import alibi.utils.pytorch
