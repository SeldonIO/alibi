# flake8: noqa: F401
from alibi.utils.frameworks import tensorflow_installed, pytorch_installed

# run decorators in the TensorFlow module
if tensorflow_installed():
    import alibi.utils.tensorflow

# run decorators in the PyTorch module
if pytorch_installed():
    import alibi.utils.pytorch
