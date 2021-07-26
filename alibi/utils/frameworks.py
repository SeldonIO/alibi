# https://github.com/SeldonIO/alibi-detect/blob/master/alibi_detect/utils/frameworks.py

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