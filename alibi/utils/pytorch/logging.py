from alibi.utils.logging import tensorboard_logger, TensorboardWriterBase
import torch  # noqa: F401


@tensorboard_logger
class PTTensorboardWriter(TensorboardWriterBase):
    framework = 'pytorch'

    def __init__(self):
        raise NotImplementedError("TensorBoard display for PyTorch is not supported at the moment!")
