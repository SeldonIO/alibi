from alibi.utils.logging import tensorboard_logger, TensorboardWriter
import torch  # noqa: F401


@tensorboard_logger
class TensorboardWriter(TensorboardWriter):

    framework = 'pytorch'
    raise NotImplementedError("TensorBoard display for PyTorch is not supported at the moment!")
