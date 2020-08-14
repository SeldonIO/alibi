from alibi.utils.wrappers import blackbox_wrapper
import torch  # noqa: F401


@blackbox_wrapper(framework='pytorch')
def _wrap_black_box_predictor_pytorch(func):
    raise NotImplementedError("PyTorch is not suported at the moment!")
