from alibi.explainers.similarity.base import SimilarityClassifierExplainer
from typing import Union
from alibi.explainers.similarity.metrics import dot, cos, asym_dot


class ClassifierGradDotSimilarity(SimilarityClassifierExplainer):
    def __init__(self,
                 model: 'Union[tensorflow.keras.Model, torch.nn.Module]',
                 loss_fn: '''Callable[[Union[tensorflow.Tensor, torch.Tensor],
                                           Union[tensorflow.Tensor, torch.Tensor]],
                                       Union[tensorflow.Tensor, torch.Tensor]]''',
                 store_grads: bool = False,
                 seed: int = 0,
                 backend: str = "tensorflow",
                 **kwargs
                 ) -> None:

        super().__init__(model, loss_fn, dot, store_grads, seed, backend, **kwargs)


class ClassifierGradCosSimilarity(SimilarityClassifierExplainer):
    def __init__(self,
                 model: 'Union[tensorflow.keras.Model, torch.nn.Module]',
                 loss_fn: '''Callable[[Union[tensorflow.Tensor, torch.Tensor],
                                           Union[tensorflow.Tensor, torch.Tensor]],
                                       Union[tensorflow.Tensor, torch.Tensor]]''',
                 store_grads: bool = False,
                 seed: int = 0,
                 backend: str = "tensorflow",
                 **kwargs
                 ) -> None:

        super().__init__(model, loss_fn, cos, store_grads, seed, backend, **kwargs)


class ClassifierGradAsymDotSimilarity(SimilarityClassifierExplainer):
    def __init__(self,
                 model: 'Union[tensorflow.keras.Model, torch.nn.Module]',
                 loss_fn: '''Callable[[Union[tensorflow.Tensor, torch.Tensor],
                                           Union[tensorflow.Tensor, torch.Tensor]],
                                       Union[tensorflow.Tensor, torch.Tensor]]''',
                 store_grads: bool = False,
                 seed: int = 0,
                 backend: str = "tensorflow",
                 **kwargs
                 ) -> None:

        super().__init__(model, loss_fn, asym_dot, store_grads, seed, backend, **kwargs)