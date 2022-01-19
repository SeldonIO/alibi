import numpy as np
from typing import TYPE_CHECKING, Dict, Any, Callable, Optional, Union
from tqdm import tqdm
from alibi.api.interfaces import Explainer, Explanation
from alibi.explainers.similarity.backends import select_backend


if TYPE_CHECKING:
    import tensorflow
    import torch


class GradMatrixGradExplainer(Explainer):
    def __init__(self,
                 model: 'Union[tensorflow.keras.Model, torch.nn.Module]',
                 loss_fn: '''Callable[[
                                       Union[tensorflow.Tensor, torch.Tensor],
                                       Union[tensorflow.Tensor, torch.Tensor]],
                                   Union[tensorflow.Tensor, torch.Tensor]]''',
                 sim_fn: Union[Callable, str] = 'dot',
                 store_grads: bool = False,
                 seed: int = 0,
                 backend: str = "tensorflow",
                 **kwargs
                 ):
        """
        Constructor

        Parameters
        ----------
        model:
            Model to be explained.
        loss_fn:
            Loss function.
        sim_fn:
            Similarity function. This can be either: `dot` | `cos`.
        store_grads:
            Whether to precompute and store the gradients when fitting.
        seed:
            Seed to be used,
        backend:
            Deep learning backend: `tensorflow` | `pytorch`. Default `tensorflow`.
        """
        # Select backend.
        self.backend = select_backend(backend, **kwargs)

        # Set seed for reproducibility.
        self.backend.set_seed(seed)

        self.model = model
        self.loss_fn = loss_fn
        self.sim_fn = sim_fn
        self.store_grads = store_grads
        self.seed = seed
        self.x_train = None
        self.y_train = None
        self.x_train_full = None
        self.y_train_full = None
        self.grad_x_train = []

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_train_full: Optional[np.ndarray] = None,
            y_train_full: Optional[np.ndarray] = None) -> "Explainer":

        self.x_train = x_train
        self.y_train = y_train
        self.x_train_full = x_train if x_train_full is None else x_train_full
        self.y_train_full = y_train if y_train_full is None else y_train_full

        # compute and store gradients
        if self.store_grads:
            for i in tqdm(range(x_train.shape[0])):
                x = self.backend.to_tensor(x_train[i:i + 1])
                y = self.backend.to_tensor(y_train[i:i + 1])
                grad_x_train = self.backend.get_grads(self.model, x, y, self.loss_fn)
                self.grad_x_train.append(grad_x_train)
            self.grad_x_train = np.concatenate(self.grad_x_train, axis=0)
        return self

    def explain(self, x: Any) -> "Explanation":
        return Explanation(meta={'params': ''}, data={})
