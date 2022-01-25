from alibi.explainers.similarity.base import BaseSimilarityExplainer
from alibi.explainers.similarity.metrics import dot, cos, asym_dot
from typing import TYPE_CHECKING, Any, Callable, Optional, Union
from alibi.api.interfaces import Explanation
import numpy as np

if TYPE_CHECKING:
    import tensorflow
    import torch


class SimilarityExplainer(BaseSimilarityExplainer):
    def __init__(self,
                 model: 'Union[tensorflow.keras.Model, torch.nn.Module]',
                 loss_fn: '''Callable[[Union[tensorflow.Tensor, torch.Tensor],
                                       Union[tensorflow.Tensor, torch.Tensor]],
                                       Union[tensorflow.Tensor, torch.Tensor]]''',
                 sim_fn: str = 'grad_dot',
                 task: str = "classification",
                 store_grads: bool = False,
                 seed: int = 0,
                 backend: str = "tensorflow",
                 **kwargs
                 ):

        sim_fn_opts = {
            'grad_dot': dot,
            'grad_cos': cos,
            'grad_asym_dot': asym_dot
        }

        if sim_fn not in sim_fn_opts.keys():
            raise ValueError(f'Unknown method {sim_fn}. Consider using: `{"` | `".join(sim_fn_opts.keys())}`.')

        sim_fn = sim_fn_opts[sim_fn]

        if task not in ['classification', 'regression']:
            raise ValueError(f'Unknown task {task}. Consider using: `classification` | `regression`.')

        self.task = task

        super().__init__(model, loss_fn, sim_fn, store_grads, seed, backend, **kwargs)

    def explain(self, x: np.ndarray, y: Optional[Union[np.ndarray, Callable]] = None) -> "Explanation":
        if self.task == 'regression' and y is None:
            raise ValueError('Regression task requires a target value.')

        if not y:
            y = self.backend.to_numpy(self.model(x))
            y = np.argmax(y)
        elif callable(y):
            y = y(x)

        grad_x_test = self.backend.get_grads(self.model, x, y, self.loss_fn)
        if not self.store_grads:
            scores = self.compute_adhoc_similarity(grad_x_test)
        else:
            scores = self.sim_fn(self.grad_x_train, grad_x_test)
        return Explanation(meta={'params': ''}, data={'scores': scores})

