from alibi.utils.frameworks import has_pytorch, has_tensorflow
from alibi.explainers.cfrl_base import CounterfactualRLBase, Postprocessing
from alibi.explainers.backends.cfrl_tabular import sample, conditional_vector, statistics
from alibi.api.interfaces import Explainer, Explanation

import numpy as np
from itertools import count
from functools import partial
from typing import Tuple, List, Dict, Callable, Union, Optional

if has_pytorch:
    # import pytorch backend
    import alibi.explainers.backends.pytorch.cfrl_tabular as pytorch_tabular_backend

if has_tensorflow:
    # import tensorflow backend
    import alibi.explainers.backends.tflow.cfrl_tabular as tensorflow_tabular_backend


class SamplePostprocessing(Postprocessing):
    def __init__(self,  category_map: Dict[int, List[str]], stats: Dict[int, Dict[str, float]]):
        super().__init__()
        self.category_map = category_map
        self.stats = stats

    def __call__(self, x_cf: List[np.ndarray], x: np.ndarray, c: np.ndarray):
        return sample(x_hat_split=x_cf,
                      x_ohe=x,
                      cond=c,
                      stats=self.stats,
                      category_map=self.category_map)


class ConcatPostprocessing(Postprocessing):
    def __call__(self, x_cf: List[np.ndarray], x: np.ndarray, c: np.ndarray):
        return np.concatenate(x_cf, axis=1)


class CounterfactualRLTabular(CounterfactualRLBase):
    """ Counterfactual Reinforcement Learning Tabular. """

    def __init__(self,
                 ae,
                 latent_dim: int,
                 ae_preprocessor: Callable,
                 ae_inv_preprocessor: Callable,
                 predict_func: Callable,
                 coeff_sparsity: float,
                 coeff_consistency: float,
                 num_classes: int,
                 category_map: Dict[int, List[str]],
                 feature_names: List[str],
                 ranges: Optional[Dict[str, Tuple[int, int]]] = None,  # TODO: infer it (make it optional)
                 immutable_features: Optional[List[str]] = None,
                 backend: str = "tensorflow",
                 weight_num: float = 1.0,
                 weight_cat: float = 1.0,
                 **kwargs):
        """
        Constructor.

        Parameters
        ----------
        ae
           Pre-trained autoencoder.
        actor
           Actor network.
        critic
           Critic network.
        predict_func.
           Prediction function. This corresponds to the classifier.
        coeff_sparsity
           Sparsity loss coefficient.
        coeff_consistency
           Consistency loss coefficient.
        backend
           Deep learning backend: `tensorflow`|`pytorch`. Default `tensorflow`.
        weight_num
            Numerical loss weight.
        weight_cat
            Categorical loss weight.
        """
        super().__init__(ae=ae, latent_dim=latent_dim, predict_func=predict_func, coeff_sparsity=coeff_sparsity,
                         coeff_consistency=coeff_consistency, num_classes=num_classes, backend=backend, **kwargs)

        # Set ae preprocessor and inverse preprocessor.
        self.params["ae_preprocessor"] = ae_preprocessor
        self.params["ae_inv_preprocessor"] = ae_inv_preprocessor

        # Set dataset specific arguments.
        self.params["category_map"] = category_map
        self.params["feature_names"] = feature_names
        self.params["ranges"] = ranges if (ranges is not None) else dict()
        self.params["immutable_features"] = immutable_features if (immutable_features is not None) else list()
        self.params["weight_num"] = weight_num
        self.params["weight_cat"] = weight_cat

        # Set sparsity loss if not user-specified.
        if "sparsity_loss" not in kwargs:
            self.params["sparsity_loss"] = partial(self.backend.sparsity_loss,
                                                   category_map=self.params["category_map"],
                                                   weight_num=weight_num,
                                                   weight_cat=weight_cat)

        # Set consistency loss if not user-specified.
        if "consistency_loss" not in kwargs:
            self.params["consistency_loss"] = self.backend.consistency_loss

        # Set training conditional function generator if not user-specified.
        if "conditional_func" not in kwargs:
            self.params["conditional_func"] = partial(self.backend.generate_condition,
                                                      feature_names=self.params["feature_names"],
                                                      category_map=self.params["category_map"],
                                                      ranges=self.params["ranges"],
                                                      immutable_features=self.params["immutable_features"])

        # Set testing conditional function generator if not user-specified.
        if "conditional_vector" not in kwargs:
            self.params["conditional_vector"] = partial(conditional_vector,
                                                        preprocessor=self.params["ae_preprocessor"],
                                                        feature_names=self.params["feature_names"],
                                                        category_map=self.params["category_map"],
                                                        ranges=self.params["ranges"],
                                                        immutable_features=self.params["immutable_features"])

    def _select_backend(self, backend, **kwargs):
        """
        Selects the backend according to the `backend` flag.

        Parameters
        ----------
        backend
            Deep learning backend. `tensorflow`|`pytorch`. Default `tensorflow`.
        """
        return tensorflow_tabular_backend if backend == "tensorflow" else pytorch_tabular_backend

    def fit(self, x: np.ndarray) -> 'Explainer':
        # Compute vector of statistics to clamp numerical values between the minimum and maximum
        # value from the training set.
        self.params["stats"] = statistics(x=x,
                                          preprocessor=self.params["ae_preprocessor"],
                                          category_map=self.params["category_map"])

        # Set postprocessing functions. Needs `stats`.
        self.params["postprocessing_funcs"] = [
            SamplePostprocessing(stats=self.params["stats"], category_map=self.params["category_map"]),
            ConcatPostprocessing(),
        ]

        return super().fit(x)

    def explain(self,
                x: np.ndarray,
                y_t: np.ndarray,
                c: Optional[List[Dict[str, List[Union[str, int]]]]] = None,
                diversity: bool = False,
                num_samples: int = 1,
                batch_size: int = 100,
                patience: int = 1000,
                tolerance: float = 1e-3) -> "Explanation":

        """
        Computes counterfactuals for the given instances conditioned on the target and the conditional vector.

        Parameters
        ----------
        x
            Input instances to generate counterfactuals for.
        y_t
            Target labels.
        c
            List of conditional dictionaries.
        diversity
            Whether to generate diverse counterfactual set for the given instance. Only supported for a single
            input instance.
        num_samples
            Number of diversity samples to be generated. Considered only if `diversity=True`.
        batch_size
            Batch size to use when generating counterfactuals.
        patience
            Maximum number of iterations to perform diversity search stops. If -1, the search stops only if
            the desired number of samples has been found.
        tolerance
            Tolerance to distinguish two counterfactual instances.
        """
        if len(x.shape) > 2:
            raise ValueError("The input should be a 2D array.")

        # Reshape the input vector to have be 2D.
        x = np.atleast_2d(x)

        # Check if the number of features matches the expected one.
        if x.shape[1] != len(self.params["feature_names"]):
            raise ValueError(f"Unexpected number of features. The expected number "
                             f"is {len(self.params['feature_names'])}, but the input has {x.shape[1]} features.")

        # Check if diversity flag is on.
        if diversity:
            return self._diversity(x=x,
                                   y_t=y_t,
                                   c=c,
                                   num_samples=num_samples,
                                   batch_size=batch_size,
                                   patience=patience,
                                   tolerance=tolerance)

        # Check the number of target labels.
        if y_t.shape[0] != 1 and y_t.shape[0] != x.shape[0]:
            raise ValueError("The number target labels should be 1 or equals the number of samples in x.")

        # Repeat the same label to match the number of input instances.
        if y_t.shape[0] == 1:
            y_t = np.tile(y_t, x.shape[0])

        # Define conditional vector if `None`.
        if c is None:
            c = [dict()]

        # Check the number of conditions.
        if len(c) != 1 and len(c) != x.shape[0]:
            raise ValueError("The number of conditions should be 1 or equals the number of samples in x.")

        # If only one condition is passed.
        if len(c) == 1:
            c_vec = self.params["conditional_vector"](x=x,
                                                      condition=c[0],
                                                      stats=self.params["stats"])
        else:
            # If multiple conditions were passed.
            c_vecs = []

            for i in range(len(c)):
                # Generate conditional vector for each instance. Note that this depends on the input instance.
                c_vecs.append(self.params["conditional_vector"](x=np.atleast_2d(x[i]),
                                                                condition=c[i],
                                                                stats=self.params["stats"]))

            # Concatenate all conditional vectors.
            c_vec = np.concatenate(c_vecs, axis=0)
        return super().explain(x, y_t, c_vec)

    def _diversity(self,
                   x: np.ndarray,
                   y_t: np.ndarray,
                   c: Optional[List[Dict[str, List[Union[str, int]]]]],
                   num_samples: int = 1,
                   batch_size: int = 100,
                   patience: int = 1000,
                   tolerance: float = 1e-3) -> np.ndarray:

        # Reshape input array.
        x = x.reshape(1, -1)

        # Check the number of labels.
        if y_t.shape[0] != 1:
            raise ValueError("Only a single label can be passed.")

        # Check the number of conditions.
        if len(c) != 1:
            raise ValueError("Only a single condition can be passed")

        # Generate a batch of data.
        x = np.tile(x, (batch_size, 1))
        y_t = np.tile(y_t, batch_size)

        # Define counterfactual buffer.
        cf_buff = None

        for i in count():
            if i == patience:
                break

            if (cf_buff is not None) and (cf_buff.shape[0] >= num_samples):
                break

            # Generate conditional vector.
            c_vec = conditional_vector(x=x,
                                       condition=c[0],
                                       preprocessor=self.params["ae_preprocessor"],
                                       feature_names=self.params["feature_names"],
                                       category_map=self.params["category_map"],
                                       stats=self.params["stats"],
                                       immutable_features=self.params["immutable_features"],
                                       diverse=True)

            # Generate counterfactuals.
            x_cf = super().explain(x, y_t, c_vec)

            # Get prediction.
            y_cf_m = self.params["predict_func"](x_cf)

            # Select only counterfactuals where prediction matches the target.
            x_cf = x_cf[y_t == y_cf_m]
            if x_cf.shape[0] == 0:
                continue

            # Find unique counterfactuals.
            x_cf = np.unique(np.floor(x_cf / tolerance).astype(int), axis=0) * tolerance

            # Add them to the unique buffer but make sure not to add duplicates.
            if cf_buff is None:
                cf_buff = x_cf
            else:
                cf_buff = np.concatenate([cf_buff, x_cf], axis=0)
                cf_buff = np.unique(np.floor(cf_buff / tolerance).astype(int), axis=0) * tolerance

        # TODO construct explanation
        return cf_buff[:num_samples] if (cf_buff is not None) else np.array([])
