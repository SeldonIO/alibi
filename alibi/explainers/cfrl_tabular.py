from alibi.api.interfaces import Explainer, Explanation
from alibi.utils.frameworks import has_pytorch, has_tensorflow
from alibi.explainers.cfrl_base import CounterfactualRL, Postprocessing, _PARAM_TYPES
from alibi.explainers.backends.cfrl_tabular import sample, get_conditional_vector, get_statistics

import numpy as np
from tqdm import tqdm
from itertools import count
from functools import partial
from typing import Tuple, List, Dict, Callable, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import tensorflow

if has_pytorch:
    # import pytorch backend
    from alibi.explainers.backends.pytorch import cfrl_tabular as pytorch_tabular_backend

if has_tensorflow:
    # import tensorflow backend
    from alibi.explainers.backends.tensorflow import cfrl_tabular as tensorflow_tabular_backend


class SampleTabularPostprocessing(Postprocessing):
    """
    Tabular sampling post-processing. Given the output of the heterogeneous autoencoder the post-processing
    functions samples the output according to the conditional vector. Note that the original input instance
    is required to perform the conditional sampling.
    """

    def __init__(self,  category_map: Dict[int, List[str]], stats: Dict[int, Dict[str, float]]):
        """
        Constructor.

        Parameters
        ----------
        category_map
            Dictionary of category mapping. The keys are column indexes and the values are lists containing the
            possible feature values.
        stats
            Dictionary of statistic of the training data. Contains the minimum and maximum value of each numerical
            feature in the training set. Each key is an index of the column and each value is another dictionary
            containing `min` and `max` keys.
        """
        super().__init__()
        self.category_map = category_map
        self.stats = stats

    def __call__(self, X_cf: List[np.ndarray], X: np.ndarray, C: Optional[np.ndarray]) -> List[np.ndarray]:
        """
        Performs counterfactual conditional sampling acording to the conditional vector and the original input.

        Parameters
        ----------
        X_cf
            Decoder reconstruction of the counterfactual instance. The decoded instance is a list where each
            element in the list correspond to the reconstruction of a feature.
        X
            Input instance.
        C
            Conditional vector.

        Returns
        -------
            Conditional sampled counterfactual instance.
        """
        return sample(X_hat_split=X_cf,
                      X_ohe=X,
                      C=C,
                      stats=self.stats,
                      category_map=self.category_map)


class ConcatTabularPostprocessing(Postprocessing):
    """ Tabular feature columns concatenation post-processing. """

    def __call__(self, X_cf: List[np.ndarray], X: np.ndarray, C: Optional[np.ndarray]) -> np.ndarray:
        """
        Performs a concatenation of the counterfactual feature columns along the axis 1.

        Parameters
        ----------
        X_cf
            List of counterfactual feature columns.
        X
            Input instance. Not used. Included for consistency.
        C
            Conditional vector. Not used. Included for consistency.

        Returns
        -------
            Concatenation of the counterfactual feature columns.
        """
        return np.concatenate(X_cf, axis=1)


# update parameter types for the tabular case
_PARAM_TYPES["complex"] += ["conditional_vector", "stats"]


class CounterfactualRLTabular(CounterfactualRL):
    """ Counterfactual Reinforcement Learning Tabular. """

    def __init__(self,
                 predictor: Callable,
                 encoder: 'Union[tensorflow.keras.Model, torch.nn.Module]',
                 decoder: 'Union[tensorflow.keras.Model, torch.nn.Module]',
                 encoder_preprocessor: Callable,
                 decoder_inv_preprocessor: Callable,
                 coeff_sparsity: float,
                 coeff_consistency: float,
                 feature_names: List[str],
                 category_map: Dict[int, List[str]],
                 immutable_features: Optional[List[str]] = None,
                 ranges: Optional[Dict[str, Tuple[int, int]]] = None,
                 weight_num: float = 1.0,
                 weight_cat: float = 1.0,
                 latent_dim: Optional[int] = None,
                 backend: str = "tensorflow",
                 seed: int = 0,
                 **kwargs):
        """
        Constructor.

        Parameters
        ----------
        predictor.
            A callable that takes a tensor of N data points as inputs and returns N outputs. For classification task,
            the second dimension of the output should match the number of classes. Thus, the output can be either
            a soft label distribution or a hard label distribution (i.e. one-hot encoding) without affecting the
            performance since `argmax` is applied to the predictor's output.
        encoder
            Pretrained heterogeneous encoder network.
        decoder
            Pretrained heterogeneous decoder network. The output of the decoder must be a list of tensors.
        encoder_preprocessor
            Autoencoder data pre-processor. Depending on the input format, the pre-processor can normalize
            numerical attributes, transform label encoding to one-hot encoding etc.
        decoder_inv_preprocessor
            Autoencoder data inverse pre-processor. This is the invers function of the pre-processor. It can
            denormalize numerical attributes, transfrom one-hot encoding to label encoding, feature type casting etc.
        coeff_sparsity
           Sparsity loss coefficient.
        coeff_consistency
           Consistency loss coefficient.
        feature_names
            List of feature names. This should be provided by the dataset.
        category_map
            Dictionary of category mapping. The keys are column indexes and the values are lists containing the possible
            values for a feature. This should be provided by the dataset.
        immutable_features
            List of immutable features.
        ranges
            Numerical feature ranges. Note that exist numerical features such as `Age`, which are  allowed to increase
            only. We denote those by `inc_feat`. Similarly, there exist features  allowed to decrease only. We denote
            them by `dec_feat`. Finally, there are some free feature, which we denote by `free_feat`. With the previous
            notation, we can define `range = {'inc_feat': [0, 1], 'dec_feat': [-1, 0], 'free_feat': [-1, 1]}`.
            `free_feat` can be omitted, as any unspecified feature is considered free. Having the ranges of a feature
            `{'feat': [a_low, a_high}`, when sampling is performed the numerical value will be clipped between
            `[a_low * (max_val - min_val), a_high * [max_val - min_val]]`, where `a_low` and `a_high` are the minimum
            and maximum values the feature `feat`. This implies that `a_low` and `a_high` are not restricted to {-1, 0}
            and {0, 1}, but can be any float number in-between `[-1, 0]` and `[0, 1]`.
        weight_num
            Numerical loss weight.
        weight_cat
            Categorical loss weight.
        latent_dim
            Autoencoder latent dimension. Can be omitted if the actor network is user specified.
        backend
           Deep learning backend: `tensorflow` | `pytorch`. Default `tensorflow`.
        seed
            Seed for reproducibility. The results are not reproducible for `tensorflow` backend.
        kwargs
            Used to replace any default parameter from :py:data:`alibi.explainers.cfrl_base.DEFAULT_BASE_PARAMS`.
        """
        super().__init__(encoder=encoder, decoder=decoder, latent_dim=latent_dim, predictor=predictor,
                         coeff_sparsity=coeff_sparsity, coeff_consistency=coeff_consistency, backend=backend,
                         seed=seed, **kwargs)

        # Set encoder preprocessor and decoder inverse preprocessor.
        self.params["encoder_preprocessor"] = encoder_preprocessor
        self.params["decoder_inv_preprocessor"] = decoder_inv_preprocessor

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
            self.params["conditional_vector"] = partial(get_conditional_vector,
                                                        preprocessor=self.params["encoder_preprocessor"],
                                                        feature_names=self.params["feature_names"],
                                                        category_map=self.params["category_map"],
                                                        ranges=self.params["ranges"],
                                                        immutable_features=self.params["immutable_features"])

        # update metadata
        self.meta["params"].update(CounterfactualRLTabular._serialize_params(self.params))

    def _select_backend(self, backend, **kwargs):
        """
        Selects the backend according to the `backend` flag.

        Parameters
        ----------
        backend
            Deep learning backend. `tensorflow` | `pytorch`. Default `tensorflow`.
        """
        return tensorflow_tabular_backend if backend == "tensorflow" else pytorch_tabular_backend

    def _validate_input(self, X: np.ndarray):
        """
        Validates the input instances by checking the appropriate dimensions.

        Parameters
        ----------
        X
            Input instances.
        """
        if len(X.shape) != 2:
            raise ValueError(f"The input should be a 2D array. Found {len(X.shape)}D instead.")

        # Check if the number of features matches the expected one.
        if X.shape[1] != len(self.params["feature_names"]):
            raise ValueError(f"Unexpected number of features. The expected number "
                             f"is {len(self.params['feature_names'])}, but the input has {X.shape[1]} features.")

        return X

    def fit(self, X: np.ndarray) -> 'Explainer':
        # Compute vector of statistics to clamp numerical values between the minimum and maximum
        # value from the training set.
        self.params["stats"] = get_statistics(X=X,
                                              preprocessor=self.params["encoder_preprocessor"],
                                              category_map=self.params["category_map"])

        # Set postprocessing functions. Needs `stats`.
        self.params["postprocessing_funcs"] = [
            SampleTabularPostprocessing(stats=self.params["stats"], category_map=self.params["category_map"]),
            ConcatTabularPostprocessing(),
        ]

        # update metadata
        self.meta["params"].update(CounterfactualRLTabular._serialize_params(self.params))

        # validate dataset
        self._validate_input(X)

        # call base class fit
        return super().fit(X)

    def explain(self,
                X: np.ndarray,
                Y_t: np.ndarray = None,  # TODO remove default value (mypy error)
                C: Optional[List[Dict[str, List[Union[str, float]]]]] = None,
                batch_size: int = 100,
                diversity: bool = False,
                num_samples: int = 1,
                patience: int = 1000,
                tolerance: float = 1e-3) -> Explanation:

        """
        Computes counterfactuals for the given instances conditioned on the target and the conditional vector.

        Parameters
        ----------
        X
            Input instances to generate counterfactuals for.
        Y_t
            Target labels.
        C
            List of conditional dictionaries. If `None`, it means that no conditioning was used during training
            (i.e. the `conditional_func` returns `None`). If conditioning was used during training but no conditioning
            is desired for the current input, an empty list is expected.
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
        # General validation.
        self._validate_input(X)
        self._validate_target(Y_t)

        # Check if diversity flag is on.
        if diversity:
            return self._diversity(X=X,
                                   Y_t=Y_t,
                                   C=C,
                                   num_samples=num_samples,
                                   batch_size=batch_size,
                                   patience=patience,
                                   tolerance=tolerance)

        # Get conditioning for a zero input. Used for a sanity check of the user-specified conditioning.
        X_zeros = np.zeros((1, X.shape[1]))
        C_zeros = self.params["conditional_func"](X_zeros)

        # If the conditional vector is `None`. This is equivalent of no conditioning at all, not even during training.
        if C is None:
            # Check if the conditional function actually a `None` conditioning
            if C_zeros is not None:
                raise ValueError("A `None` conditioning is not a valid input when training with conditioning. "
                                 "If no feature conditioning is desired for the given input, `C` is expected to be an "
                                 "empty list. A `None` conditioning is valid only when no conditioning was used "
                                 "during training (i.e. `conditional_func` returns `None`).")

            return super().explain(X=X, Y_t=Y_t, C=C, batch_size=batch_size)

        elif C_zeros is None:
            raise ValueError("Conditioning different than `None` is not a valid input when training without "
                             "conditioning. If feature conditioning is desired, consider defining an appropriate "
                             "`conditional_func` that does not return `None`.")

        # Define conditional vector if an empty list. This is equivalent of no conditioning, but the conditional
        # vector was used during training.
        if len(C) == 0:
            C = [dict()]

        # Check the number of conditions.
        if len(C) != 1 and len(C) != X.shape[0]:
            raise ValueError("The number of conditions should be 1 or equals the number of samples in x.")

        # If only one condition is passed.
        if len(C) == 1:
            C_vec = self.params["conditional_vector"](X=X,
                                                      condition=C[0],
                                                      stats=self.params["stats"])
        else:
            # If multiple conditions were passed.
            C_vecs = []

            for i in range(len(C)):
                # Generate conditional vector for each instance. Note that this depends on the input instance.
                C_vecs.append(self.params["conditional_vector"](X=np.atleast_2d(X[i]),
                                                                condition=C[i],
                                                                stats=self.params["stats"]))

            # Concatenate all conditional vectors.
            C_vec = np.concatenate(C_vecs, axis=0)

        explanation = super().explain(X=X, Y_t=Y_t, C=C_vec, batch_size=batch_size)
        explanation.data.update({"condition": C})
        return explanation

    def _diversity(self,
                   X: np.ndarray,
                   Y_t: np.ndarray,
                   C: Optional[List[Dict[str, List[Union[str, float]]]]],
                   num_samples: int = 1,
                   batch_size: int = 100,
                   patience: int = 1000,
                   tolerance: float = 1e-3) -> Explanation:
        """
        Generates a set of diverse counterfactuals given a single instance, target and conditioning.

        Parameters
        ----------
        X
            Input instance.
        Y_t
            Target label.
        C
            List of conditional dictionaries. If `None`, it means that no conditioning was used during training
            (i.e. the `conditional_func` returns `None`).
        num_samples
            Number of counterfactual samples to be generated.
        batch_size
            Batch size used at inference.
        num_samples
            Number of diversity samples to be generated. Considered only if `diversity=True`.
        batch_size
            Batch size to use when generating counterfactuals.
        patience
            Maximum number of iterations to perform diversity search stops. If -1, the search stops only if
            the desired number of samples has been found.
        tolerance
            Tolerance to distinguish two counterfactual instances.

        Returns
        -------
            Explanation object containing the diverse counterfactuals.
        """
        # Check if condition. If no conditioning was used during training, the method can not generate a diverse
        # set of counterfactual instances
        if C is None:
            raise ValueError("A diverse set of counterfactual can not be generated if a `None` conditioning is "
                             "used during training. Use the `explain` method to generate a counterfactual. The "
                             "generation process is deterministic in its core. If conditioning is used during training "
                             "a diverse set of counterfactual can be generated by restricting each feature condition "
                             "to a subset to remain feasible.")

        # Check the number of inputs
        if X.shape[0] != 1:
            raise ValueError("Only a single input instance can be passed.")

        # Check the number of labels.
        if Y_t.shape[0] != 1:
            raise ValueError("Only a single label can be passed.")

        # Check the number of conditions.
        if (C is not None) and len(C) > 1:
            raise ValueError("At most, one condition can be passed.")

        # Generate a batch of data.
        X_repeated = np.tile(X, (batch_size, 1))
        Y_t = np.tile(np.atleast_2d(Y_t), (batch_size, 1))

        # Define counterfactual buffer.
        X_cf_buff = None

        for i in tqdm(count()):
            if i == patience:
                break

            if (X_cf_buff is not None) and (X_cf_buff.shape[0] >= num_samples):
                break

            # Generate conditional vector.
            C_vec = get_conditional_vector(X=X_repeated,
                                           condition=C[0] if len(C) else {},
                                           preprocessor=self.params["encoder_preprocessor"],
                                           feature_names=self.params["feature_names"],
                                           category_map=self.params["category_map"],
                                           stats=self.params["stats"],
                                           immutable_features=self.params["immutable_features"],
                                           diverse=True)

            # Generate counterfactuals.
            results = self._compute_counterfactual(X=X_repeated, Y_t=Y_t, C=C_vec)
            X_cf, Y_m_cf, Y_t = results["X_cf"], results["Y_m_cf"], results["Y_t"]

            # Select only counterfactuals where prediction matches the target.
            X_cf = X_cf[Y_t == Y_m_cf]
            if X_cf.shape[0] == 0:
                continue

            # Find unique counterfactuals.
            _, indices = np.unique(np.floor(X_cf / tolerance).astype(int), return_index=True, axis=0)

            # Add them to the unique buffer but make sure not to add duplicates.
            if X_cf_buff is None:
                X_cf_buff = X_cf[indices]
            else:
                X_cf_buff = np.concatenate([X_cf_buff, X_cf[indices]], axis=0)
                _, indices = np.unique(np.floor(X_cf_buff / tolerance).astype(int), return_index=True, axis=0)
                X_cf_buff = X_cf_buff[indices]

        # Construct counterfactuals to the explanation.
        X_cf = X_cf_buff[:num_samples] if (X_cf_buff is not None) else np.array([])

        # Compute model's prediction on the counterfactual instances
        Y_m_cf = self.params["predictor"](X_cf) if X_cf.shape[0] != 0 else np.array([])
        if self._is_classification(pred=Y_m_cf):
            Y_m_cf = np.argmax(Y_m_cf, axis=1)

        # Compute model's prediction on the original input.
        Y_m = self.params["predictor"](X)
        if self._is_classification(Y_m):
            Y_m = np.argmax(Y_m, axis=1)

        # Update target representation if necessary.
        if self._is_classification(Y_t):
            Y_t = np.argmax(Y_t, axis=1)

        return self._build_explanation(X=X, Y_m=Y_m, X_cf=X_cf, Y_m_cf=Y_m_cf, Y_t=Y_t, C=C)
