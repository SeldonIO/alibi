import torch
import torch.nn as nn
import torch.nn.functional as F

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from alibi.explainers.cfrl_base import CounterfactualRLBase, Postprocessing
from alibi.models.pytorch.autoencoder import AE as PytorchAE
from alibi.models.tensorflow.autoencoder import AE as TensorflowAE
from alibi.explainers.backends.cfrl_tabular import CounterfactualRLTabularBackend

import numpy as np
from itertools import count
from functools import partial
from typing import Tuple, List, Dict, Callable, Union


class SamplePostprocessing(Postprocessing):
    def __init__(self,  category_map: Dict[int, List[str]], stats: Dict[int, Dict[str, float]]):
        super().__init__()
        self.category_map = category_map
        self.stats = stats

    def __call__(self, x_cf: List[np.ndarray], x: np.ndarray, c: np.ndarray):
        return CounterfactualRLTabularBackend.sample(x_hat_split=x_cf,
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
                 ae: Union[TensorflowAE, PytorchAE],
                 ae_preprocessor: Callable,
                 ae_inv_preprocessor: Callable,
                 actor: Union[keras.Sequential, nn.Sequential],
                 critic: Union[keras.Sequential, nn.Sequential],
                 predict_func: Callable,
                 coeff_sparsity: float,
                 coeff_consistency: float,
                 num_classes: int,
                 category_map: Dict[int, List[str]],
                 feature_names: List[str],
                 stats: Dict[int, Dict[str, float]],  # TODO: infer it in the constructor
                 ranges: Dict[str, Tuple[int, int]],  # TODO: infer it (make it optional)
                 immutable_attr: List[str],           # TODO: ren
                 attr_types: Dict[int, type],         # TODO: delete this
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
        super().__init__(ae=ae, actor=actor, critic=critic, predict_func=predict_func, coeff_sparsity=coeff_sparsity,
                         coeff_consistency=coeff_consistency, num_classes=num_classes, backend=backend, **kwargs)

        # Set ae preprocessor and inverse preprocessor.
        self.params["ae_preprocessor"] = ae_preprocessor
        self.params["ae_inv_preprocessor"] = ae_inv_preprocessor

        # Set dataset specific arguments.
        self.params["category_map"] = category_map
        self.params["feature_names"] = feature_names
        self.params["ranges"] = ranges
        self.params["immutable_attr"] = immutable_attr
        self.params["stats"] = stats
        self.params["attr_types"] = attr_types
        self.params["weight_num"] = weight_num
        self.params["weight_cat"] = weight_cat

        # Set postprocessing functions.
        self.params["postprocessing_funcs"] = [
            SamplePostprocessing(stats=self.params["stats"], category_map=self.params["category_map"]),
            ConcatPostprocessing(),
        ]

        if "sparsity_loss" not in kwargs:
            # select backend specific function
            he_sparsity_loss = tensorflow_he_sparsity_loss \
                if self.params["backend"] == CounterfactualRLBase.TENSORFLOW \
                else pytorch_he_sparsity_loss

            # set sparsity loss
            self.params["sparsity_loss"] = partial(he_sparsity_loss,
                                                   category_map=self.params["category_map"],
                                                   weight_num=weight_num,
                                                   weight_cat=weight_cat)

        if "consistency_loss" not in kwargs:
            # select backend specific function
            self.params["consistency_loss"] = tensorflow_he_consistency_loss \
                if self.params["backend"] == CounterfactualRLBase.TENSORFLOW \
                else pytorch_he_consistency_loss

        if "conditional_func" not in kwargs:
            # set conditional function generator
            self.params["conditional_func"] = partial(generate_condition,
                                                      feature_names=self.params["feature_names"],
                                                      category_map=self.params["category_map"],
                                                      ranges=self.params["ranges"],
                                                      immutable_attr=self.params["immutable_attr"])

    def explain(self, x: np.ndarray, y_t: np.ndarray, c: Dict[str, List[Union[str, int]]]) -> "Explanation":
        # TODO: check if `c` can be optional
        # TODO: extend --- a condition for each instance


        # convert dictionary conditioning to array
        c = conditional_vector(x=x,
                               condition=c,
                               preprocessor=self.params["ae_preprocessor"],
                               feature_names=self.params["feature_names"],
                               category_map=self.params["category_map"],
                               stats=self.params["stats"],
                               ranges=self.params["ranges"],
                               immutable_attr=self.params["immutable_attr"])
        return super().explain(x, y_t, c)


    # TODO: try to include this in the explain method, with a flag
    def diversity(self,
                  x: np.ndarray,
                  y_t: np.ndarray,
                  c: Dict[str, List[Union[str, int]]],
                  num_samples: int = 1,
                  batch_size: int = 100,
                  patience: int = 1000,
                  tolerance: float = 1e-3) -> np.ndarray:

        # reshape input array
        x = x.reshape(1, -1)

        if x.shape[1] != len(self.params["feature_names"]):
            raise ValueError("Only a single input can be passed.")

        if len(y_t) != 1:
            raise ValueError("Only a single label can be passed.")

        # generate a batch of data
        x = np.tile(x, (batch_size, 1))
        y_t = np.tile(y_t, batch_size)

        # define counterfactual buffer
        cf_buff = None

        from tqdm import tqdm
        for i in tqdm(count()):
            if i == patience:
                break

            if (cf_buff is not None) and (cf_buff.shape[0] >= num_samples):
                break

            # generate conditional vector
            c_vec = conditional_vector(x=x,
                                       condition=c,
                                       preprocessor=self.params["ae_preprocessor"],
                                       feature_names=self.params["feature_names"],
                                       category_map=self.params["category_map"],
                                       stats=self.params["stats"],
                                       immutable_attr=self.params["immutable_attr"],
                                       diverse=True)

            # generate counterfactuals
            x_cf = super().explain(x, y_t, c_vec)

            # get prediction
            y_cf_m = self.params["predict_func"](x_cf)

            # select only counterfactuals where prediction matches the target
            x_cf = x_cf[y_t == y_cf_m]
            if x_cf.shape[0] == 0:
                continue

            # find unique counterfactuals
            x_cf = np.unique(np.floor(x_cf / tolerance).astype(int), axis=0) * tolerance

            # add them to the unique buffer but make sure not to add duplicates
            if cf_buff is None:
                cf_buff = x_cf
            else:
                cf_buff = np.concatenate([cf_buff, x_cf], axis=0)
                cf_buff = np.unique(np.floor(cf_buff / tolerance).astype(int), axis=0) * tolerance

        return cf_buff[:num_samples] if (cf_buff is not None) else np.array([])


# if __name__ == "__main__":
#     from alibi.datasets import fetch_adult
#
#     adult = fetch_adult()
#     x = adult.data
#     category_map = adult.category_map
#     feature_names = adult.feature_names
#     preprocessor, inv_preprocessor = he_preprocessor(x, feature_names, category_map)
#     stats = statistics(x, preprocessor, category_map)
#
#
#     print(feature_names)
#     print(x[0])
#     print(category_map[feature_names.index('Workclass')])
#
#     condition = {"Age": [-5, 20], "Workclass": ["State-gov", "?", "Local-gov"]}
#     immutable_attr = ['Education', 'Marital Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain',
#                       'Capital Loss', 'Hours per week', 'Country']
#
#     c = conditional_vector(x=x[:10],
#                            condition=condition,
#                            preprocessor=preprocessor,
#                            feature_names=feature_names,
#                            category_map=category_map,
#                            stats=stats,
#                            immutable_attr=immutable_attr)
#
#     print(c[:2])
