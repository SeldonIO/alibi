import numpy as np
from typing import Callable, Optional, Tuple, Union, TYPE_CHECKING
import tensorflow as tf
import logging

from captum.attr import IntegratedGradients, LayerConductance, NeuronConductance
import torch
#import torch.nn as nn
import pandas as pd
from scipy import stats

from alibi.utils.gradients import num_grad_batch
from alibi.utils.tf import _check_keras_or_tf

if TYPE_CHECKING:  # pragma: no cover
    import keras  # noqa

logger = logging.getLogger(__name__)


def _define_model(model):
    return model


class IntegratedGradients:

    def __init__(self, model, **kwargs):
        """

        Parameters
        ----------
        model
        kwargs
        """
        self.ig = IntegratedGradients(model, **kwargs)

    def explain(self, X,
                baselines = None,
                target = None,
                additional_forward_args = None,
                n_steps = 50,
                method = "gausslegendre",
                internal_batch_size = None,
                return_convergence_delta = False
                ):
        """

        Parameters
        ----------
        X
        baselines
        target
        additional_forward_args
        n_steps
        method
        internal_batch_size
        return_convergence_delta

        Returns
        -------

        """
        test_input_tensor = torch.from_numpy(X).type(torch.FloatTensor)
        test_input_tensor.requires_grad_()
        if return_convergence_delta:
            attr, delta = self.ig.attribute(test_input_tensor,
                                            baselines=baselines,
                                            target=target,
                                            additional_forward_args=additional_forward_args,
                                            n_steps=n_steps,
                                            method=method,
                                            internal_batch_size=internal_batch_size,
                                            return_convergence_delta=return_convergence_delta)
        else:
            attr = self.ig.attribute(test_input_tensor,
                                     baselines=baselines,
                                     target=target,
                                     additional_forward_args=additional_forward_args,
                                     n_steps=n_steps,
                                     method=method,
                                     internal_batch_size=internal_batch_size,
                                     return_convergence_delta=return_convergence_delta)

        attr = attr.detach().numpy()

        return attr
