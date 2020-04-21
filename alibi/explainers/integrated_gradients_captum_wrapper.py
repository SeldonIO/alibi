from typing import TYPE_CHECKING
import logging

from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
import torch

if TYPE_CHECKING:  # pragma: no cover
    import keras  # noqa

logger = logging.getLogger(__name__)


def _define_model(model):
    return model


class IntGradsPytorch(object):

    def __init__(self, model):
        """

        Parameters
        ----------
        model
        """
        self.am = IntegratedGradients(model)

    def explain(self, X,
                baselines=None,
                target=None,
                additional_forward_args=None,
                n_steps=50,
                method="gausslegendre",
                internal_batch_size=None,
                return_convergence_delta=False
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
        setattr(self.am, 'return_convergence_delta', return_convergence_delta)
        if return_convergence_delta:
            attr, delta = self.am.attribute(test_input_tensor,
                                            baselines=baselines,
                                            target=target,
                                            additional_forward_args=additional_forward_args,
                                            n_steps=n_steps,
                                            method=method,
                                            internal_batch_size=internal_batch_size,
                                            return_convergence_delta=return_convergence_delta)
        else:
            attr = self.am.attribute(test_input_tensor,
                                     baselines=baselines,
                                     target=target,
                                     additional_forward_args=additional_forward_args,
                                     n_steps=n_steps,
                                     method=method,
                                     internal_batch_size=internal_batch_size,
                                     return_convergence_delta=return_convergence_delta)

        attr = attr.detach().numpy()

        return attr


class NoiseTunnelPytorch(object):

    def __init__(self, pytorch_explainer):
        if hasattr(pytorch_explainer, 'am'):
            self.am = pytorch_explainer.am
            self.nt = NoiseTunnel(self.am)
        else:
            raise AttributeError('explainer must have am (attribute method) attribute')

    def explain(self, X,
                nt_type="smoothgrad",
                n_samples=5,
                stdevs=1.0,
                draw_baseline_from_distrib=False,
                **kwargs):

        inputs = torch.from_numpy(X).type(torch.FloatTensor)
        if self.am.return_convergence_delta:
            attr, delta = self.nt.attribute(inputs,
                                            nt_type=nt_type,
                                            n_samples=n_samples,
                                            stdevs=stdevs,
                                            draw_baseline_from_distrib=draw_baseline_from_distrib,
                                            **kwargs)
        else:
            attr = self.nt.attribute(inputs,
                                     nt_type=nt_type,
                                     n_samples=n_samples,
                                     stdevs=stdevs,
                                     draw_baseline_from_distrib=draw_baseline_from_distrib,
                                     **kwargs)

        attr = attr.detach().numpy()

        return attr
