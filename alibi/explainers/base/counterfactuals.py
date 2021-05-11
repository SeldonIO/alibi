import logging

import numpy as np

from alibi.explainers.backend import load_backend
from alibi.utils.frameworks import _validate_framework, _infer_device
from alibi.utils.logging import tensorboard_loggers
from collections import defaultdict
from functools import partial
from typing import Optional, Dict, Union, Tuple, Any, Mapping, Set
from typing_extensions import Literal

logger = logging.getLogger(__name__)


class CounterfactualBase:
    """
    A base class from which all counterfactual implementations should inherit. The class implements the following
    functionality:

        - Initialises the backend, which is an object used to implement training and inference functions in TensorFlow \
        or PyTorch. The backend is set as the `backend` property
        - Attribute setting: `method_opts` contents are set as object attributes. These should be the algorithm \
        default hyperparameters
        - Initialize a TensorBoard writer object setting it as `tensorboard` attribute.
    """  # noqa W605

    def __init__(self,
                 predictor,
                 framework: Literal["pytorch", "tensorflow"],
                 loss_spec: Optional[Dict] = None,
                 method_opts: Optional[Dict] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 **kwargs):

        """
        Initialises the backend of the algorithm, sets default explainer hyperparameters and does basic logging setup.
        
        Parameters
        ----------
        predictor
            A model for which counterfactual instances are to be searched.
        class_name
            The name of the sub-class initialising the object
        framework: {'pytorch', 'tensorflow'}
            Framework that was used for implementing the backend of the algorithm. If the counterfactual algorithm has 
            access to the `predictor` internals, then it should be implemented in the same framework as the backend.
        loss_spec
            Each backend should have a default loss specification (see e.g., 
            `alibi.explainers.backend.tensorflow.counterfactuals`). The purpose of this argument is to allow the user 
            to customize terms of the original loss specification, with the assumption that the terms will have the same 
            positional arguments as the default specification.
        method_opts
            The default hyperparamters of the method. These can be overriden by the user by specifying the `method_opts`
            arguments to the subclass. 
        feature_range
            The min and the max of the valid counterfactual range. If `np.ndarray`, it should have the same shape as the
            instance to be explained.
        kwargs
            Valid options include:
            
                - `predictor_type` ('{'blackbox', 'whitebox'}'): pass `blackbox` to specify that the sub-class \
                implements a black-box counterfactual search. Defaults to `whitebox`.
                - `backend_kwargs`: use to specify any keyword arguments for the backend constructor 
                - `predictor_device`: specifies the device on which the predictor is stored (placeholder for PyTorch \
                support)
                
        Raises
        ------
        ImportError
            If the specified framework is not installed.
        NotImplementedError
            If the value of `framework` is not 'pytorch' or 'tensorflow'.
        """  # noqa W605

        _validate_framework(framework)
        self.fitted = False
        self.params = {}  # type: Dict[str, Any] # used by API classes to update metadata
        predictor_device = kwargs.get('predictor_device', None)
        predictor_type = kwargs.get('predictor_type', None)
        if not predictor_type:
            logger.warning("Predictor type not specified, assuming whitebox predictor.")
            predictor_type = 'whitebox'
        if not predictor_device:
            self.predictor_device = _infer_device(predictor, predictor_type, framework)
        else:
            self.predictor_device = predictor_device

        backend = load_backend(
            class_name=self.__class__.__name__,
            framework=framework,
            predictor_type=predictor_type
        )
        backend_kwargs = kwargs.get("backend_kwargs", {})
        self.backend = backend(
            predictor=predictor,
            loss_spec=loss_spec,
            predictor_type=predictor_type,
            feature_range=feature_range,
            **backend_kwargs
        )
        # track attributes set
        self._expected_attributes = set()
        # create attributes and set them with default values, passed by sub-class
        self.set_attributes(method_opts)
        # TODO: ALEX: TBD: ATTRS "POLICE"
        self._expected_attributes |= self.backend._expected_attributes
        self.set_expected_attributes = partial(self.set_attributes, expected=self._expected_attributes)

        # placeholders for options passed at runtime.
        self.log_traces = True
        self.logging_opts = {}  # type: Dict[str, Any]
        self.tensorboard = tensorboard_loggers[self.backend.framework]()
        # container for the data logged to tensorboard at every step
        self.data_store = defaultdict(lambda: None)  # type: defaultdict

        # optimisation step, defined for logging purposes
        self.step = -1
        self.lam_step = -1

    def __setattr__(self, key: str, value: Any):
        super().__setattr__(key, value)

    def __getattr__(self, key: Any) -> Any:
        return self.__getattribute__(key)

    def set_attributes(self, attrs: Union[Mapping[str, Any], None], expected: Optional[Set[str]] = None) -> None:
        """
        Sets the attributes of the explainer using the (key, value) pairs in attributes. Ignores attributes that
        are not in `expected` if the latter is specified.

        Parameters
        ----------
        attrs
            key-value pairs represent attribute names and values to be set.
        expected
            A dictionary indicating which attributes can be set for the object.
        """

        # called with None if the attributes are not overridden by user
        if not attrs:
            return

        for key, value in attrs.items():
            if isinstance(value, Mapping):
                # recurse to set properties in nested dictionaries
                self.set_attributes(value, expected)
            else:
                # TODO: SHOULD TAKE AN ITERABLE OF KEY-VALUE PAIRS ALSO
                # only set those attributes in the _DEFAULT configuration
                if expected and key not in expected:
                    logger.warning(f"Attribute {key} unknown. Attribute will not be set.")
                    continue
                self.__setattr__(key, value)
                self._expected_attributes.add(key)
                # when user calls __init__ with `method_opts`, we need to ensure that the backend default options are
                # also overridden. The way to do so is to set them explicitly in optimizer __init__.
                if hasattr(self.backend, key):
                    self.backend.__setattr__(key, value)

    def initialise_response(self):
        """
        Initialises the templates that will contain the respone. The response should be returned to the API class
        by the `counterfactual` method.
        """
        raise NotImplementedError("Sub-class should define response template!")

    def counterfactual(self, instance: np.ndarray, optimised_features: np.ndarray, *args, **kwargs) -> Dict:
        """
        Returns a counterfactual for `instance`. Should be called by API classes from the `explain` method.

        Parameters
        ----------
        instance
            The data point for which a counterfactual is searched
        optimised_features
            A boolean mask where `1` indicates that the feature can be optimised during counterfactual search. Should be
            the same shape as `instance`.
        """
        raise NotImplementedError("Sub-class must define counterfactual computation method!")

    def initialise_variables(self, *args, **kwargs):
        """
        This method should be used to initialise any variables that are needed for the computation of the counterfactual
        or during the`fit` step.
        """
        raise NotImplementedError("Sub-class must initialise problem variables!")

    def fit(self, *args, **kwargs):
        """
        This method should be used to change the state of the object prior to searching for counterfactuals.
        """
        raise NotImplementedError("Sub-class must define its own fit method!")

    def reset_step(self):
        """
        Resets the optimisation step. This should be used between subsequent calls to the `counterfactual` methods and/
        or `fit` steps so that variables tracked by TensorBoard have the correct step.
        """
        self.step = -1
        self.lam_step = -1

    def setup_tensorboard(self):
        """
        Initialises the TensorBoard writer. This method should be called at the beginning of the `counterfactual` method.
        """  # noqa

        if self.log_traces:
            self.tensorboard.setup(self.logging_opts)
        self.set_attributes(self.logging_opts)
