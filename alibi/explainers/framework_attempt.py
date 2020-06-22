import logging

import numpy as np
import tensorflow as tf

from alibi.api.interfaces import Explainer, Explanation
from alibi.api.defaults import DEFAULT_META_CF, DEFAULT_DATA_CF
from copy import deepcopy
from typing import Any, Dict, Optional, Union


logger = logging.getLogger(__name__)


CF_LOGGING_OPTS_DEFAULT = {
    'verbose': False,
    'log_traces': True,
    'trace_dir': None,
    'summary_freq': 1,
    'image_summary_freq': 10,
}


def _setup_tb(opts, framework='tensorflow'):

    if framework == 'tensorflow':
        return _setup_tb_tensorflow(opts)
    else:
        raise NotImplementedError


def _setup_tb_tensorflow(opts: dict):
    pass


class PyTorchCF: pass
    # TODO: THIS SHOULD BEHAVE THE SAME AS THE TENSORFLOW COUNTERPART

class TensorFlowCF:

    # TODO: ALEX: TBD THIS SHOULD

    def __init__(self, predictor, loss_spec, optimiser):

        self._set_attributes(CF_LOGGING_OPTS_DEFAULT)
        self.cf_constraint=None
        self.variables=None

    def _update_logging(self, opts: Optional[Dict] = None):

        # no updates to settings, just create the writer
        if not opts:
            self.writer = _setup_tb(opts)
            return
        if not opts['log_traces']:
            self.log_traces = False

    def _set_default_optimiser(self):
        # sets a default optimiser
        pass

    def reset_optimiser(self, optimizer):
        pass

    def _set_attributes(self, attrs: Union[Dict[str, Any], None]) -> None:
        # allows dynamic creation of attributes to that
        if not attrs:
            return

        for key, value in attrs.items():
            self.__setattr__(key, value)

    def write_tb(self, step, records: list, prefix: str = '') -> None:
        # TODO: ALEX: CREATE SPECIALISED RECORDS FOR PYTORCH/TF

    def select_features(self, optimised_features: np.ndarray):
        self.mask = tf.identity(optimised_features)


    def make_prediction(self, X: tf.Tensor) -> tf.Tensor:
        pass

    def cf_step(self):
        pass

    def display_solution(self):
        pass


    def get_gradients(self):
        pass

    def apply_gradients(self):
        pass

    def initialise_variables(self, variable_spec: dict):
        pass

class WatcherCFSearch:

    def __init__(self, predictor, loss, optimiser: Optional[dict] = None, framework='tensorflow'):

        if framework=='tensorflow':
            self.framework = TensorFlowCF(predictor, loss)
        else:
            raise NotImplementedError

    def search(self, X_init, optimised_features: np.ndarray, range_constraints: Union[float, np.ndarray]):
        pass

    def _initialise_response(self):
        pass

    def _update_cf_constraints(self, constraints: np.ndarray):
        self.framework.cf_constraints = constraints

    def _collect_step_data(self):
        pass

    def _update_step_result(self):
        pass

    def _bisect_lam(self):
        pass


class WatcherCFSearchBlackBox(WatcherCFSearch):

    def __init__(self, predictor, loss, optimiser, framework='tensorflow'):

        super().__init__(predictor, loss, optimiser, framework)

    def search(self, X_init: np.ndarray, optimised_features: np.ndarray, range_constraints: Union[float, np.ndarray]):
        pass



def _get_search_algorithm(method:str, predictor_type:str):

    if method == 'watcher':
        if predictor_type == 'black-box':
            cls = WatcherCFSearch
        else:
            cls = WatcherCFSearchBlackBox
    else:
        raise NotImplementedError

    return cls


class Counterfactuals(Explainer):

    def __init__(self,
                 predictor,
                 predictor_type: str = 'blackbox',
                 method: str = 'watcher',
                 shape,
                 loss_spec : Optional[dict] = None,
                 framework='tensorflow',
                 **kwargs):

        # TODO: ALEX: HOW CAN THE USER SPEC THE LOSS
        # TODO: ALEX: LOSS SPEC SHOULD BE TIGHTLY COUPLED WITH THE METHOD AND VALIDATED FOR
        search_algorithm = _get_search_algorithm(method, predictor_type)
        self._search_strategy = search_algorithm(predictor, loss_spec)


    def fit(self, X: Optional[np.ndarray]=None, constrain_features=True):

        # TODO: ALEX:  WHAT SHOULD THE ROLE OF FIT BE? CURRENTLY WE SCALE AND FIND FEATURE RANGES
        #  BUT SCALING IS A BIT CUSTOM TO THE WATCHER LOSS? THEREFORE, THE SCALING WOULD
        #  HAVE TO BE IMPLEMENTED BY THE USER EXTERNALLY BECAUSE WE DON'T HAVE KNOWLEDGE OF
        #  LOSS FUNCTION. THEY WOULD PASS IT AS A KWARG OF THE RIGHT TYPE.

        self._search_strategy._update_feat_constraints(self.feature_constraints)

        # TODO: HOWEVER, THE CONSTRAINT FEATURE IS MORE GENERAL; WE COULD DO IT IN CASE FIT IS
        #  CALLED AND PASS IT TO THE OPTIMISER

    def explain(self,
                X: np.ndarray,
                optimisation_opts: Optional[dict] = None,
                logging_opts: Optional[dict] = None,
                feature_whitelist: Union[np.ndarray, str] = 'all') -> "Explanation":

        # TODO: ALEX: THIS NEEDS TO CALL THE SEARCH ALGORITHM
        # TODO: ALEX: NEEDS TO RESET THE OPTIMIZER AND OTHER THINGS LIKE THIS
        # TODO: ALEX: HOW IS THE FEATURE WHITLIST HANDLED

        if logging_opts:
            if logging_opts['verbose']:
                logging.basicConfig(level=logging.DEBUG)




# TODO: ALEX: TBD. SHOULD METHOD AND BLACK-BOX BE PART OF THE EXPLAIN OR

optimisation_opts = {
    'lam_opts': {},
    'search_opts': {},
    'optimizer': None,
    # TODO: ALEX: HOW CAN WE SPECIFY CONSTRAINTS
    'constraints': [],
}

# CONCLUSION: I DON'T THINK THESE FRAMEWORK AGNOSTIC CLASSES MAKE THAT MUCH SENSE. IT BECAUSE EXTREMELY
# AWKWARD TO DEFINE THE PROBLEM (NEED TO PASS VARIABLE AND LOSS SPECS AND THEN YOU HAVE THE PROBLEM OF COMPUTING
# CERTAIN THINGS (E.G., PREDICTIONS, ETC)).