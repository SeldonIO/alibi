import copy
import logging

import numpy as np
import tensorflow as tf

from alibi.api.interfaces import Explainer, Explanation
from alibi.api.defaults import DEFAULT_META_CF, DEFAULT_DATA_CF
from alibi.utils.wrappers import methdispatch
from copy import deepcopy
from functools import partial, singledispatch
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from typing import Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING


if TYPE_CHECKING:  # pragma: no cover
    import keras  # noqa

logger = logging.getLogger(__name__)


# TODO: ALEX: TBD: Move the casting generic function to an appropriate utility file.
# TODO: ALEX: This is a simple draft to get the CF code to work. More generally, we would
#  define an object/series of functions to check if PyTorch/TF/both are installed and return an
#  appropriate casting function.

@singledispatch
def to_numpy_arr(X: Union[tf.Tensor, tf.Variable, np.ndarray]):

    supported_types = [tf.Tensor, tf.Variable, np.ndarray]
    raise TypeError(f"Expected input object type to be one of {supported_types} but got type {type(X)}!")


@to_numpy_arr.register
def _(X: np.ndarray) -> np.ndarray:
    return X


@to_numpy_arr.register
def _(X: tf.Tensor) -> np.ndarray:
    return X.numpy()


@to_numpy_arr.register
def _(X: tf.Variable) -> np.ndarray:
    return X.numpy()


# TODO: ALEX: TBD I think it's best practice to define all the custom
#  errors in one file (e.g., exceptions.py/error.py). We
#  could standardise this across the library


class CounterfactualError(Exception):
    pass


CF_SUPPORTED_DISTANCE_FUNCTIONS = ['l1']
CF_VALID_SCALING = ['MAD']
CF_PARAMS = ['scale_loss', 'constrain_features', 'feature_whitelist']


def counterfactual_loss(instance: tf.Tensor,
                        cf: tf.Variable,
                        lam: float,
                        feature_scale: tf.Tensor,
                        pred_probas: Optional[tf.Variable] = None,
                        target_probas: Optional[tf.Tensor] = None,
                        distance_fcn: str ='l1',
                        ) -> tf.Tensor:

    # TODO: ALEX: TBD: TO SUPPORT BLACK-BOX, PYTORCH WE NEED A SIMILAR FUNCTION AND A HIGHER LEVEL ROUTINE TO
    #  "DISPATCH" AMONG THESE FUNCTIONS

    # TODO: ALEX: DOCSTRING

    distance = lam * distance_loss(instance, cf, feature_scale, distance_fcn=distance_fcn)
    if pred_probas is None or target_probas is None:
        return distance
    pred = pred_loss(pred_probas, target_probas)
    return distance + pred


def distance_loss(instance: tf.Tensor,
                  cf: tf.Variable,
                  feature_scale: tf.Tensor,
                  distance_fcn: str = 'l1') -> tf.Tensor:

    # TODO: ALEX: DOCSTRING

    if distance_fcn not in CF_SUPPORTED_DISTANCE_FUNCTIONS:
        raise NotImplementedError(f"Distance function {distance_fcn} is not supported!")

    ax_sum = tuple(np.arange(1, len(instance.shape)))

    return tf.reduce_sum(tf.abs(cf - instance) / feature_scale, axis=ax_sum, name='l1')


def pred_loss(pred_probas: tf.Variable, target_probas: tf.Tensor) -> tf.Tensor:

    # TODO: ALEX: DOCSTRING

    return tf.square(pred_probas - target_probas)


def range_constraint(X: Union[tf.Variable, tf.Tensor], low: Union[float, np.ndarray], high: Union[float, np.ndarray]):

    # TODO: ALEX: TBD: SHOULD WE DO ERROR HANDLING FOR THIS FUNCTION (E.G., LOW < HIGH)

    # TODO: ALEX: ADD DOCSTRING,
    # TODO: ALEX: CROSS REFERENCE INIT DOC FOR MIN AND MAX, MAKE SURE THEY DEFINE SHAPES FOR MIN/MAX

    return tf.clip_by_value(X, clip_value_min=low, clip_value_max=high)


class Counterfactual(Explainer):

    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model, 'keras.Model'],
                 shape: Tuple[int, ...],
                 distance_fn: str = 'l1',
                 feature_range: Tuple[Union[float, np.ndarray], Union[float, np.ndarray]] = (-1e10, 1e10),
                 cf_init: str = 'identity',
                 write_dir: str = None,
                 debug: bool = False,
                 save_every: int = 1,
                 image_summary_freq: int = 10,
                 **kwargs) -> None:
        """
        Initialize counterfactual explanation method based on Wachter et al. (2017).

        Parameters
        ----------
        predictor
            Keras or TensorFlow model, assumed to return tf.Tensor object that contains class probabilities. The object 
            returned should always be two-dimensional. For example, a single-output classification model operating on a 
            single input instance should return a tensor or array with shape  `(1, 1)`. In the future this explainer may 
            be extended to work with regression models.
        shape
            Shape of input data starting with batch size.
        distance_fn
            Distance function to use in the loss term.
        feature_range
            Tuple with upper and lower bounds for feature values of the counterfactual. The upper and lower bounds can
            be floats or numpy arrays with dimension :math:`(N, )` where `N` is the number of features, as might be the
            case for a tabular data application. For images, a tensor of the same shape as the input data can be applied
            for pixel-wise constraints. If `fit` is called with argument `X`, then  feature_range is automatically
            updated as `(X.min(axis=0), X.max(axis=0))`.
        cf_init
            Initialization method for the search of counterfactuals, currently must be `'identity'`.
        write_dir
            Directory to write Tensorboard files to.
        debug
            Flag to write Tensorboard summaries for debugging.
        save_every
            Controls the frequency at which data is written to Tensorboard.
        kwargs
            Keyword arguments can be passed to initialise the optimiser state and search parameters. The
            parameters that are passed are listed in the documentation for `explain`. The user is encouraged
            to pick adequate defaults for their problem and adjust the parameters if the search procedure fails
            to converge.
        """  # noqa W605

        # TODO: ALEX: TBD: I think this

        super().__init__(meta=copy.deepcopy(DEFAULT_META_CF))
        # get params for storage in meta
        params = locals()
        remove = ['self', 'predictor', '__class__']
        for key in remove:
            params.pop(key)
        self.meta['params'].update(params)
        self.fitted = False

        self.predictor = predictor
        self.batch_size = shape[0]

        # default options for the optimizer, can override at explain time
        self.learning_rate_init = kwargs.get('learning_rate_init', 0.1)
        self.lr_schedule = self.learning_rate_init
        self.decay = kwargs.get('decay', True)
        self.max_iter = kwargs.get('max_iter', 1000)
        self.lam_init = kwargs.get('lam_init', 0.1)
        self.max_lam_steps = kwargs.get('max_lam_steps', 10)
        self.early_stop = kwargs.get('early_stop', 50)
        self.eps = kwargs.get('eps', 0.01)

        # feature scale updated in fit if a dataset is passed
        self.feature_scale = tf.identity(1.0)
        self.feature_scale_numpy = 1.0
        self.loss = partial(
            counterfactual_loss,
            feature_scale=self.feature_scale,
            distance_fcn=distance_fn,
        )
        self.distance_fn = distance_fn

        # counterfactual search is constrained s.t. maximum absolute
        # value difference between predicted probability and target
        # probability for counterfactual is smaller than tol
        self.tol = kwargs.get('tol', 0.05)

        # init. at explain time
        self.target_proba = None  # type: tf.Tensor
        self.instance = None  # type: tf.Tensor
        self.cf = None  # type: tf.Variable

        # initialisation method and constraints for counterfactual
        self.cf_init = cf_init
        self.cf_constraint = partial(range_constraint, low=feature_range[0], high=feature_range[1])
        # scheduler and optimizer initialised at explain time
        self.optimizer = None

        # TODO: ALEX: TBD: I feel that we should really expose `debug` in `explain` and create the `writer` there.
        #  If your explanation does not converge, chances are you want to just write "debug=True" there and see the
        #  detailed traces ... We can still allow it to be passed here as per the optimiser options

        self.debug = debug
        self.summary_freq = save_every
        self.image_summary_freq = image_summary_freq
        self.step = 0
        self.lam_step = 0
        if write_dir is not None:
            self.writer = tf.summary.create_file_writer(write_dir)

        # return templates
        self._initialise_response()

    def _initialise_response(self):
        """ Initialises the templates that will form the body of the `explaination.data` field."""
        # TODO: ALEX: DOCSTRING

        # NB: (class, proba) could be renamed to (output_idx, output) to make sense for regression settings
        self.this_result = dict.fromkeys(
            ['X', 'distance_loss', 'prediction_loss', 'total_loss', 'lambda', 'step', 'proba', 'class']
        )
        self.search_results = copy.deepcopy(DEFAULT_DATA_CF)
        self.search_results['all'] = {i: [] for i in range(self.max_lam_steps)}
        self.search_results['lambda_sweep'] = {}

    def fit(self,
            X: Optional[np.ndarray] = None,
            scale: Union[bool, str] = False,
            constrain_features: bool = True) -> "Counterfactual":
        r"""
        Calling this method with an array of :math:`N` data points, assumed to be the leading dimension of `X`, has the
        following functions:
            - If `constrain_features=True`, the minimum and maximum of the array along the leading dimension constrain 
            the minimum and the maximum of the counterfactual
            - If the `scale` argument is set to `True` or `MAD`, then the distance between the input and the
            counterfactual is scaled, feature-wise at each optimisiation step by the feature-wise mean absolute
            deviation (MAD) calculated from `X` as detailed in the notes. Other options might be supported in the 
            future (raise a feature request).

        Parameters
        ----------
        X
            An array of :math:`N` data points.
        scale
            If set to `True` or 'MAD', the MAD is computed and applied to the distance part of the function. 
        constrain_features
            If `True` the counterfactual features are constrained.
        
        Returns
        -------
        A fitted counterfactual explainer object.
        
        Notes
        -----

        If `scale='MAD'` then the following calculation is performed:

        .. math:: MAD_{j} = \text{median}_{i \in \{1, ..., n\}}(|x_{i,j} - \text{median}_{l \in \{1,...,n\}}(x_{l,j})|)
        """ # noqa W605

        # TODO: ALEX: TBD: Should automatic feature range be optional (e.g., have a constrain_range=True bool)? (No?)
        # TODO: ALEX: TBD  Allow pd.DataFrame in fit?
        # TODO: ALEX: TBD: Could allow an optional scaler argument s.t. people can use skelearn (this might require
        #  effort so I would put it on hold for now)

        # TODO: epsilons ?

        self._check_scale(scale)

        if X is not None:

            if self.scale:
                # infer median absolute deviation (MAD) and update loss
                if scale == 'MAD' or isinstance(scale, bool):
                    self._mad_scaling(X)
            if constrain_features:
                # infer feature ranges and update counterfactual constraints
                feat_min, feat_max = np.min(X, axis=0), np.max(X, axis=0)
                self.cf_constraint.keywords['low'] = feat_min
                self.cf_constraint.keywords['high'] = feat_max

        self.fitted = True

        params = {
            'scale_loss': self.scale,
            'constrain_features': constrain_features,
            'fitted': self.fitted,
        }

        self._update_metadata(params, params=True)

        return self

    def _mad_scaling(self, X: np.ndarray):
        """
        Computes feature-wise mean absolute deviation. Fixes the latter as an invariant input to the
        loss function, which divides the loss by this value.
        """

        # TODO: ALEX: TBD: THROW WARNINGS IF THE FEATURE SCALE IS EITHER VERY LARGE OR VERY SMALL?

        feat_median = np.median(X, axis=0)
        mad = np.median(np.abs(X - feat_median), axis=0)
        self.feature_scale = tf.identity(mad)
        self.feature_scale_numpy = mad
        self.loss.keywords['feature_scale'] = self.feature_scale

    def _check_scale(self, scale: Union[bool, str]) -> None:

        self.scale = False
        if isinstance(scale, str):
            if scale not in CF_VALID_SCALING:
                logger.warning(f"Received unrecognised option {scale} for scale. No scaling will take place. "
                               f"Recognised scaling methods are {CF_VALID_SCALING}.")
            else:
                self.scale = True

        if isinstance(scale, bool):
            if scale:
                logger.info(f"Defaulting to mean absolute deviation scaling!")
                self.scale = True



    def explain(self,
                X: np.ndarray,
                target_class: Union[str, int] = 'other',
                target_proba: float = 1.0,
                tol: float = 0.05,
                lam_init: float = 1e-1,
                max_lam_steps: int = 10,
                learning_rate_init=0.1,
                max_iter: int = 1000,
                decay: bool = True,
                early_stop: int = 50,
                eps: Union[float, np.ndarray] = 0.01,  # feature-wise epsilons
                ) -> Explanation:
        """
        Explain an instance and return the counterfactual with metadata.

        Parameters
        ----------
        X
            Instance to be explained. Only 1 instance can be explained at one time, so the shape of the `X` 
            array is expected to be `(1, ...)` where the ellipsis corresponds to the dimension of one datapoint.
            In the future, batches of instances may be supported via a distributed implementation.  
        target_class
            Target class for the counterfactual to reach. Allowed values are:

                -`'same'`: the predicted class of the counterfactual will be the same as the instance to be explained
                -`'other'`: the predicted class of the counterfactual will be the class with the closest probability to \
                the instance to be explained 
                -``int``: an integer denoting desired class membership for the counterfactual instance
        target_proba
            Target probability for the counterfactual to reach.
        tol
            Tolerance for the counterfactual target probability.
        lam_init
            Initial regularization constant for the prediction part of the Wachter loss.
        max_lam_steps
            Maximum number of times to adjust the regularization constant before terminating the search (number of
            outer loops).
        learning_rate_init
            Initial gradient descent learning rate, reset at each outer loop iteration.
        max_iter
            Maximum number of iterations to run the gradient descent for (number of inner loops for each outer loop).
        early_stop
            Number of steps after which to terminate gradient descent if all or none of found instances are solutions
        eps
            Gradient step sizes used in calculating numerical gradients, defaults to a single value for all
            features, but can be passed an array for feature-wise step sizes.
        decay
            Flag to decay learning rate to zero for each outer loop over lambda.

        Returns
        -------
        *explanation* - a dictionary containing the counterfactual with additional metadata.

        """  # noqa W605

        # TODO: ALEX: DOCUMENT SHAPE REQUIREMENTS for `X` in docstring above
        # TODO: ALEX: Improve docstring for `early_stopping`  and `tol` above?
        # TODO: EXPLAIN WHEN `eps` is USED above?

        if X.shape[0] != 1:
            logger.warning('Currently only single instance explanations supported (first dim = 1), '
                           'but first dim = %s', X.shape[0])
        # TODO: ALEX: CHECK TENSOR SHAPES HERE AND UPDATE ACCORDINGLY
        self.instance_numpy = X
        self.target_proba_numpy = np.array([target_proba])[:, np.newaxis]

        # update options for the optimiser
        self.decay = decay
        self.learning_rate_init = learning_rate_init
        self.lr_schedule = learning_rate_init
        self.max_iter = max_iter
        self._initialise_optimiser()

        # make a prediction
        Y = self.make_prediction(X)
        instance_class = self._get_class(to_numpy_arr(Y))
        instance_proba = to_numpy_arr(Y[:, instance_class])

        # helper function to return the model output given the target class
        self.get_cf_prediction = partial(self._get_cf_prediction, instance_class=instance_class)
        # initialize optimised variables, targets and loss weight
        self._initialise_variables(X, target_class, target_proba)
        lam_dict = self._initialise_lam()

        # search for a counterfactual by optimising loss starting from the original solution
        result = self._search(init_cf=X, init_lam=lam_dict['midpoint'], init_lb=lam_dict['lb'], init_ub=lam_dict['ub'])
        self._reset_step()

        return self.build_explanation(X, result, instance_class, instance_proba)

    def make_prediction(self, X: Union[np.ndarray, tf.Variable, tf.Tensor]) -> tf.Tensor:
        """
        Makes a prediction for data points in `X`
        """
        # TODO: ALEX: DOCSTRING

        return self.predictor(X, training=False)

    def _get_class(self, Y: np.ndarray, threshold: float = 0.5) -> int:
        """
        Given a prediction vector containing the model output for a prediction on a single data point,
        it returns an integer representing the predicted class of the data point. Accounts for the case
        where a binary classifier returns a single output.
        """

        # TODO: ALEX: TBD: Parametrise `threhold` in explain or do we assume people always assign labels on this rule?
        # TODO: ALEX: ADD DOCSTRING

        if Y.shape[1] == 1:
            return int(Y > threshold)
        else:
            return np.argmax(Y)

    def _initialise_optimiser(self):
        """
        Initialises the optimiser that will provide gradients for the counterfactual search.
        This method is called each time :math:`\lambda`, the weight of the distance between
        the counterfactual and the instance :math:`X` provided to the `explain` method is 
        updated, in order to provide a reset the learning rate schedule for each 
        :math:`\lambda` optimisation step.
        """ # noqa W605

        # TODO: ALEX: TBD: We should make the type of optimiser an option. An argument for this is testing, where you may
        #  want to check things like updates being correct as lam varies which are more awkward with complex optimisers
        # TODO: ALEX: TBD: Should we have a separate `reset_optimiser` method? The case for this is that for a
        #  constant learning rate we don't need to re-initialise this object time and again?

        # TODO: ALEX: ADD DOCSTRING

        if self.decay:
            self.lr_schedule = PolynomialDecay(self.learning_rate_init, self.max_iter, end_learning_rate=0.002, power=1)
        self.optimizer = Adam(learning_rate=self.lr_schedule)

    @methdispatch
    def _get_cf_prediction(self, target_class: Union[int, str], cf: tf.Tensor, instance_class: int) -> tf.Tensor:
        """
        Returns the slice of the model outputs that corresponds to the target class for the counterfactual that
        was specified by the user.
        """
        raise TypeError(f"Expected target_class to be str or int but target_class was {type(target_class)}!")

    @_get_cf_prediction.register
    def _(self, target_class: str, cf: tf.Tensor, instance_class: int) -> tf.Tensor:

        # TODO: ALEX: DOCSTRING
        # TODO: ALEX: CHECK: Is type of cf correct?

        prediction = self.make_prediction(cf)
        if target_class == 'same':
            return prediction[:, instance_class]

        _, indices = tf.math.top_k(prediction, k=2)
        if indices[0][0] == instance_class:
            cf_prediction = prediction[:, indices[0][1].numpy()]
        else:
            cf_prediction = prediction[:, indices[0][0].numpy()]
        return cf_prediction

    @_get_cf_prediction.register
    def _(self, target_class: int, cf: tf.Tensor, instance_class: int) -> tf.Tensor:

        # TODO: ALEX: DOCSTRING

        return self.predictor(cf, training=False)[:, target_class]

    def _initialise_variables(self, X: np.ndarray, target_class: int, target_proba: float) -> None:
        """
        Initialises optimisation variables so that the TensorFlow auto-differentiation framework
        can be used for counterfactual search.
        """

        # TODO: ALEX: DOCS
        # TODO initialization strategies ("same_class", "random", "from_train")

        # tf.identity is the same as constant but does not always create tensors on CPU
        self.target_proba = tf.identity(target_proba * np.ones(self.batch_size, dtype=X.dtype), name='target_proba')
        self.target_class = target_class
        self.instance = tf.identity(X, name='instance')
        self._initialise_cf(X)

    def _initialise_cf(self, X: np.ndarray) -> None:
        """
        Initialisises the counterfactual to the data point `X`. It is assumed that this data point
        is the same as the data point whose counterfactual is to be determined.
        """
        # TODO: ALEX: DOCS

        if self.cf_init == 'identity':
            logger.debug('Initializing counterfactual search at the instance to be explained ... ')
            self.cf = tf.Variable(
                initial_value=X,
                trainable=True,
                name='counterfactual',
                constraint=self.cf_constraint,
            )
        else:
            raise ValueError("Initialization method should be 'identity'!")

    def _search(self, *, init_cf: np.ndarray, init_lam: float, init_lb: float, init_ub: float) -> dict:
        """
        Searches a counterfactual given an initial condition for the counterfactual. The search has two loops:
            - An outer loop, where :math:`\lambda` (the weight of the distance between the current counterfactual
            and the input `X` to explain in the loss term) is optimised using bisection
            - An inner loop, where for constant `lambda` the current counterfactual is updated using the gradient 
            of the counterfactual loss function. 
        """  # noqa: W605

        # TODO: ALEX: DOCSTRING

        summary_freq = self.summary_freq
        cf_found = np.zeros((self.max_lam_steps, ), dtype=np.uint16)
        # re-init. cf as initial lambda sweep changed the initial condition
        self._initialise_cf(init_cf)
        lam, lam_lb, lam_ub = init_lam, init_lb, init_ub
        for lam_step in range(self.max_lam_steps):
            # re-set learning rate
            self._initialise_optimiser()
            found, not_found = 0, 0
            for gd_step in range(self.max_iter):
                self._cf_step(lam)
                constraint_satisfied = self._check_constraint(self.cf, self.target_proba, self.tol)
                cf_prediction = self.make_prediction(self.cf)

                # save and optionally display results of current gradient descent step
                # TODO: ALEX: TBD: We could make `lam` AND `step` object properties and not have to pass this
                #  current state to the functions, but maybe it is a bit clearer what happens?
                current_state = (self.step, lam, to_numpy_arr(self.cf), to_numpy_arr(cf_prediction))
                write_summary = self.step % summary_freq == 0 and self.debug
                if constraint_satisfied:
                    cf_found[lam_step] += 1
                    self._update_search_result(*current_state)
                    found += 1
                    not_found = 0
                else:
                    found = 0
                    not_found += 1
                    if write_summary:
                        self._collect_step_data(*current_state)

                if write_summary:
                    self._write_tb(self.step, lam, lam_lb, lam_ub, cf_found, prefix='counterfactual_search/')
                self.step += 1

                # early stopping criterion - if no solutions or enough solutions found, change lambda
                if found >= self.early_stop or not_found >= self.early_stop:
                    break

            lam, lam_lb, lam_ub = self._bisect_lambda(cf_found, lam_step, lam, lam_lb, lam_ub)
            self.lam_step += 1

        if self.debug:
            self._display_solution(prefix='best_solutions/')

        return deepcopy(self.search_results)

    def _display_solution(self, prefix=''):

        with self.writer.as_default():
            tf.summary.image(
                '{}counterfactuals/optimal_cf'.format(prefix),
                self.search_results['cf']['X'],
                step=0,
                description=r"A counterfactual `X'` that satisfies `|f(X') - y'| < \epsilon` and minimises `d(X, X')`."
                            " Here `y'` is the target probability and `f(X')` is the model prediction on the counterfactual."
                            "The weight of distance part of the loss is {:.5f}".format(self.search_results['cf']['lambda']) # noqa
            )

            tf.summary.image(
                '{}counterfactuals/original_input'.format(prefix),
                self.instance,
                step=0,
                description="Instance for which a counterfactual is to be found."
            )

        self.writer.flush()

    def _reset_step(self):
        self.step = 0
        self.lam_step = 0

    def _initialise_lam(self, lams: Optional[np.ndarray] = None, n_orders: int = 10) -> Dict[str, float]:
        """
        Runs a search procedure over a specified range of lambdas in order to determine a good initial value
        for this parameter. If `lams` is not specified, then an exponential decaying lambda starting at `lambda_init`
        over `n_orders` of magnitude is employed. The method saves the results of this sweep in the `'lambda_sweep'`
        field of the response.

        Returns
        -------
        A dictionary containg as keys:
            - 'lb': lower bound for lambda
            - 'ub': upper bound for lambda
            - 'midpoint': the midpoints of the interval [lb, ub]
        """
        # TODO: ALEX: DOCSTRING

        # TODO: ALEX: TBD: PASS lams and n_orders to this method?
        # TODO: ALEX: TBD: Maybe it would be nice if we also expose this method to the user? That way they can
        #  play with lam and the settings - this increases the likelihood of finding a decent lam to start with?
        #  Then they can tell the explainer to skip this step.

        # TODO: ALEX: Log the data but at the end copy it into an appropriately named field and then reset the `cf` and
        #  `all` fields so that only data from the main search is stored in there

        n_steps = self.max_iter // n_orders
        if lams is None:
            lams = np.array([self.lam_init / 10 ** i for i in range(n_orders)])  # exponential decay
        cf_found = np.zeros(lams.shape, dtype=np.uint16)

        logger.debug('Initial lambda sweep: %s', lams)

        for lam_step, lam in enumerate(lams):
            # optimiser is re-created so that lr schedule is reset for every lam
            self._initialise_optimiser()
            for gd_step in range(n_steps):
                # update cf with loss gradient for a fixed lambda
                self._cf_step(lam)
                constraint_satisfied = self._check_constraint(self.cf, self.target_proba, self.tol)
                cf_prediction = self.make_prediction(self.cf)

                # save search results and log to TensorBoard
                write_summary = self.debug and self.step % self.summary_freq == 0
                current_state = (self.step, lam, to_numpy_arr(self.cf), to_numpy_arr(cf_prediction))
                if constraint_satisfied:
                    cf_found[lam_step] += 1
                    self._update_search_result(*current_state)
                else:
                    if write_summary:
                        self._collect_step_data(*current_state)

                if write_summary:
                    self._write_tb(self.step, lam, 0, 0, cf_found, prefix='lambda_sweep/')
                self.step += 1
            self.lam_step += 1

        self._reset_step()

        logger.debug(f"Counterfactuals found: {cf_found}")

        # determine upper and lower bounds given the solutions found
        lb_idx_arr = np.where(cf_found > 0)[0]
        if len(lb_idx_arr) < 2:
            logger.exception("Could not find lower bound for lambda, try decreasing lam_init!")
            raise CounterfactualError("Could not find lower bound for lambda, try decreasing lam_init!")
        lam_lb = lams[lb_idx_arr[1]]

        ub_idx_arr = np.where(cf_found == 0)[0]
        if len(ub_idx_arr) == 1:
            logger.warning(
                f"Could not find upper bound for lambda where no solutions found. "
                f"Setting upper bound to lam_init={lams[0]}")
            ub_idx = 0
        else:
            ub_idx = ub_idx_arr[-1]
        lam_ub = lams[ub_idx]

        logger.debug(f"Found upper and lower bounds for lambda: {lam_ub}, {lam_lb}")
        assert lam_lb - lam_ub > -1e-3

        # re-initialise response s.t. 'cf' and 'all' fields contain only main search resuts
        sweep_results = deepcopy(self.search_results)
        self._initialise_response()
        self.search_results['lambda_sweep']['all'] = sweep_results['all']
        self.search_results['lambda_sweep']['cf'] = sweep_results['cf']

        return {'lb': lam_lb, 'ub': lam_ub, 'midpoint': 0.5*(lam_ub + lam_lb)}

    def _cf_step(self, lam: float):
        """
        Runs a gradient descent step and updates current solution.
        """
        # TODO: ALEX: DOCSTRING

        gradients = self._get_gradients(lam)
        self._apply_gradients(gradients)

    def _get_gradients(self, lam: float) -> List[tf.Tensor]:
        """
        Calculates the gradients of the counterfactual loss.
        """
        # TODO: ALEX: DOCSTRING

        with tf.GradientTape() as tape:
            prediction = self.get_cf_prediction(self.target_class, self.cf)
            loss = self.loss(
                instance=self.instance,
                cf=self.cf,
                lam=lam,
                pred_probas=prediction,
                target_probas=self.target_proba,
            )
        gradients = tape.gradient(loss, [self.cf])

        return gradients

    def _apply_gradients(self, gradients: List[tf.Tensor]) -> None:
        """
        Updates the current solution with the gradients of the counterfactua loss.
        """
        # TODO: ALEX: DOCSTRING

        self.optimizer.apply_gradients(zip(gradients, [self.cf]))

    def _check_constraint(self, cf: tf.Variable, target_proba: tf.Tensor, tol: float) -> bool:
        """
        Checks if the constraint |f(cf) - target_proba| < self.eps holds where f is the model
        prediction for the class specified by the user, given the counterfactual `cf`. If the
        constraint holds, a counterfactual has been found.
        """

        # TODO: ALEX: DOCSTRING

        return tf.reduce_all(
            tf.math.abs(self.get_cf_prediction(self.target_class, cf) - target_proba) <= tol,
        ).numpy()

    def _update_search_result(self,
                              step: int,
                              lam: float,
                              current_cf: np.ndarray,
                              current_cf_pred: np.ndarray,
                              ):
        """
        Updates the model response. Called only if a valid solution to the optimisation problem is found.
        """

        # TODO: ALEX: DOCSTRING

        # perform necessary calculations and update `self.instance_dict`
        self._collect_step_data(step, lam, current_cf, current_cf_pred)
        self.search_results['all'][self.lam_step].append(deepcopy(self.this_result))

        # update best CF if it has a smaller distance
        if not self.search_results['cf']:
            self.search_results['cf'] = deepcopy(self.this_result)
        elif self.this_result['distance_loss'] < self.search_results['cf']['distance_loss']:
            self.search_results['cf'] = deepcopy(self.this_result)

        logger.debug(f"CF found at step {step}.")

    def _collect_step_data(self, step: int, lam: float, current_cf: np.ndarray, current_cf_pred: np.ndarray):
        """
        Collects data from the current optimisation step. This data is part of the response only if the current
        optimisation step yields a valid solution. Otherwise, if running in debug mode, the data is written to a
        TensorFlow event file for visualisation purposes.
        """

        # TODO: ALEX: DOCSTRING

        instance = self.instance_numpy

        # compute loss terms for current counterfactual
        # TODO: ALEX: TBD: Depending on how we implement the loss of the black-box, we could leverage that here
        #  as opposed to hardcoding the calculation
        ax_sum = tuple(np.arange(1, len(instance.shape)))
        if self.distance_fn == 'l1':
            dist_loss = np.sum(np.abs(current_cf - instance) / self.feature_scale_numpy, axis=ax_sum)
        else:
            dist_loss = np.nan

        pred_class = self._get_class(current_cf_pred)
        pred_proba = current_cf_pred[:, pred_class]
        pred_loss = (pred_proba - self.target_proba_numpy) ** 2
        # populate the return dict
        self.this_result['X'] = current_cf
        self.this_result['lambda'] = lam
        self.this_result['step'] = step
        self.this_result['class'] = pred_class
        self.this_result['proba'] = pred_proba.item()
        self.this_result['distance_loss'] = dist_loss.item()
        self.this_result['prediction_loss'] = pred_loss.item()
        self.this_result['total_loss'] = (pred_loss + lam * dist_loss).item()

    def _write_tb(self,
                  step: int,
                  lam: float,
                  lam_lb: float,
                  lam_ub: float,
                  cf_found: np.ndarray,
                  **kwargs):
        """
        Writes data to a TensorFlow event file for visuatisation purposes. Called only if running in debug mode.
        """

        prefix = kwargs.get('prefix', '')
        found = kwargs.get('found', 0)
        not_found = kwargs.get('not_found', 0)
        lr = self._get_learning_rate()

        # TODO: ALEX: TBD: For image data, it would be nice if we displayed the counterfactuals as well?
        # TODO: ALEX: TBD: I don't quite like how we take a random set of arguments here. Would be nice to have
        #  a cleaner interface something that maybe takes step as pos, a dict with other things you might want to
        #  pass it but mostly takes what is in a dictionary that saves the algo state (in this case, `instance_dict`)
        #  and outputs that. Might be more hassle than its worth.

        # TODO: ALEX: ENSURE STEP IS PASSED CORRECTLY FROM BOTH THE INITIAL LOOP AND THEREAFTER
        # TODO: ALEX: COULD RETRIEVE STEP FROM THE instance_dictionary instead?
        instance_dict = self.this_result

        with tf.summary.record_if(tf.equal(step % self.summary_freq, 0)):
            with self.writer.as_default():

                # optimiser data
                tf.summary.scalar("{}global_step".format(prefix), step, step=step)
                tf.summary.scalar(
                    "{}lr/lr".format(prefix),
                    lr,
                    step=step,
                    description="Gradient descent optimiser learning rate."
                )

                # loss data
                tf.summary.scalar(
                    "{}losses/loss_dist".format(prefix),
                    instance_dict['distance_loss'],
                    step=step,
                    description="The distance between the counterfactual at the current optimisation step and the "
                                "instance to be explained."
                )
                tf.summary.scalar(
                    "{}losses/loss_pred".format(prefix),
                    instance_dict['prediction_loss'],
                    step=step,
                    description="The squared difference between the output of the model evaluated with the "
                                "the counterfactual at the current optimisation step and the model output evaluated at "
                                "the instance to be explained."
                )
                tf.summary.scalar(
                    "{}losses/total_loss".format(prefix),
                    instance_dict['total_loss'],
                    step=step,
                    description="The value of the counterfactual loss for the current optimisation step (sum loss_dist "
                                "and loss_pred"
                )
                # tf.summary.scalar("{}losses/pred_div_dist".format(prefix), loss_pred / (lam * dist), step=step)

                # distance loss weight data
                tf.summary.scalar(
                    "{}lambda/lambda".format(prefix),
                    lam,
                    step=step,
                    description="The weight of the distance_loss part of the counterfactual loss."
                )
                tf.summary.scalar(
                    "{}lambda/l_bound".format(prefix),
                    lam_lb,
                    step=step,
                    description="The upper bound of the interval on which lambda is optimised using binary search.",
                )
                tf.summary.scalar(
                    "{}lambda/u_bound".format(prefix),
                    lam_ub,
                    step=step,
                    description="The lower bound of the interval on which lambda is optimised using binary search."
                )

                tf.summary.scalar(
                    "{}lambda/lam_interval_len".format(prefix),
                    lam_ub - lam_lb,
                    step=step,
                    description="The length of the interval whithin which lambda is optimised."
                )

                # counterfactual data
                tf.summary.scalar(
                    "{}counterfactuals/prediction".format(prefix),
                    instance_dict['proba'],
                    step=step,
                    description="The output of the model at the current optimisation step.",
                )
                tf.summary.scalar(
                    "{}counterfactuals/total_counterfactuals".format(prefix),
                    cf_found.sum(),
                    step=step,
                    description="Total number of counterfactuals found in the optimisation so far."
                )
                tf.summary.scalar(
                    "{}counterfactuals/found".format(prefix),
                    found,
                    step=step,
                    description="Counter incremented when the constraint on the model output for the current "
                                "counterfactual is satisfied. Reset to 0 when the constraint is not met."
                )

                tf.summary.scalar(
                    "{}counterfactuals/not_found".format(prefix),
                    not_found,
                    step=step,
                    description="Counter incremented by one if the constraint on the model value evaluated for the "
                                "current counterfactual is not met. Resets to 0 when the constraint is satisfied",
                )

                with tf.summary.record_if(step % self.image_summary_freq == 0):
                    tf.summary.image(
                        '{}counterfactuals/current_solution'.format(prefix),
                        self.cf,
                        step=step,
                        description="Current counterfactual"
                    )

        self.writer.flush()

    def _get_learning_rate(self):
        """
        Returns the learning rate of the optimizer for visualisation purposes.
        """
        return self.optimizer._decayed_lr(tf.float32)

    def _bisect_lambda(self,
                       cf_found: np.ndarray,
                       lam_step: int,
                       lam: float,
                       lam_lb: float,
                       lam_ub: float,
                       lam_cf_threshold: int = 5) -> Tuple[float, float, float]:
        """
        Runs a bisection algorithm on lambda in order to find a better counterfactual ( # TODO: ADD DETAILS)
        """

        # TODO: ALEX: DOCSTRING
        # TODO: ALEX: TBD: SHOULD WE EXPOSE lam_cf_threshold as parameter to user?

        # lam_cf_threshold: minimum number of CF instances to warrant increasing lambda
        if cf_found[lam_step] >= lam_cf_threshold:
            lam_lb = max(lam, lam_lb)
            logger.debug(f"Lambda bounds: ({lam_lb}, {lam_ub})")
            if lam_ub < 1e9:
                lam = (lam_lb + lam_ub) / 2
            else:
                lam *= 10
                logger.debug(f"Changed lambda to {lam}")

        elif cf_found[lam_step] < lam_cf_threshold:
            # if not enough solutions found so far, decrease lambda by a factor of 10,
            # otherwise bisect up to the last known successful lambda
            lam_ub = min(lam_ub, lam)
            logger.debug(f"Lambda bounds: ({lam_lb}, {lam_ub})")
            if lam_lb > 0:
                lam = (lam_lb + lam_ub) / 2
                logger.debug(f"Changed lambda to {lam}")
            else:
                lam /= 10

        return lam, lam_lb, lam_ub

    def build_explanation(self, X: np.ndarray, result: dict, instance_class: int, instance_proba: float) -> Explanation:
        """
        Creates an explanation object and re-initialises the response to allow calling `explain` multiple times on
        the same explainer.
        """

        # create explanation object
        result['instance'] = X
        result['instance_class'] = instance_class
        result['instance_proba'] = instance_proba
        explanation = Explanation(meta=copy.deepcopy(self.meta), data=result)
        # reset response
        self._initialise_response()

        return explanation


    def _update_metadata(self, data_dict: dict, params: bool = False) -> None:
        """
        This function updates the metadata of the explainer using the data from
        the `data_dict`. If the params option is specified, then each key-value
        pair is added to the metadata `'params'` dictionary only if the key is
        included in `KERNEL_SHAP_PARAMS`.

        Parameters
        ----------
        data_dict
            Dictionary containing the data to be stored in the metadata.
        params
            If True, the method updates the `'params'` attribute of the metatadata.
        """

        if params:
            for key in data_dict.keys():
                if key not in CF_PARAMS:
                    continue
                else:
                    self.meta['params'].update([(key, data_dict[key])])
        else:
            self.meta.update(data_dict)

# TODO: ALEX: Test that the constrains are appropriate when calling fit with a dataset