import os
import logging
import numpy as np
from tqdm import tqdm  # type: ignore
from copy import deepcopy
from typing import Union, Any, Callable, Optional, Tuple, Dict, List, TYPE_CHECKING
from abc import ABC, abstractmethod

from alibi.api.defaults import DEFAULT_META_CFRL, DEFAULT_DATA_CFRL
from alibi.api.interfaces import Explainer, Explanation, FitMixin
from alibi.utils.frameworks import has_pytorch, has_tensorflow, Framework

if TYPE_CHECKING:
    from alibi.models.tensorflow.autoencoder import AE as TensorflowAE
    from alibi.models.pytorch.autoencoder import AE as PytorchAE

if has_pytorch:
    # import pytorch backend
    import alibi.explainers.backends.pytorch.cfrl_base as pytorch_base_backend

if has_tensorflow:
    # import tensorflow backend
    import alibi.explainers.backends.tensorflow.cfrl_base as tensorflow_base_backend

# define logger
logger = logging.getLogger(__name__)


class NormalActionNoise:
    """ Normal noise generator. """

    def __init__(self, mu: float, sigma: float) -> None:
        """
        Constructor.

        Parameters
        ----------
        mu
            Mean of the normal noise.
        sigma
            Standard deviation of the noise.
        """
        self.mu = mu
        self.sigma = sigma

    def __call__(self, shape: Tuple[int, ...]) -> np.ndarray:
        """
        Generates normal noise with the appropriate mean and standard deviation.

        Parameters
        ----------
        shape
            Shape of the tensor to be generated

        Returns
        -------
        Normal noise with the appropriate mean, standard deviation and shape.
        """
        return self.mu + self.sigma * np.random.randn(*shape)

    def __repr__(self) -> str:
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class ReplayBuffer:
    """
    Circular experience replay buffer for `CounterfactualRL`(DDPG). When the buffer is filled, then the oldest
    experience is replaced by the new one (FIFO). The experience batch size is kept constant and inferred when
    the first batch of data is stored. Allowing flexible batch size can generate Tensorflow warning due to
    the `tf.function` retracing, which can lead to a drop in performance.
    """

    def __init__(self, size: int = 1000) -> None:
        """
        Constructor.

        Parameters
        ----------
        size
            Dimension of the buffer in batch size. This that the total memory allocated is proportional with the
            `size` * `batch_size`, where `batch_size` is inferred from the input tensors passed in the `append`
            method.
        """
        self.X, self.X_cf = None, None           # buffers for the input and the counterfactuals
        self.Y_m, self.Y_t = None, None          # buffers for the model's prediction and counterfactual target
        self.Z, self.Z_cf_tilde = None, None     # buffers for the input embedding and noised counterfactual embedding
        self.C = None                            # buffer for the conditional tensor
        self.R_tilde = None                      # buffer for the noised counterfactual reward tensor

        self.idx = 0                             # cursor for the buffer
        self.len = 0                             # current length of the buffer
        self.size = size                         # buffer's maximum capacity
        self.batch_size = None                   # batch size (inferred during `append`)

    def append(self,
               X: np.ndarray,
               X_cf: np.ndarray,
               Y_m: np.ndarray,
               Y_t: np.ndarray,
               Z: np.ndarray,
               Z_cf_tilde: np.ndarray,
               C: Optional[np.ndarray],
               R_tilde: np.ndarray,
               **kwargs) -> None:
        """
        Adds experience to the replay buffer. When the buffer is filled, then the oldest experience is replaced
        by the new one (FIFO).

        Parameters
        ----------
        X
            Input array.
        X_cf
            Counterfactual array.
        Y_m
            Model's prediction class of x.
        Y_t
            Counterfactual target class.
        Z
            Input's embedding.
        Z_cf_tilde
            Noised counterfactual embedding.
        C
            Conditional array.
        R_tilde
            Noised counterfactual reward array.
        """
        # Initialize the buffers.
        if self.X is None:
            self.batch_size = X.shape[0]

            # Allocate memory.
            self.X = np.zeros((self.size * self.batch_size, *X.shape[1:]), dtype=np.float32)
            self.X_cf = np.zeros((self.size * self.batch_size, *X_cf.shape[1:]), dtype=np.float32)
            self.Y_m = np.zeros((self.size * self.batch_size, *Y_m.shape[1:]), dtype=np.float32)
            self.Y_t = np.zeros((self.size * self.batch_size, *Y_t.shape[1:]), dtype=np.float32)
            self.Z = np.zeros((self.size * self.batch_size, *Z.shape[1:]), dtype=np.float32)
            self.Z_cf_tilde = np.zeros((self.size * self.batch_size, *Z_cf_tilde.shape[1:]), dtype=np.float32)
            self.R_tilde = np.zeros((self.size * self.batch_size, *R_tilde.shape[1:]), dtype=np.float32)

            # Conditional tensor can be `None` when no condition is included. If it is not `None`, allocate memory.
            if C is not None:
                self.C = np.zeros((self.size * self.batch_size, *C.shape[1:]), dtype=np.float32)

        # Increase the length of the buffer if not full.
        if self.len < self.size:
            self.len += 1

        # Compute the first position where to add most recent experience.
        start = self.batch_size * self.idx

        # Add new data / replace old experience (note that a full batch is added at once).
        self.X[start:start + self.batch_size] = X
        self.X_cf[start:start + self.batch_size] = X_cf
        self.Y_m[start:start + self.batch_size] = Y_m
        self.Y_t[start:start + self.batch_size] = Y_t
        self.Z[start:start + self.batch_size] = Z
        self.Z_cf_tilde[start:start + self.batch_size] = Z_cf_tilde
        self.R_tilde[start:start + self.batch_size] = R_tilde

        if C is not None:
            self.C[start:start + self.batch_size] = C

        # Compute the next index. Not that if the buffer reached its maximum capacity, for the next iteration
        # we start replacing old batches.
        self.idx = (self.idx + 1) % self.size

    def sample(self) -> Dict[str, Optional[np.ndarray]]:
        """
        Sample a batch of experience form the replay buffer.

        Returns
        --------
        A batch experience. For a description of the keys/values returned, see parameter descriptions in `append`
        method. The batch size returned is the same as the one passed in the `append`.
        """
        # Generate random indices to be sampled.
        rand_idx = np.random.randint(low=0, high=self.len * self.batch_size, size=(self.batch_size,))

        # Extract data form buffers.
        X = self.X[rand_idx]                                        # input array
        X_cf = self.X_cf[rand_idx]                                  # counterfactual
        Y_m = self.Y_m[rand_idx]                                    # model's prediction
        Y_t = self.Y_t[rand_idx]                                    # counterfactual target
        Z = self.Z[rand_idx]                                        # input embedding
        Z_cf_tilde = self.Z_cf_tilde[rand_idx]                      # noised counterfactual embedding
        C = self.C[rand_idx] if (self.C is not None) else None      # conditional array if exists
        R_tilde = self.R_tilde[rand_idx]                            # noised counterfactual reward

        return {
            "X": X,
            "X_cf": X_cf,
            "Y_m": Y_m,
            "Y_t": Y_t,
            "Z": Z,
            "Z_cf_tilde": Z_cf_tilde,
            "C": C,
            "R_tilde": R_tilde
        }


DEFAULT_BASE_PARAMS = {
    "act_noise": 0.1,
    "act_low": -1.0,
    "act_high": 1.0,
    "replay_buffer_size": 1000,
    "batch_size": 128,
    "num_workers": 4,
    "shuffle": True,
    "num_classes": 2,
    "exploration_steps": 100,
    "update_every": 1,
    "update_after": 10,
    "train_steps": 100000,
    "backend": "tensorflow",
    "ae_preprocessor": lambda x: x,
    "ae_inv_preprocessor": lambda x: x,
    "reward_func": lambda y_pred, y_true: y_pred == y_true,
    "postprocessing_funcs": [],
    "conditional_func": lambda x: None,
    "experience_callbacks": [],
    "train_callbacks": [],
    "actor": None,
    "critic": None,
    "optimizer_actor": None,
    "optimizer_critic": None,
    "lr_actor": 1e-3,
    "lr_critic": 1e-3,
    "actor_hidden_dim": 256,
    "critic_hidden_dim": 256,
}
"""
Default Counterfactual with Reinforcement Learning parameters.

    - ``'act_noise'``: float, standard deviation for the normal noise added to the actor for exploration.

    - ``'act_low'``: float, minimum action value. Each action component takes values between `[act_low, act_high]`.

    - ``'act_high'``: float, maximum action value. Each action component takes values between `[act_low, act_high]`.

    - ``'replay_buffer_size'``: int, dimension of the replay buffer in `batch_size` units. The total memory
    allocated is proportional with the `size` * `batch_size`.

    - ``'batch_size'``: int, training batch size.

    - ``'num_workers'``: int, number of workers used by the data loader if `pytorch` backend is selected.

    - ``'shuffle'``: bool, whether to shuffle the datasets every epoch.

    - ``'num_classes'``: int, number of classes to be considered.

    - ``'latent_dim'``: int, autoencoder latent dimension.

    - ``'exploration_steps'``: int, number of exploration steps. For the firts `exploration_steps`, the
    counterfactual embedding coordinates are sampled uniformly at random from the interval `[act_low, act_high]`.

    - ``'update_every'``: int, number of steps that should elapse between gradient updates. Regardless of the
    waiting steps, the ratio of waiting steps to gradient steps is locked to 1.

    - ``'update_after'``: int, number of steps to wait before start updating the actor and critic. This ensures that
    the replay buffers is full enough for useful updates.

    - ``'backend'``: str, backend to be used: `tensorflow`|`pytorch`. Default `tensorflow`.

    - ``'train_steps'``: int, number of train steps (interactions).

    - ``'ae_preprocessor'``: Callable, autoencoder data preprocessors. Transforms the input data into the format
    expected by the autoencoder. By default, the identity function.

    - ``'ae_inv_preprocessor'``: Callable, autoencoder data inverse preprocessor. Transforms data from the autoencoder
    expected format to the original input format. Before calling the prediction function, the data is inverse
    preprocessed to match the original input format. By default, the identity function.

    - ``'reward_func'``: Callable, element-wise reward function. By default, checks if the counterfactual prediction
    label matches the target label. Note that this is element-wise, so a tensor is expected to be returned.

    - ``'postprocessing_funcs'``: List[Postprocessing], post-processing list of functions. The function are applied in
    the order, from low to high index. Non-differentiable postprocessing can be applied. The function expects as
    arguments `x_cf` - the counterfactual instance, `x` - the original input instance and `c` - the conditional vector,
    and returns the post-processed counterfactual instance `x_cf_pp` which is passed as `x_cf` for the following
    functions. By default, no post-processing is applied (empty list).

    - ``'conditional_func'``: Callable, generates a conditional vector given a input instance. By default, the function
    returns `None` which is equivalent to no conditioning.

    - ``'experience_callbacks'``: List[ExperienceCallback], list of callback function applied at the end of each
    experience step.

    - ``'train_callbacks'``: List[TrainingCallback], list of callback functions applied at the end of each training
    step.

    - ``'actor'``: Optional[keras.Model, torch.nn.Module], actor network.

    - ``'critic;``: Optional[keras.Model, torch.nn.Module], critic network.

    - ``'optimizer_actor'``: Optional[keras.optimizers.Optimizer, torch.optim.Optimizer], actor optimizer.

    - ``'optimizer_critic'``: Optional[keras.optimizer.Optimizer, torch.optim.Optimizer], critic optimizer.

    - ``'lr_actor'``: float, actor learning rate.

    - ``'lr_critic'``: float, critic learning rate.

    - ``'actor_hidden_dim'``: int, actor hidden layer dimension.

    - ``'critic_hidden_dim'``: int, critic hidden layer dimension.
"""

PARAM_TYPES = {
    "primitives": [
        "act_noise", "act_low", "act_high", "replay_buffer_size", "batch_size", "num_workers", "shuffle",
        "num_classes", "exploration_steps", "update_every", "update_after", "train_steps", "backend",
        "actor_hidden_dim", "critic_hidden_dim",
    ],
    "complex": [
        "ae_preprocessor", "ae_inv_preprocessor", "reward_func", "postprocessing_funcs", "conditional_func",
        "experience_callbacks", "train_callbacks", "actor", "critic", "optimizer_actor", "optimizer_critic",
        "ae", "predictor", "sparsity_loss", "consistency_loss",
    ]
}
"""
Parameter types for serialization

    - ''`primitives`'': List[str], list of parameters having primitive data types.

    - ''`complex`'': List[str], list of parameters having complex data types (e.g., functions, models, optimizers etc.)
"""


class CounterfactualRLBase(Explainer, FitMixin):
    """ Counterfactual Reinforcement Learning Base. """

    def __init__(self,
                 predictor: Callable,
                 ae: Union['TensorflowAE', 'PytorchAE'],
                 latent_dim: int,
                 coeff_sparsity: float,
                 coeff_consistency: float,
                 num_classes: int,
                 backend: str = "tensorflow",
                 seed: int = 0,
                 **kwargs):
        """
        Constructor.

        Parameters
        ----------
        predictor
            Prediction function. This corresponds to the classifier.
        ae
            Pre-trained autoencoder.
        latent_dim
            Autoencoder latent dimension.
        coeff_sparsity
            Sparsity loss coefficient.
        coeff_consistency
            Consistency loss coefficient.
        num_classes
            Number of classes to be considered
        backend
            Deep learning backend: `tensorflow`|`pytorch`. Default `tensorflow`.
        seed
            Seed for reproducibility. The results are not reproducible for `tensorflow` backend.
        kwargs
            Used to replace any default parameter from :py:data:`alibi.expaliners.cfrl_base.DEFAULT_BASE_PARAMS`.
        """
        super().__init__(meta=deepcopy(DEFAULT_META_CFRL))

        # Clean backend flag.
        backend = backend.strip().lower()

        # Verify backend installed
        CounterfactualRLBase.verify_backend(backend)

        # Select backend.
        self.backend = self.select_backend(backend, **kwargs)

        # Set seed for reproducibility.
        self.backend.set_seed(seed)

        # Validate arguments.
        self.params, all_params = self.validate_kwargs(predictor=predictor,
                                                       ae=ae,
                                                       latent_dim=latent_dim,
                                                       coeff_sparsity=coeff_sparsity,
                                                       coeff_consistency=coeff_consistency,
                                                       num_classes=num_classes,
                                                       backend=backend,
                                                       seed=seed,
                                                       **kwargs)

        # If pytorch backend, the if GPU available, send everything to GPU
        if self.params["backend"] == Framework.PYTORCH:
            from alibi.explainers.backends.pytorch.cfrl_base import get_device
            self.params.update({"device": get_device()})

            # Send auto-encoder to device.
            self.params["ae"].to(self.params["device"])

            # Sent actor and critic to device.
            self.params["actor"].to(self.params["device"])
            self.params["critic"].to(self.params["device"])

        # update meta-data with all parameters passed (correct and incorrect)
        self.meta["params"].update(CounterfactualRLBase.serialize_params(all_params))

    @staticmethod
    def serialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parameter serialization. The function replaces object by human-readable representation

        Parameters
        ----------
        params
            Dictionary of parameters to be serialized.

        Returns
        -------
        Human-readable replacement of data.
        """
        meta = dict()

        for param, value in params.items():
            if param in PARAM_TYPES["primitives"]:
                # primitive types are passed as they are
                meta.update({param: value})

            elif param in PARAM_TYPES["complex"]:
                if isinstance(value, list):
                    # each complex element in the list is serialized by replacing it with a name
                    meta.update({param: [CounterfactualRLBase.get_name(v) for v in value]})
                else:
                    # complex element is serialized by replacing it with a name
                    meta.update({param: CounterfactualRLBase.get_name(value)})
            else:
                # Unknown parameters are passed as they are. TODO: think of a better way to handle this.
                meta.update({param: value})

        return meta

    @staticmethod
    def get_name(a: Any) -> str:
        """
        Constructs a name for the given object. If the object has as built-in name, the name is return.
        If the object has a built-in class name, the name of the class is returned. Otherwise `unknown` is returned.

        Parameters
        ----------
        a
            Object to give the name for.

        Returns
        -------
        Name of the object.
        """
        if hasattr(a, "__name__"):
            return a.__name__

        if hasattr(a, "__class__"):
            return str(a.__class__)

        return "unknown"

    @staticmethod
    def verify_backend(backend):
        """
        Verifies if the backend is supported.

        Parameters
        ----------
        backend
            Backend to be checked.
        """

        # Check if pytorch/tensorflow backend supported.
        if (backend == Framework.PYTORCH and not has_pytorch) or \
                (backend == Framework.TENSORFLOW and not has_tensorflow):
            raise ImportError(f'{backend} not installed. Cannot initialize and run the CounterfactualRL'
                              f' with {backend} backend.')

        # Allow only pytorch and tensorflow.
        elif backend not in [Framework.PYTORCH, Framework.TENSORFLOW]:
            raise NotImplementedError(f'{backend} not implemented. Use `tensorflow` or `pytorch` instead.')

    def select_backend(self, backend, **kwargs):
        """
        Selects the backend according to the `backend` flag.

        Parameters
        ---------
        backend
            Deep learning backend: `tensorflow`|`pytorch`. Default `tensorflow`.
        """
        return tensorflow_base_backend if backend == "tensorflow" else pytorch_base_backend

    def validate_kwargs(self,
                        predictor: Callable,
                        ae: Union['TensorflowAE', 'PytorchAE'],
                        latent_dim: float,
                        coeff_sparsity: float,
                        coeff_consistency: float,
                        num_classes: int,
                        backend: str,
                        seed: int,
                        **kwargs):
        """
        Validates arguments.

        Parameters
        ----------
        predictor.
            Prediction function. This corresponds to the classifier.
        ae
            Pre-trained autoencoder.
        latent_dim
            Autoencoder latent dimension.
        coeff_sparsity
            Sparsity loss coefficient.
        coeff_consistency
            Consistency loss coefficient.
        num_classes
            Number of classes to consider.
        backend
            Deep learning backend: `tensorflow`|`pytorch`.
        """
        # Copy default parameters.
        params = deepcopy(DEFAULT_BASE_PARAMS)

        # Update parameters with mandatory arguments
        params.update({
            "ae": ae,
            "latent_dim": latent_dim,
            "predictor": predictor,
            "coeff_sparsity": coeff_sparsity,
            "coeff_consistency": coeff_consistency,
            "num_classes": num_classes,
            "backend": backend,
            "seed": seed,
        })

        # Add actor if not user-specified.
        not_specified = {"actor": False, "critic": False}
        if "actor" not in kwargs:
            not_specified["actor"] = True
            params["actor"] = self.backend.get_actor(hidden_dim=params["actor_hidden_dim"],
                                                     output_dim=params["latent_dim"])

        if "critic" not in kwargs:
            not_specified["critic"] = True
            params["critic"] = self.backend.get_critic(hidden_dim=params["critic_hidden_dim"])

        # Add optimizers if not user-specified.
        optimizers = ["optimizer_actor", "optimizer_critic"]

        for optim in optimizers:
            # extract model in question
            model_name = optim.split("_")[1]
            model = params[model_name]
            lr = params["lr_" + model_name]

            # If the optimizer is user-specified, just update the params
            if optim in kwargs:
                params.update({optim: kwargs[optim]})
                if self.params["backend"] == Framework.PYTORCH and not_specified[model_name]:
                    raise ValueError(f"Can not specify {optim} when {model_name} not specified for pytorch backend.")

            # If the optimizer is not user-specified, it need to be initialized. The initialization is backend specific.
            elif params['backend'] == Framework.TENSORFLOW:
                params.update({optim: self.backend.get_optimizer(lr=lr)})
            else:
                params.update({optim: self.backend.get_optimizer(model=model, lr=lr)})

        # Add sparsity loss if not user-specified.
        params["sparsity_loss"] = self.backend.sparsity_loss if "sparsity_loss" not in kwargs \
            else kwargs["sparsity_loss"]

        # Add consistency loss if not user-specified.
        params["consistency_loss"] = self.backend.consistency_loss if "consistency_loss" not in kwargs \
            else kwargs["consistency_loss"]

        # Define dictionary of all parameters. Shallow copy of values.
        all_params = params.copy()

        # Validate arguments.
        allowed_keys = set(params.keys())
        provided_keys = set(kwargs.keys())
        common_keys = allowed_keys & provided_keys

        # Check if some provided keys are incorrect
        if len(common_keys) < len(provided_keys):
            incorrect_keys = ", ".join(provided_keys - common_keys)
            logger.warning("The following keys are incorrect: " + incorrect_keys)

        # Update default parameters and all parameters
        params.update({key: kwargs[key] for key in common_keys})
        all_params.update(kwargs)
        return params, all_params

    @classmethod
    def load(cls, path: Union[str, os.PathLike], predictor: Any) -> "Explainer":
        return super().load(path, predictor)

    def reset_predictor(self, predictor: Any) -> None:
        self.params["predictor"] = predictor
        self.meta.update(CounterfactualRLBase.serialize_params(self.params))

    def save(self, path: Union[str, os.PathLike]) -> None:
        super().save(path)

    def fit(self, X: np.ndarray, ) -> "Explainer":
        """
        Fit the model agnostic counterfactual generator.

        Parameters
        ----------
        X
            Training data array.

        Returns
        -------
        self
            The explainer itself.
        """
        # Define replay buffer (this will deal only with numpy arrays).
        replay_buff = ReplayBuffer(size=self.params["replay_buffer_size"])

        # Define noise variable.
        noise = NormalActionNoise(mu=0, sigma=self.params["act_noise"])

        # Define data generator.
        data_generator = self.backend.data_generator(X=X, **self.params)
        data_iter = iter(data_generator)

        for step in tqdm(range(self.params["train_steps"])):
            # Sample training data.
            try:
                data = next(data_iter)
            except StopIteration:
                data_iter = iter(data_generator)
                data = next(data_iter)

            # Add None condition if condition does not exist.
            if "C" not in data:
                data["C"] = None

            # Compute input embedding.
            Z = self.backend.encode(X=data["X"], **self.params)
            data.update({"Z": Z})

            # Compute counterfactual embedding.
            Z_cf = self.backend.generate_cf(step=step, **data, **self.params)
            data.update({"Z_cf": Z_cf})

            # Add noise to the counterfactual embedding.
            Z_cf_tilde = self.backend.add_noise(noise=noise, step=step, **data, **self.params)
            data.update({"Z_cf_tilde": Z_cf_tilde})

            # Decode counterfactual and apply postprocessing step to x_cf_tilde.
            X_cf = self.backend.decode(Z=data["Z_cf"], **self.params)
            X_cf_tilde = self.backend.decode(Z=data["Z_cf_tilde"], **self.params)

            for pp_func in self.params["postprocessing_funcs"]:
                # Post-process counterfactual.
                X_cf = pp_func(self.backend.to_numpy(X_cf),
                               self.backend.to_numpy(data["X"]),
                               self.backend.to_numpy(data["C"]))

                # Post-process noised counterfactual.
                X_cf_tilde = pp_func(self.backend.to_numpy(X_cf_tilde),
                                     self.backend.to_numpy(data["X"]),
                                     self.backend.to_numpy(data["C"]))

            data.update({
                "X_cf": X_cf,
                "X_cf_tilde": X_cf_tilde
            })

            # Compute model's prediction on the noised counterfactual
            X_cf_tilde = self.params["ae_inv_preprocessor"](self.backend.to_numpy(data["X_cf_tilde"]))
            Y_m_cf_tilde = self.params["predictor"](X_cf_tilde)

            # Compute reward.
            R_tilde = self.params["reward_func"](self.backend.to_numpy(Y_m_cf_tilde),
                                                 self.backend.to_numpy(data["Y_t"]))
            data.update({"R_tilde": R_tilde, "Y_m_cf_tilde": Y_m_cf_tilde})

            # Store experience in the replay buffer.
            data = {key: self.backend.to_numpy(data[key]) for key in data.keys()}
            replay_buff.append(**data)

            # Call all experience callbacks.
            for exp_cb in self.params["experience_callbacks"]:
                exp_cb(step=step, model=self, sample=data)

            if step % self.params['update_every'] == 0 and step > self.params["update_after"]:
                for i in range(self.params['update_every']):
                    # Sample batch of experience form the replay buffer.
                    sample = replay_buff.sample()

                    if "C" not in sample:
                        sample["C"] = None

                    # Update critic by one-step gradient descent.
                    losses = self.backend.update_actor_critic(**sample, **self.params)

                    # Convert all losses from tensors to numpy arrays.
                    losses = {key: self.backend.to_numpy(losses[key]).item() for key in losses.keys()}

                    # Call all train callbacks.
                    for train_cb in self.params["train_callbacks"]:
                        train_cb(step=step, update=i, model=self, sample=sample, losses=losses)

        return self

    def explain(self,
                X: np.ndarray,
                Y_t: np.ndarray = None,  # TODO: remove default value (mypy error)
                C: Any = None,
                batch_size: int = 100) -> Explanation:
        """
        Explains an input instance

        Parameters
        ----------
        X
            Instances to be explained.
        Y_t
            Counterfactual targets.
        C
            Conditional vector.
        batch_size
            Batch size to be used in a forward pass.


        Returns
        -------
        `Explanation` object containing the inputs with the corresponding labels, the counterfactuals with the
        corresponding labels, targets and additional metadata.
        """
        # Check the number of target labels.
        if Y_t.shape[0] != 1 and Y_t.shape[0] != X.shape[0]:
            raise ValueError("The number target labels should be 1 or equals the number of samples in X.")

        if (C is not None) and C.shape[0] != 1 and C.shape[0] != X.shape[0]:
            raise ValueError("The number of conditional vectors should be 1 or equals the number if samples in X,")

        # Repeat the same label to match the number of input instances.
        if Y_t.shape[0] == 1:
            Y_t = np.tile(Y_t, X.shape[0])

        # Repeat the same conditional vectors to match the number of input instances.
        if (C is not None) and C.shape[0] == 1:
            C = np.tile(C, X.shape[0])

        # Perform prediction in mini-batches.
        n_minibatch = int(np.ceil(X.shape[0] / batch_size))
        all_results: Dict[str, np.ndarray] = {}

        for i in range(n_minibatch):
            istart, istop = i * batch_size, min((i + 1) * batch_size, X.shape[0])
            results = self.compute_counterfactual(X=X[istart:istop],
                                                  Y_t=Y_t[istart:istop],
                                                  C=C[istart:istop] if (C is not None) else C)
            # initialize the dict
            if not all_results:
                all_results = results
                continue

            # append the new batch off results
            for key in all_results:
                if all_results[key] is not None:
                    all_results[key] = np.concatenate([all_results[key], results[key]], axis=0)

        return self.build_explanation(**all_results)

    def compute_counterfactual(self,
                               X: np.ndarray,
                               Y_t: np.ndarray,
                               C: Optional[np.ndarray] = None) -> Dict[str, Optional[np.ndarray]]:
        """
        Compute counterfactual instance for a given input, target and condition vector

        Parameters
        ----------
        X
            Instances to be explained.
        Y_t
            Counterfactual targets.
        C
            Conditional vector.

        Returns
        -------
        Dictionary containing the input instances in the original format, input classification labels, counterfactual
        instances in the original format, counterfactual classification labels, target labels, conditional vectors.
        """
        # Compute models prediction.
        X_orig = X
        Y_m = self.params["predictor"](X_orig)

        # Apply autoencoder preprocessing step.
        X = self.params["ae_preprocessor"](X_orig)

        # Convert to tensors.
        X = self.backend.to_tensor(X, **self.params)
        Y_m = self.backend.to_tensor(Y_m, **self.params)
        Y_t = self.backend.to_tensor(Y_t, **self.params)

        # Encode instance.
        Z = self.backend.encode(X, **self.params)

        # Generate counterfactual embedding.
        Z_cf = self.backend.generate_cf(Z, Y_m, Y_t, C, **self.params)

        # Decode counterfactual.
        X_cf = self.backend.decode(Z_cf, **self.params)
        X_cf = self.backend.to_numpy(X_cf)

        # Apply postprocessing functions.
        for pp_func in self.params["postprocessing_funcs"]:
            X_cf = pp_func(X_cf, X, C)

        # Apply inverse autoencoder pre-processor.
        X_cf = self.params["ae_inv_preprocessor"](X_cf)

        # Classify counterfactual instances.
        y_m_cf = self.params["predictor"](X_cf)

        # convert tensors to numpy
        Y_m = self.backend.to_numpy(Y_m)
        Y_t = self.backend.to_numpy(Y_t)

        return {
            "X": X_orig,         # input instances
            "Y_m": Y_m,          # input classification labels
            "X_cf": X_cf,        # counterfactual instances
            "Y_m_cf": y_m_cf,    # counterfactual classification labels
            "Y_t": Y_t,          # target labels
            "C": C               # conditional vectors
        }

    def build_explanation(self,
                          X: np.ndarray,
                          Y_m: np.ndarray,
                          X_cf: np.ndarray,
                          Y_m_cf: np.ndarray,
                          Y_t: np.ndarray,
                          C: Any) -> Explanation:
        """
        Builds the explanation of the current object.

        Parameters
        ----------
        X
            Inputs instance in the original format.
        Y_m
            Inputs classification labels.
        X_cf
            Counterfactuals instances in the original format.
        Y_m_cf
            Counterfactuals classification labels.
        Y_t
            Target labels.
        C
            Condition
        Returns
        -------
        `Explanation` object containing the inputs with the corresponding labels, the counterfactuals with the
        corresponding labels, targets and additional metadata.
        """
        data = deepcopy(DEFAULT_DATA_CFRL)

        # update original input entrance
        data["orig"] = {}
        data["orig"].update({"X": X, "class": Y_m})

        # update counterfactual entrance
        data["cf"] = {}
        data["cf"].update({"X": X_cf, "class": Y_m_cf})

        # update target and condition
        data["target"] = Y_t
        data["condition"] = C
        return Explanation(meta=self.meta, data=data)


class Postprocessing(ABC):
    @abstractmethod
    def __call__(self, X_cf: Union[np.ndarray, List[np.ndarray]], X: np.ndarray, C: np.ndarray) -> Any:
        """
        Post-processing function

        Parameters
        ----------
        X_cf
           Counterfactual instance. The datatype depends on the output of the decoder. For example, for an image
           dataset, the output is `np.ndarray`. For a tabular dataset, the output is `List[np.ndarray]` where each
           element of the list corresponds to a feature. This corresponds to the decoder's output from the
           heterogeneous autoencoder (see :py:class:`alibi.models.tensorflow.autoencoder.HeAE` and
           :py:class:`alibi.models.pytorch.autoencoder.HeAE`).
        X
           Input instance.
        C
           Conditional vector.

        Returns
        -------
        X_cf
            Post-processed X_cf.
        """
        pass


class ExperienceCallback:
    """
    Experience callback class. This is not an ABC since it can not be pickled.
    TODO: Maybe go for something else?
    """

    def __call__(self,
                 step: int,
                 model: CounterfactualRLBase,
                 sample: Dict[str, np.ndarray]) -> None:
        """
        Experience call-back applied after gather an experience.

        Parameters
        ----------
        step
            Current experience step.
        model
            CounterfactualRLBase explainer.
        sample
            Dictionary of sample gathered in an experience. This includes dataset inputs and intermediate results
            obtained during an experience.
        """
        pass


class TrainingCallback:
    """
    Training callback class. This is not an ABC since it can not be pickled.
    TODO: Maybe go for something else?
    """

    def __call__(self,
                 step: int,
                 update: int,
                 model: CounterfactualRLBase,
                 sample: Dict[str, np.ndarray],
                 losses: Dict[str, float]) -> None:
        """
        Training call-back applied after every training step.

        Parameters
        -----------
        step
            Current experience step.
        update
            Current update. The ration between the number experience steps and the number of training updates is
            bound to 1.
        model
            CounterfactualRLBase explainer.
        sample
            Dictionary of samples used for an update. This is sampled from the replay buffer.
        losses
            Dictionary of losses.
        """
        pass
