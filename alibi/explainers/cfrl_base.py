import os
import logging
import numpy as np
from tqdm import tqdm  # type: ignore
from copy import deepcopy
from typing import Union, Any, Callable, Optional, Tuple, Dict, TYPE_CHECKING
from abc import ABC, abstractmethod

from alibi.api.defaults import DEFAULT_META_CFRL, DEFAULT_DATA_CFRL
from alibi.api.interfaces import Explainer, Explanation, FitMixin
from alibi.utils.frameworks import has_pytorch, has_tensorflow, Framework
from alibi.explainers.backends.cfrl_base import identity_function, generate_empty_condition,\
    get_classification_reward, get_hard_distribution

if TYPE_CHECKING:
    import torch
    import tensorflow

if has_pytorch:
    # import pytorch backend
    from alibi.explainers.backends.pytorch import cfrl_base as pytorch_base_backend

if has_tensorflow:
    # import tensorflow backend
    from alibi.explainers.backends.tensorflow import cfrl_base as tensorflow_base_backend

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
    Circular experience replay buffer for `CounterfactualRL` (DDPG). When the buffer is filled, then the oldest
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
            `size` * `batch_size`, where `batch_size` is inferred from the first tensors to be stored.
        """
        self.X: Optional[np.ndarray] = None            # buffer for the inputs
        self.Y_m: Optional[np.ndarray] = None          # buffer for the model's prediction
        self.Y_t: Optional[np.ndarray] = None          # buffer for the counterfactual targets
        self.Z: Optional[np.ndarray] = None            # buffer for the input embedding
        self.Z_cf_tilde: Optional[np.ndarray] = None   # buffer for the noised counterfactual embedding
        self.C: Optional[np.ndarray] = None            # buffer for the conditional tensor
        self.R_tilde: Optional[np.ndarray] = None      # buffer for the noised counterfactual reward tensor

        self.idx = 0                    # cursor for the buffer
        self.len = 0                    # current length of the buffer
        self.size = size                # buffer's maximum capacity
        self.batch_size = 0             # batch size (inferred during `append`)

    def append(self,
               X: np.ndarray,
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
        -------
            A batch experience. For a description of the keys and values returned, see parameter descriptions \
            in :py:meth:`alibi.explainers.cfrl_base.ReplayBuffer.append` method. The batch size returned is the same \
            as the one passed in the :py:meth:`alibi.explainers.cfrl_base.ReplayBuffer.append`.
        """
        # Generate random indices to be sampled.
        rand_idx = np.random.randint(low=0, high=self.len * self.batch_size, size=(self.batch_size,))

        # Extract data form buffers.
        X = self.X[rand_idx]                                        # input array
        Y_m = self.Y_m[rand_idx]                                    # model's prediction
        Y_t = self.Y_t[rand_idx]                                    # counterfactual target
        Z = self.Z[rand_idx]                                        # input embedding
        Z_cf_tilde = self.Z_cf_tilde[rand_idx]                      # noised counterfactual embedding
        C = self.C[rand_idx] if (self.C is not None) else None      # conditional array if exists
        R_tilde = self.R_tilde[rand_idx]                            # noised counterfactual reward

        return {
            "X": X,
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
    "batch_size": 100,
    "num_workers": 4,
    "shuffle": True,
    "exploration_steps": 100,
    "update_every": 1,
    "update_after": 10,
    "train_steps": 100000,
    "backend": "tensorflow",
    "encoder_preprocessor": identity_function,
    "decoder_inv_preprocessor": identity_function,
    "reward_func": get_classification_reward,
    "postprocessing_funcs": [],
    "conditional_func": generate_empty_condition,
    "callbacks": [],
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

    - ``'replay_buffer_size'``: int, dimension of the replay buffer in `batch_size` units. The total memory \
    allocated is proportional with the `size` * `batch_size`.

    - ``'batch_size'``: int, training batch size.

    - ``'num_workers'``: int, number of workers used by the data loader if `pytorch` backend is selected.

    - ``'shuffle'``: bool, whether to shuffle the datasets every epoch.

    - ``'exploration_steps'``: int, number of exploration steps. For the firts `exploration_steps`, the \
    counterfactual embedding coordinates are sampled uniformly at random from the interval `[act_low, act_high]`.

    - ``'update_every'``: int, number of steps that should elapse between gradient updates. Regardless of the \
    waiting steps, the ratio of waiting steps to gradient steps is locked to 1.

    - ``'update_after'``: int, number of steps to wait before start updating the actor and critic. This ensures that \
    the replay buffers is full enough for useful updates.

    - ``'backend'``: str, backend to be used: `tensorflow` | `pytorch`. Default `tensorflow`.

    - ``'train_steps'``: int, number of train steps.

    - ``'encoder_preprocessor'``: Callable, encoder/autoencoder data preprocessors. Transforms the input data into the \
    format expected by the autoencoder. By default, the identity function.

    - ``'decoder_inv_preprocessor'``: Callable, decoder/autoencoder data inverse preprocessor. Transforms data from \
    the autoencoder output format to the original input format. Before calling the prediction function, the data is \
    inverse preprocessed to match the original input format. By default, the identity function.

    - ``'reward_func'``: Callable, element-wise reward function. By default, considers classification task and \
    checks if the counterfactual prediction label matches the target label. Note that this is element-wise, so a \
    tensor is expected to be returned.

    - ``'postprocessing_funcs'``: List[Postprocessing], list of post-processing functions. The function are applied in \
    the order, from low to high index. Non-differentiable post-processing can be applied. The function expects as \
    arguments `X_cf` - the counterfactual instance, `X` - the original input instance and `C` - the conditional \
    vector, and returns the post-processed counterfactual instance `X_cf_pp` which is passed as `X_cf` for the \
    following functions. By default, no post-processing is applied (empty list).

    - ``'conditional_func'``: Callable, generates a conditional vector given a pre-processed input instance. By \
    default, the function returns `None` which is equivalent to no conditioning.

    - ``'callbacks'``: List[Callback], list of callback functions applied at the end of each training step.

    - ``'actor'``: Optional[Union[tensorflow.keras.Model, torch.nn.Module]], actor network.

    - ``'critic;``: Optional[Union[tensorflow.keras.Model, torch.nn.Module]], critic network.

    - ``'optimizer_actor'``: Optional[Union[tensorflow.keras.optimizers.Optimizer, torch.optim.Optimizer]], actor \
    optimizer.

    - ``'optimizer_critic'``: Optional[Union[tensorflow.keras.optimizer.Optimizer, torch.optim.Optimizer]], critic \
    optimizer.

    - ``'lr_actor'``: float, actor learning rate.

    - ``'lr_critic'``: float, critic learning rate.

    - ``'actor_hidden_dim'``: int, actor hidden layer dimension.

    - ``'critic_hidden_dim'``: int, critic hidden layer dimension.
"""

_PARAM_TYPES = {
    "primitives": [
        "act_noise", "act_low", "act_high", "replay_buffer_size", "batch_size", "num_workers", "shuffle",
        "exploration_steps", "update_every", "update_after", "train_steps", "backend", "actor_hidden_dim",
        "critic_hidden_dim",
    ],
    "complex": [
        "encoder_preprocessor", "decoder_inv_preprocessor", "reward_func", "postprocessing_funcs", "conditional_func",
        "callbacks", "actor", "critic", "optimizer_actor", "optimizer_critic", "encoder", "decoder", "predictor",
        "sparsity_loss", "consistency_loss",
    ]
}
"""
Parameter types for serialization

    - ``'primitives'``: List[str], list of parameters having primitive data types.

    - ``'complex'``: List[str], list of parameters having complex data types (e.g., functions, models, optimizers etc.)
"""


class CounterfactualRL(Explainer, FitMixin):
    """ Counterfactual Reinforcement Learning. """

    def __init__(self,
                 predictor: Callable,
                 encoder: 'Union[tensorflow.keras.Model, torch.nn.Module]',
                 decoder: 'Union[tensorflow.keras.Model, torch.nn.Module]',
                 coeff_sparsity: float,
                 coeff_consistency: float,
                 latent_dim: Optional[int] = None,
                 backend: str = "tensorflow",
                 seed: int = 0,
                 **kwargs):
        """
        Constructor.

        Parameters
        ----------
        predictor
            A callable that takes a tensor of N data points as inputs and returns N outputs. For classification task,
            the second dimension of the output should match the number of classes. Thus, the output can be either
            a soft label distribution or a hard label distribution (i.e. one-hot encoding) without affecting the
            performance since `argmax` is applied to the predictor's output.
        encoder
            Pretrained encoder network.
        decoder
            Pretrained decoder network.
        coeff_sparsity
            Sparsity loss coefficient.
        coeff_consistency
            Consistency loss coefficient.
        latent_dim
            Autoencoder latent dimension. Can be omitted if the actor network is user specified.
        backend
            Deep learning backend: `tensorflow` | `pytorch`. Default `tensorflow`.
        seed
            Seed for reproducibility. The results are not reproducible for `tensorflow` backend.
        kwargs
            Used to replace any default parameter from :py:data:`alibi.explainers.cfrl_base.DEFAULT_BASE_PARAMS`.
        """
        super().__init__(meta=deepcopy(DEFAULT_META_CFRL))

        # Clean backend flag.
        backend = backend.strip().lower()

        # Verify backend installed
        CounterfactualRL._verify_backend(backend)

        # Select backend.
        self.backend = self._select_backend(backend, **kwargs)

        # Set seed for reproducibility.
        self.backend.set_seed(seed)

        # Validate arguments.
        self.params = self._validate_kwargs(predictor=predictor,
                                            encoder=encoder,
                                            decoder=decoder,
                                            latent_dim=latent_dim,
                                            coeff_sparsity=coeff_sparsity,
                                            coeff_consistency=coeff_consistency,
                                            backend=backend,
                                            seed=seed,
                                            **kwargs)

        # If pytorch backend, the if GPU available, send everything to GPU
        if self.params["backend"] == Framework.PYTORCH:
            from alibi.explainers.backends.pytorch.cfrl_base import get_device
            self.params.update({"device": get_device()})

            # Send encoder and decoder to device.
            self.params["encoder"].to(self.params["device"])
            self.params["decoder"].to(self.params["device"])

            # Sent actor and critic to device.
            self.params["actor"].to(self.params["device"])
            self.params["critic"].to(self.params["device"])

        # Update meta-data with all parameters passed (correct and incorrect).
        self.meta["params"].update(CounterfactualRL._serialize_params(self.params))

    @staticmethod
    def _serialize_params(params: Dict[str, Any]) -> Dict[str, Any]:
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
            if param in _PARAM_TYPES["primitives"]:
                # primitive types are passed as they are
                meta.update({param: value})

            elif param in _PARAM_TYPES["complex"]:
                if isinstance(value, list):
                    # each complex element in the list is serialized by replacing it with a name
                    meta.update({param: [CounterfactualRL._get_name(v) for v in value]})
                else:
                    # complex element is serialized by replacing it with a name
                    meta.update({param: CounterfactualRL._get_name(value)})
            else:
                # Unknown parameters are passed as they are. TODO: think of a better way to handle this.
                meta.update({param: value})

        return meta

    @staticmethod
    def _get_name(a: Any) -> str:
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
    def _verify_backend(backend):
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

    def _select_backend(self, backend, **kwargs):
        """
        Selects the backend according to the `backend` flag.

        Parameters
        ---------
        backend
            Deep learning backend: `tensorflow` | `pytorch`. Default `tensorflow`.
        """
        return tensorflow_base_backend if backend == "tensorflow" else pytorch_base_backend

    def _validate_kwargs(self,
                         predictor: Callable,
                         encoder: 'Union[tensorflow.keras.Model, torch.nn.Module]',
                         decoder: 'Union[tensorflow.keras.Model, torch.nn.Module]',
                         latent_dim: float,
                         coeff_sparsity: float,
                         coeff_consistency: float,
                         backend: str,
                         seed: int,
                         **kwargs):
        """
        Validates arguments.

        Parameters
        ----------
        predictor.
            A callable that takes a tensor of N data points as inputs and returns N outputs.
        encoder
            Pretrained encoder network.
        decoder
            Pretrained decoder network.
        latent_dim
            Autoencoder latent dimension.
        coeff_sparsity
            Sparsity loss coefficient.
        coeff_consistency
            Consistency loss coefficient.
        backend
            Deep learning backend: `tensorflow` | `pytorch`.
        """
        # Copy default parameters.
        params = deepcopy(DEFAULT_BASE_PARAMS)

        # Update parameters with mandatory arguments
        params.update({
            "encoder": encoder,
            "decoder": decoder,
            "latent_dim": latent_dim,
            "predictor": predictor,
            "coeff_sparsity": coeff_sparsity,
            "coeff_consistency": coeff_consistency,
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
        return params

    @classmethod
    def load(cls, path: Union[str, os.PathLike], predictor: Any) -> "Explainer":
        return super().load(path, predictor)

    def reset_predictor(self, predictor: Any) -> None:
        """
        Resets the predictor to be explained.

        Parameters
        ----------
        predictor
            New predictor to be set.
        """
        self.params["predictor"] = predictor
        self.meta["params"].update(CounterfactualRL._serialize_params(self.params))

    def save(self, path: Union[str, os.PathLike]) -> None:
        super().save(path)

    def fit(self, X: np.ndarray) -> "Explainer":
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
        # Define boolean flag for initializing actor and critic network for Tensorflow backend.
        initialize_actor_critic = False

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
                if hasattr(data_generator, "on_epoch_end"):
                    # This is just for tensorflow backend.
                    data_generator.on_epoch_end()

                data_iter = iter(data_generator)
                data = next(data_iter)

            # Add None condition if condition does not exist.
            if "C" not in data:
                data["C"] = None

            # Compute input embedding.
            Z = self.backend.encode(X=data["X"], **self.params)
            data.update({"Z": Z})

            # Compute counterfactual embedding.
            Z_cf = self.backend.generate_cf(**data, **self.params)
            data.update({"Z_cf": Z_cf})

            # Add noise to the counterfactual embedding.
            Z_cf_tilde = self.backend.add_noise(noise=noise, step=step, **data, **self.params)
            data.update({"Z_cf_tilde": Z_cf_tilde})

            # Decode noised counterfactual and apply postprocessing step to X_cf_tilde.
            X_cf_tilde = self.backend.decode(Z=data["Z_cf_tilde"], **self.params)

            for pp_func in self.params["postprocessing_funcs"]:
                # Post-process noised counterfactual.
                X_cf_tilde = pp_func(self.backend.to_numpy(X_cf_tilde),
                                     self.backend.to_numpy(data["X"]),
                                     self.backend.to_numpy(data["C"]))
            data.update({"X_cf_tilde": X_cf_tilde})

            # Compute model's prediction on the noised counterfactual
            X_cf_tilde = self.params["decoder_inv_preprocessor"](self.backend.to_numpy(data["X_cf_tilde"]))
            Y_m_cf_tilde = self.params["predictor"](X_cf_tilde)

            # Compute reward.
            R_tilde = self.params["reward_func"](self.backend.to_numpy(Y_m_cf_tilde),
                                                 self.backend.to_numpy(data["Y_t"]))
            data.update({"R_tilde": R_tilde, "Y_m_cf_tilde": Y_m_cf_tilde})

            # Store experience in the replay buffer.
            data = {key: self.backend.to_numpy(data[key]) for key in data.keys()}
            replay_buff.append(**data)

            if step % self.params['update_every'] == 0 and step > self.params["update_after"]:
                for i in range(self.params['update_every']):
                    # Sample batch of experience form the replay buffer.
                    sample = replay_buff.sample()

                    # Initialize actor and critic. This is required for tensorflow in order to reinitialize the
                    # explainer object and call fit multiple times. If the models are not reinitialized, the
                    # error: "tf.function-decorated function tried to create variables on non-first call" is raised.
                    # This is due to @tf.function and building the model for the first time in a compiled function
                    if not initialize_actor_critic and self.params["backend"] == Framework.TENSORFLOW:
                        self.backend.initialize_actor_critic(**sample, **self.params)
                        self.backend.initialize_optimizers(**sample, **self.params)
                        initialize_actor_critic = True

                    if "C" not in sample:
                        sample["C"] = None

                    # Decode counterfactual. This procedure has to be done here and not in the experience loop
                    # since the actor is updating but old experience is used. Thus, the decoding of the counterfactual
                    # will not correspond to the latest actor network. Remember that the counterfactual is used
                    # for the consistency loss. The counterfactual generation is performed here due to @tf.function
                    # which does not allow all post-processing functions.
                    Z_cf = self.backend.generate_cf(Z=self.backend.to_tensor(sample["Z"], **self.params),
                                                    Y_m=self.backend.to_tensor(sample["Y_m"], **self.params),
                                                    Y_t=self.backend.to_tensor(sample["Y_t"], **self.params),
                                                    C=self.backend.to_tensor(sample["C"], **self.params),
                                                    **self.params)

                    X_cf = self.backend.decode(Z=Z_cf, **self.params)
                    for pp_func in self.params["postprocessing_funcs"]:
                        # Post-process counterfactual.
                        X_cf = pp_func(self.backend.to_numpy(X_cf),
                                       self.backend.to_numpy(sample["X"]),
                                       self.backend.to_numpy(sample["C"]))

                    # Add counterfactual instance to the sample to be used in the update function for consistency loss
                    sample.update({"Z_cf": self.backend.to_numpy(Z_cf),
                                   "X_cf": self.backend.to_numpy(X_cf)})

                    # Update critic by one-step gradient descent.
                    losses = self.backend.update_actor_critic(**sample, **self.params)

                    # Convert all losses from tensors to numpy arrays.
                    losses = {key: self.backend.to_numpy(losses[key]).item() for key in losses.keys()}

                    # Call all callbacks.
                    for callback in self.params["callbacks"]:
                        callback(step=step, update=i, model=self, sample=sample, losses=losses)
        return self

    @staticmethod
    def _validate_target(Y_t: Optional[np.ndarray]):
        """
        Validate the targets by checking the dimensions.

        Parameters
        ----------
        Y_t
            Targets to be checked.
        """
        if Y_t is None:
            raise ValueError("Target can not be `None`.")

        if len(Y_t.shape) not in [1, 2]:
            raise ValueError(f"Target shape should be at least 1 and at most 2. Found {len(Y_t.shape)} instead.")

    @staticmethod
    def _validate_condition(C: np.ndarray):
        """
        Validate condition vector.

        Parameters
        ----------
        C
            Condition vector.
        """
        if (C is not None) and len(C.shape) != 2:
            raise ValueError(f"Condition vector shape should be 2. Found {len(C.shape)} instead.")

    @staticmethod
    def _is_classification(pred: np.ndarray) -> bool:
        """
        Check if the prediction task is classification by looking at the model's prediction shape.

        Parameters
        ----------
        pred
            Model's prediction.

        Returns
        -------
        `True` if the prediction has shape of 2 and the second dimension bigger grater than 1. `False` otherwise.
        """
        return len(pred.shape) == 2 and pred.shape[1] > 1

    def explain(self,
                X: np.ndarray,
                Y_t: np.ndarray = None,   # TODO: remove default value (mypy error. explanation in the validation step)
                C: Optional[Any] = None,  # TODO: narrow down the type from `Any` (explanation in the validation step)
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
            Conditional vectors. If `None`, it means that no conditioning was used during training (i.e. the
            `conditional_func` returns `None`).
        batch_size
            Batch size to be used when generating counterfactuals.

        Returns
        -------
        `Explanation` object containing the inputs with the corresponding labels, the counterfactuals with the \
        corresponding labels, targets and additional metadata.
        """
        # General validation.
        #
        # If `Y_t` doesn't have the default value `None` I will get a warning saying that "Signature of method does not
        # match the signature of class" (Liskov substitution principle). That's why `Y_t` can be None but None is not a
        # valid value since the target must be specified.
        #
        # `C` can be in fact only `np.ndarray`, but for the tabular case a `Dict[str, List]` is expected.
        # Similar behavior as in the previous comment.
        self._validate_target(Y_t)
        self._validate_condition(C)

        # Check the number of target labels.
        if Y_t.shape[0] != 1 and Y_t.shape[0] != X.shape[0]:
            raise ValueError("The number target labels should be 1 or equals the number of samples in X.")

        # Check the number of conditional vectors
        if (C is not None) and C.shape[0] != 1 and C.shape[0] != X.shape[0]:
            raise ValueError("The number of conditional vectors should be 1 or equals the number if samples in X.")

        # Transform target into a 2D array.
        Y_t = Y_t.reshape(Y_t.shape[0], -1)

        # Repeat the same label to match the number of input instances.
        if Y_t.shape[0] == 1:
            Y_t = np.tile(Y_t, (X.shape[0], 1))

        # Repeat the same conditional vectors to match the number of input instances.
        if C is not None:
            C = np.tile(C, (X.shape[0], 1))

        # Perform prediction in mini-batches.
        n_minibatch = int(np.ceil(X.shape[0] / batch_size))
        all_results: Dict[str, np.ndarray] = {}

        for i in tqdm(range(n_minibatch)):
            istart, istop = i * batch_size, min((i + 1) * batch_size, X.shape[0])
            results = self._compute_counterfactual(X=X[istart:istop],
                                                   Y_t=Y_t[istart:istop],
                                                   C=C[istart:istop] if (C is not None) else C)

            # Initialize the dict.
            if not all_results:
                all_results = results
                continue

            # Append the new batch off results.
            for key in all_results:
                if all_results[key] is not None:
                    all_results[key] = np.concatenate([all_results[key], results[key]], axis=0)

        return self._build_explanation(**all_results)

    def _compute_counterfactual(self,
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
            Conditional vector. If `None`, it means that no conditioning was used during training (i.e. the
            `conditional_func` returns `None`).

        Returns
        -------
            Dictionary containing the input instances in the original format, input classification labels,
            counterfactual instances in the original format, counterfactual classification labels, target labels,
            conditional vectors.
        """
        # Save original input for later usage.
        X_orig = X

        # Compute models prediction.
        Y_m = self.params["predictor"](X_orig)

        # Check if the prediction task is classification. Please refer to
        # `alibi.explainers.backends.cfrl_base.CounterfactualRLDataset` for a justification.
        if self._is_classification(pred=Y_m):
            Y_m = get_hard_distribution(Y=Y_m, num_classes=Y_m.shape[1])
            Y_t = get_hard_distribution(Y=Y_t, num_classes=Y_m.shape[1])
        else:
            Y_m = Y_m.reshape(-1, 1)
            Y_t = Y_t.reshape(-1, 1)

        # Apply autoencoder preprocessing step.
        X = self.params["encoder_preprocessor"](X_orig)

        # Convert to tensors.
        X = self.backend.to_tensor(X, **self.params)
        Y_m = self.backend.to_tensor(Y_m, **self.params)
        Y_t = self.backend.to_tensor(Y_t, **self.params)
        C = self.backend.to_tensor(C, **self.params)

        # Encode instance.
        Z = self.backend.encode(X, **self.params)

        # Generate counterfactual embedding.
        Z_cf = self.backend.generate_cf(Z, Y_m, Y_t, C, **self.params)

        # Decode counterfactual.
        X_cf = self.backend.decode(Z_cf, **self.params)

        # Convert to numpy for postprocessing
        X_cf = self.backend.to_numpy(X_cf)
        X = self.backend.to_numpy(X)
        C = self.backend.to_numpy(C)

        # Apply postprocessing functions.
        for pp_func in self.params["postprocessing_funcs"]:
            X_cf = pp_func(X_cf, X, C)

        # Apply inverse autoencoder pre-processor.
        X_cf = self.params["decoder_inv_preprocessor"](X_cf)

        # Classify counterfactual instances.
        Y_m_cf = self.params["predictor"](X_cf)

        # Convert tensors to numpy.
        Y_m = self.backend.to_numpy(Y_m)
        Y_t = self.backend.to_numpy(Y_t)

        # If the prediction is a classification task.
        if self._is_classification(pred=Y_m):
            Y_m = np.argmax(Y_m, axis=1)
            Y_t = np.argmax(Y_t, axis=1)
            Y_m_cf = np.argmax(Y_m_cf, axis=1)

        return {
            "X": X_orig,         # input instances
            "Y_m": Y_m,          # input classification labels
            "X_cf": X_cf,        # counterfactual instances
            "Y_m_cf": Y_m_cf,    # counterfactual classification labels
            "Y_t": Y_t,          # target labels
            "C": C               # conditional vectors
        }

    def _build_explanation(self,
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
            Condition vector. If `None`, it means that no conditioning was used during training (i.e. the
            `conditional_func` returns `None`).

        Returns
        -------
            `Explanation` object containing the inputs with the corresponding labels, the counterfactuals with the
            corresponding labels, targets and additional metadata.
        """
        data = deepcopy(DEFAULT_DATA_CFRL)

        # update original input entrance
        data["orig"] = {}
        data["orig"].update({"X": X, "class": Y_m.reshape(-1, 1)})

        # update counterfactual entrance
        data["cf"] = {}
        data["cf"].update({"X": X_cf, "class": Y_m_cf.reshape(-1, 1)})

        # update target and condition
        data["target"] = Y_t.reshape(-1, 1)
        data["condition"] = C
        return Explanation(meta=self.meta, data=data)


class Postprocessing(ABC):
    @abstractmethod
    def __call__(self, X_cf: Any, X: np.ndarray, C: Optional[np.ndarray]) -> Any:
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
           Conditional vector. If `None`, it means that no conditioning was used during training (i.e. the
           `conditional_func` returns `None`).

        Returns
        -------
        X_cf
            Post-processed X_cf.
        """
        pass


class Callback(ABC):
    """ Training callback class. """

    @abstractmethod
    def __call__(self,
                 step: int,
                 update: int,
                 model: CounterfactualRL,
                 sample: Dict[str, np.ndarray],
                 losses: Dict[str, float]) -> None:
        """
        Training callback applied after every training step.

        Parameters
        -----------
        step
            Current experience step.
        update
            Current update step. The ration between the number experience steps and the number of training updates is
            bound to 1.
        model
            CounterfactualRL explainer. All the parameters defined in
            :py:data:`alibi.explainers.cfrl_base.DEFAULT_BASE_PARAMS` can be accessed through 'model.params'.
        sample
            Dictionary of samples used for an update which contains

                - ``'X'``: input instances.

                - ``'Y_m'``: predictor outputs for the input instances.

                - ``'Y_t'``: target outputs.

                - ``'Z'``: input embeddings.

                - ``'Z_cf_tilde'``: noised counterfactual embeddings.

                - ``'X_cf_tilde'``: noised counterfactual instances obtained ofter decoding the noised counterfactual \
                embeddings `Z_cf_tilde` and apply post-processing functions.

                - ``'C'``: conditional vector.

                - ``'R_tilde'``: reward obtained for the noised counterfactual instances.

                - ``'Z_cf'``: counterfactual embeddings.

                - ``'X_cf'``: counterfactual instances obtained after decoding the countefactual embeddings `Z_cf` and \
                apply post-processing functions.
        losses
            Dictionary of losses which contains

                - ``'loss_actor'``: actor network loss.

                - ``'loss_critic'``: critic network loss.

                - ``'sparsity_loss'``: sparsity loss for the \
                :py:class:`alibi.explainers.cfrl_base.CounterfactualRL` class.

                - ``'sparsity_num_loss'``: numerical features sparsity loss for the \
                :py:class:`alibi.explainers.cfrl_tabular.CounterfactualRLTabular` class.

                - ``'sparsity_cat_loss'``: categorical features sparsity loss for the \
                :py:class:`alibi.explainers.cfrl_tabular.CounterfactualRLTabular` class.

                - ``'consistency_loss'``: consistency loss if used.
        """
        pass
