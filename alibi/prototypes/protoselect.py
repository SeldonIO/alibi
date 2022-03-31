import os

import numpy as np
import logging
from tqdm import tqdm
from copy import deepcopy
from typing import Callable, Optional, Dict, List, Union, Any
from alibi.utils.distance import batch_compute_kernel_matrix
from alibi.api.interfaces import Explainer, Explanation, FitMixin
from alibi.api.defaults import DEFAULT_META_PROTOSELECT, DEFAULT_DATA_PROTOSELECT

logger = logging.getLogger(__name__)


class ProtoSelect(Explainer, FitMixin):
    def __init__(self,
                 eps: float,
                 kernel_distance: Callable,
                 lbd: float = None,
                 batch_size: int = int(1e10),
                 preprocess_fn: Callable = None,
                 verbose: bool = False,
                 **kwargs):
        """
        Constructor

        Parameters
        ----------
        kernel_distance
            Kernel to be used. Use `GaussianRBFDistance` or `EuclideanDistance`.
        eps
            Epsilon ball size.
        lbd
            Penalty for each prototype. Encourages a lower number of prototypes to be selected.
        batch_size
            Batch size to be used for kernel matrix computation.
        preprocess_fn
            Preprocessing function for kernel matrix computation.
        verbose
             Whether to display progression bar while computing prototypes points.
        """
        super().__init__(meta=deepcopy(DEFAULT_META_PROTOSELECT))
        self.kernel_distance = kernel_distance
        self.eps = eps
        self.lbd = lbd
        self.batch_size = batch_size
        self.preprocess_fn = preprocess_fn
        self.verbose = verbose

        # get kernel tag
        if hasattr(self.kernel_distance, '__name__'):
            kernel_distance_tag = self.kernel_distance.__name__
        elif hasattr(self.kernel_distance, '__class__'):
            kernel_distance_tag = self.kernel_distance.__class__
        else:
            kernel_distance_tag = 'unknown kernel distance'

        # update metadata
        self.meta['params'].update({
            'kernel_distance': kernel_distance_tag,
            'eps': eps,
            'lbd': lbd,
            'batch_size': batch_size,
            'verbose': verbose
        })

    def fit(self,
            X: np.ndarray,
            X_labels: Optional[np.ndarray] = None,
            Y: Optional[np.ndarray] = None) -> 'ProtoSelect':
        """
        Fit the explainer by setting the reference dataset.

        Parameters
        ---------
        X
            Reference dataset to be summarized.
        X_labels
            Labels of the reference dataset.
        Y
            Dataset to choose the prototypes from. If ``None``, the prototypes will be selected from the reference
            dataset `X`.

        Returns
        -------
        self
            Reference to itself.
        """
        self.X = X
        # if the `X_labels` are not provided, then consider that all elements belong to the same class. This means
        # that loss term which tries to avoid including in an epsilon ball elements belonging to other classes
        # will always be 0. Still the first term of the loss tries to cover as many examples as possible with
        # minimal overlap between the epsilon balls corresponding to the other prototypes.
        self.X_labels = X_labels.astype(np.int32) if (X_labels is not None) else np.zeros((len(X), ), dtype=np.int32)
        # if the set of prototypes is not provided, then find the prototypes belonging to the reference dataset.
        self.Y = Y if (Y is not None) else self.X
        # initialize penalty for adding a prototype
        if self.lbd is None:
            self.lbd = 1 / len(self.X)
            self.meta['params'].update({'lbd': self.lbd})

        self.max_label = np.max(self.X_labels)
        self.kmatrix_yx = batch_compute_kernel_matrix(x=self.Y,
                                                      y=self.X,
                                                      kernel=self.kernel_distance,
                                                      batch_size=self.batch_size,
                                                      preprocess_fn=self.preprocess_fn)

        return self

    def explain(self, num_prototypes: int = 1) -> Explanation:
        """
        Searches for the requested number of prototypes. Note that the algorithm can return a lower number of
        prototypes than the requested one. To increase the number of prototypes, reduce the epsilon-ball radius
        `eps` and the penalty `lbd` for adding a prototype.

        Parameters
        ----------
        num_prototypes
            Number of prototypes to be selected.

        Returns
        -------
        An `Explanation` object containing the prototypes, prototypes indices and protoypes labels with additional \
        metadata as attributes
        """
        if num_prototypes > len(self.Y):
            num_prototypes = len(self.Y)
            logger.warning('The number of prototypes requested is larger than the number of elements from '
                           f'the prototypes selection set. Automatically setting `num_prototypes={num_prototypes}`.')

        # dictionary of prototypes indices for each class
        protos = {l: [] for l in range(self.max_label + 1)}
        # set of available prototypes indices. Note that initially we start with the entire set of Y,
        # but as the algorithm progresses, we remove the indices of the prototypes that we already selected.
        available_indices = set(range(len(self.Y)))
        # matrix of size `[NY, NX]`, where `NY = len(Y)` and `NX = len(X)`
        # represents a mask which indicates for each element `y` in `Y` what are the elements of `X` that are in an
        # epsilon ball centered in `y`.
        B = (self.kmatrix_yx <= self.eps).astype(np.int32)
        # matrix of size `[L, NX]`, where `L` is the number of labels
        # each row `l` indicates the elements from `X` that are covered by prototypes belonging to class `l`
        B_P = np.zeros((self.max_label + 1, len(self.X)), dtype=np.int32)
        # matrix of size `[L, NX]`. Each row `l` indicates which elements form `X` are labeled as `l`
        Xl = np.concatenate([(self.X_labels == l).reshape(1, -1)
                             for l in range(self.max_label + 1)], axis=0).astype(np.int32)

        # vectorized implementation of the prototypes scores.
        # See paper (pag 8): https://arxiv.org/pdf/1202.5933.pdf for more details
        B_diff = B[:, np.newaxis, :] - B_P[np.newaxis, :, :]  # [NY, 1, NX] - [1, L, NX] -> [NY, L, NX]
        # [NY, L, NX] + [1, L, NX] -> [NY, L, NX]
        delta_xi_all = B_diff + Xl[np.newaxis, ...] >= 2
        # [NY, L]. For every row `y` and every column `l`, we compute how many new instances belonging to class
        # `l` will be covered if we add the prototype `y`.
        delta_xi_summed = np.sum(delta_xi_all, axis=-1)
        # [NY, 1, NX] +  [1, L, NX] -> [NY, L, NX]
        delta_nu_all = B[:, np.newaxis, :] + (1 - Xl[np.newaxis, ...]) >= 2
        # [NY, L]. For every row `y` and every column `l`, we compute how many new instances belonging to all the
        # other classes different then `l` will be covered if we add the prototype `y`.
        delta_nu_summed = np.sum(delta_nu_all, axis=-1)
        # compute the tradeoff score - each prototype tries to cover as many new elements as possible
        # belonging to the same class, while trying to avoid covering elements belonging to another class
        scores_all = delta_xi_summed - delta_nu_summed - self.lbd

        # add progressing bar if `verbose=True`
        generator = range(num_prototypes)
        if self.verbose:
            generator = tqdm(generator)

        for _ in generator:
            j = np.array(list(available_indices))
            scores = scores_all[j]

            # stopping criterion. The number of the returned prototypes might be lower than
            # the number of requested prototypes
            if np.all(scores < 0):
                break

            # find the index `i` of the best prototype and the class `l` that it covers
            row, col = np.unravel_index(np.argmax(scores), scores.shape)
            i, l = j[row], col

            # update the score
            covered = np.sum(delta_xi_all[:, l, B[i].astype(bool)], axis=-1)
            delta_xi_all[:, l, B[i].astype(bool)] = 0
            delta_xi_summed[:, l] -= covered
            scores_all[:, l] -= covered

            # add prototype to the corresponding list according to the class label `l` that it covers
            # and remove the index `i` from list of available indices
            protos[l].append(i)
            available_indices.remove(i)

        return self._build_explanation(protos)

    def _build_explanation(self, protos: Dict[int, List[int]]) -> Explanation:
        """
        Helper method to build `Explanation` object.
        """
        data = deepcopy(DEFAULT_DATA_PROTOSELECT)
        data['prototypes_indices'] = np.concatenate(list(protos.values())).astype(np.int32)
        data['prototypes_labels'] = np.concatenate([[l] * len(protos[l]) for l in protos]).astype(np.int32)
        data['prototypes'] = self.Y[data['prototypes_indices']]
        return Explanation(meta=self.meta, data=data)

    def save(self, path: Union[str, os.PathLike]) -> None:
        super().save(path)

    @classmethod
    def load(cls, path: Union[str, os.PathLike], predictor: Optional[Any] = None) -> "Explainer":
        return super().load(path, predictor)
