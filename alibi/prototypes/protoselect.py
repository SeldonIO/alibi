import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from skimage.transform import resize
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from alibi.api.defaults import (DEFAULT_DATA_PROTOSELECT,
                                DEFAULT_META_PROTOSELECT)
from alibi.api.interfaces import Explanation, FitMixin, Summariser
from alibi.utils.distance import batch_compute_kernel_matrix
from alibi.utils.kernel import EuclideanDistance

logger = logging.getLogger(__name__)


class ProtoSelect(Summariser, FitMixin):
    def __init__(self,
                 kernel_distance: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 eps: float,
                 lambda_penalty: Optional[float] = None,
                 batch_size: int = int(1e10),
                 preprocess_fn: Optional[Callable[[Union[list, np.ndarray]], np.ndarray]] = None,
                 verbose: bool = False):
        """
        Prototype selection for dataset distillation and interpretable classification proposed by
        Bien and Tibshirani (2012): https://arxiv.org/abs/1202.5933

        Parameters
        ----------
        kernel_distance
            Kernel distance to be used. Expected to support computation in batches. Given an input `x` of
            size `Nx x f1 x f2 x ...` and an input `y` of size `Ny x f1 x f2 x ...`, the kernel distance
            should return a kernel matrix of size `Nx x Ny`.
        eps
            Epsilon ball size.
        lambda_penalty
            Penalty for each prototype. Encourages a lower number of prototypes to be selected. Corresponds to
            :math:`\\lambda` in the paper notation. If not specified, the default value is set to `1 / N` where
            `N` is the size of the dataset to choose the prototype instances from, passed to the
            :py:meth:`alibi.prototypes.protoselect.ProtoSelect.fit` method.
        batch_size
            Batch size to be used for kernel matrix computation.
        preprocess_fn
            Preprocessing function used for kernel matrix computation. The preprocessing function takes the input as
            a `list` or a `numpy` array and transforms it into a `numpy` array which is then fed to the
            `kernel_distance` function. The use of `preprocess_fn` allows the method to be applied to any data modality.
        verbose
             Whether to display progression bar while computing prototype points.
        """
        super().__init__(meta=deepcopy(DEFAULT_META_PROTOSELECT))
        self.kernel_distance = kernel_distance
        self.eps = eps
        self.lambda_penalty = lambda_penalty
        self.batch_size = batch_size
        self.preprocess_fn = preprocess_fn
        self.verbose = verbose

        # get kernel tag
        if hasattr(self.kernel_distance, '__name__'):
            kernel_distance_tag = self.kernel_distance.__name__
        elif hasattr(self.kernel_distance, '__class__'):
            kernel_distance_tag = self.kernel_distance.__class__.__name__
        else:
            kernel_distance_tag = 'unknown kernel distance'

        # update metadata
        self.meta['params'].update({
            'kernel_distance': kernel_distance_tag,
            'eps': eps,
            'lambda_penalty': lambda_penalty,
            'batch_size': batch_size,
            'verbose': verbose
        })

    def fit(self,  # type: ignore[override]
            X: Union[list, np.ndarray],
            y: Optional[np.ndarray] = None,
            Z: Optional[Union[list, np.ndarray]] = None) -> 'ProtoSelect':
        """
        Fit the summariser. This step forms the kernel matrix in memory which has a shape of `NX x NX`,
        where `NX` is  the number of instances in `X`, if the optional dataset `Z` is not provided. Otherwise, if
        the optional dataset `Z` is provided, the kernel matrix has a shape of `NZ x NX`, where `NZ` is the
        number of instances in `Z`.

        Parameters
        ---------
        X
            Dataset to be summarised.
        y
            Labels of the dataset `X` to be summarised. The labels are expected to be represented as integers
            `[0, 1, ..., L-1]`, where `L` is the number of classes in the dataset `X`.
        Z
            Optional dataset to choose the prototypes from. If ``Z=None``, the prototypes will be selected from the
            dataset `X`. Otherwise, if `Z` is provided, the dataset to be summarised is still `X`, but
            it is summarised by prototypes belonging to the dataset `Z`.

        Returns
        -------
        self
            Reference to itself.
        """
        if y is not None:
            y = y.flatten()
            if len(X) != len(y):
                raise ValueError('The number of data instances does not match the number of labels. '
                                 f'Got len(X)={len(X)} and len(y)={len(y)}.')

        self.X = X
        # if the y is not provided, then consider that all elements belong to the same class. This means
        # that loss term which tries to avoid including in an epsilon ball elements belonging to other classes
        # will always be 0. Still the first term of the loss tries to cover as many examples as possible with
        # minimal overlap between the epsilon balls corresponding to the other prototypes.
        self.y = y.astype(np.int32) if (y is not None) else np.zeros((len(X),), dtype=np.int32)

        # redefine the labels, so they are in the interval [0, len(np.unique(y)) - 1].
        # For example, if the labels provided were [40, 51], internally, we relabel them as [0, 1].
        # This approach can reduce computation and memory allocation, as without the intermediate mapping we would
        # have to allocate memory corresponding to 52 labels, [0, ..., 51], for some internal matrices.
        self.label_mapping = {l: i for i, l in enumerate(np.unique(self.y))}
        self.label_inv_mapping = {v: k for k, v in self.label_mapping.items()}
        idx = np.nonzero(np.asarray(list(self.label_mapping.keys())) == self.y[:, None])[1]
        self.y = np.asarray(list(self.label_mapping.values()))[idx]

        # if the set of prototypes is not provided, then find the prototypes belonging to the X dataset.
        self.Z = Z if (Z is not None) else self.X
        # initialize penalty for adding a prototype
        if self.lambda_penalty is None:
            self.lambda_penalty = 1. / len(self.Z)
            self.meta['params'].update({'lambda_penalty': self.lambda_penalty})

        self.kmatrix = batch_compute_kernel_matrix(x=self.Z,
                                                   y=self.X,
                                                   kernel=self.kernel_distance,
                                                   batch_size=self.batch_size,
                                                   preprocess_fn=self.preprocess_fn)

        return self

    def summarise(self, num_prototypes: int = 1) -> Explanation:
        """
        Searches for the requested number of prototypes. Note that the algorithm can return a lower number of
        prototypes than the requested one. To increase the number of prototypes, reduce the epsilon-ball radius
        (`eps`), and the penalty for adding a prototype (`lambda_penalty`).

        Parameters
        ----------
        num_prototypes
            Maximum number of prototypes to be selected.

        Returns
        -------
        An `Explanation` object containing the prototypes, prototype indices and prototype labels with additional \
        metadata as attributes.
        """
        if num_prototypes > len(self.Z):
            num_prototypes = len(self.Z)
            logger.warning('The number of prototypes requested is larger than the number of elements from '
                           f'the prototypes selection set. Automatically setting `num_prototypes={num_prototypes}`.')

        # dictionary of prototypes indices for each class
        protos: Dict[int, List[int]] = {l: [] for l in range(len(self.label_mapping))}  # noqa: E741
        # set of available prototypes indices. Note that initially we start with the entire set of Z,
        # but as the algorithm progresses, we remove the indices of the prototypes that we already selected.
        available_indices = set(range(len(self.Z)))
        # matrix of size [NZ, NX], where NZ = len(Z) and NX = len(X)
        # represents a mask which indicates for each element z in Z what are the elements of X that are in an
        # epsilon ball centered in z.
        B = (self.kmatrix <= self.eps).astype(np.int32)
        # matrix of size [L, NX], where L is the number of labels
        # each row l indicates the elements from X that are covered by prototypes belonging to class l.
        B_P = np.zeros((len(self.label_mapping), len(self.X)), dtype=np.int32)
        # matrix of size [L, NX]. Each row l indicates which elements form X are labeled as l.
        Xl = np.concatenate([(self.y == l).reshape(1, -1)
                             for l in range(len(self.label_mapping))], axis=0).astype(np.int32)  # noqa: E741

        # vectorized implementation of the prototypes scores.
        # See paper (pag 8): https://arxiv.org/pdf/1202.5933.pdf for more details
        B_diff = B[:, np.newaxis, :] - B_P[np.newaxis, :, :]  # [NZ, 1, NX] - [1, L, NX] -> [NZ, L, NX]
        # [NZ, L, NX] + [1, L, NX] -> [NZ, L, NX]
        delta_xi_all = B_diff + Xl[np.newaxis, ...] >= 2
        # [NZ, L]. For every row z and every column l, we compute how many new instances belonging to class
        # l will be covered if we add the prototype z.
        delta_xi_summed = np.sum(delta_xi_all, axis=-1)
        # [NZ, 1, NX] +  [1, L, NX] -> [NZ, L, NX]
        delta_nu_all = B[:, np.newaxis, :] + (1 - Xl[np.newaxis, ...]) >= 2
        # [NZ, L]. For every row z and every column l, we compute how many new instances belonging to all the
        # other classes different from l will be covered if we add the prototype z.
        delta_nu_summed = np.sum(delta_nu_all, axis=-1)
        # compute the tradeoff score - each prototype tries to cover as many new elements as possible
        # belonging to the same class, while trying to avoid covering elements belonging to another class
        scores_all = delta_xi_summed - delta_nu_summed - self.lambda_penalty

        for _ in tqdm(range(num_prototypes), disable=(not self.verbose)):
            j = np.array(list(available_indices)).astype(np.int32)
            scores = scores_all[j]

            # stopping criterion. The number of the returned prototypes might be lower than
            # the number of requested prototypes.
            if np.all(scores < 0):
                break

            # find the index i of the best prototype and the class l that it covers.
            row, col = np.unravel_index(np.argmax(scores), scores.shape)
            i, l = j[row.item()], col.item()  # noqa: E741

            # update the score.
            covered = np.sum(delta_xi_all[:, l, B[i].astype(bool)], axis=-1)
            delta_xi_all[:, l, B[i].astype(bool)] = 0
            delta_xi_summed[:, l] -= covered
            scores_all[:, l] -= covered

            # add prototype to the corresponding list according to the class label l that it covers
            # and remove the index i from list of available indices.
            protos[l].append(i)
            available_indices.remove(i)

        return self._build_summary(protos)

    def _build_summary(self, protos: Dict[int, List[int]]) -> Explanation:
        """
        Helper method to build the summary as an `Explanation` object.
        """
        data = deepcopy(DEFAULT_DATA_PROTOSELECT)
        data['prototype_indices'] = np.concatenate(list(protos.values())).astype(np.int32)
        data['prototype_labels'] = np.concatenate([[self.label_inv_mapping[l]] * len(protos[l])
                                                   for l in protos]).astype(np.int32)  # noqa: E741
        data['prototypes'] = self.Z[data['prototype_indices']]
        return Explanation(meta=self.meta, data=data)


def _helper_protoselect_euclidean_1knn(summariser: ProtoSelect,
                                       num_prototypes: int,
                                       eps: float,
                                       knn_kw: dict) -> Optional[KNeighborsClassifier]:
    """
    Helper function to fit a 1-KNN classifier on the prototypes returned by the `summariser`.
    Sets the epsilon radius to be used.

    Parameters
    ----------
    summariser
        Fitted `ProtoSelect` summariser.
    num_prototypes
        Number of requested prototypes.
    eps
        Epsilon radius to be set and used for the computation of prototypes.
    knn_kw
        Keyword arguments passed to `sklearn.neighbors.KNeighborsClassifier`. See parameters description:
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

    Returns
    -------
    Fitted 1-KNN classifier with Euclidean distance metric.
    """
    # update summariser eps and get the summary
    summariser.eps = eps
    summary = summariser.summarise(num_prototypes=num_prototypes)

    # train 1-knn classifier
    X_protos, y_protos = summary.data['prototypes'], summary.data['prototype_labels']
    if len(X_protos) == 0:
        return None

    # note that the knn_kw are updated in `cv_protoselect_euclidean` to define a 1-KNN with Euclidean distance
    knn = KNeighborsClassifier(**knn_kw)
    return knn.fit(X=X_protos, y=y_protos)


def _get_splits(trainset: Tuple[np.ndarray, np.ndarray],
                valset: Tuple[Optional[np.ndarray], Optional[np.ndarray]],
                kfold_kw: dict) -> Tuple[
                    Tuple[np.ndarray, np.ndarray],
                    Tuple[np.ndarray, np.ndarray],
                    List[Tuple[np.ndarray, np.ndarray]]
                ]:
    """
    Helper function to obtain appropriate train-validation splits.

    If the validation dataset is not provided, then the method returns the appropriate datasets and indices
    to perform k-fold validation. Otherwise, if the validation dataset is provided, then use it instead of performing
    k-fold validation.

    Parameters
    ----------
    trainset
        Tuple `(X_train, y_train)` consisting of the training data instances with the corresponding labels.
    valset
        Optional tuple, `(X_val, y_val)`, consisting of validation data instances with the corresponding labels.
    kfold_kw
        Keyword arguments passed to `sklearn.model_selection.KFold`. See parameters description:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

    Returns
    -------
    Tuple consisting of training dataset, validation dataset (can overlap with training if validation is not
    provided), and a list of splits containing indices from the training and validation datasets.
    """
    X_train, y_train = trainset
    X_val, y_val = valset

    if X_val is None:
        kfold = KFold(**kfold_kw)
        splits = kfold.split(X=X_train, y=y_train)
        return trainset, trainset, list(splits)

    splits = [(np.arange(len(X_train)), np.arange(len(X_val)))]
    return trainset, valset, splits  # type: ignore


def cv_protoselect_euclidean(trainset: Tuple[np.ndarray, np.ndarray],
                             protoset: Optional[Tuple[np.ndarray, ]] = None,
                             valset: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                             num_prototypes: int = 1,
                             eps_grid: Optional[np.ndarray] = None,
                             quantiles: Optional[Tuple[float, float]] = None,
                             grid_size: int = 25,
                             n_splits: int = 2,
                             batch_size: int = int(1e10),
                             preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                             protoselect_kw: Optional[dict] = None,
                             knn_kw: Optional[dict] = None,
                             kfold_kw: Optional[dict] = None) -> dict:
    """
    Cross-validation parameter selection for `ProtoSelect` with Euclidean distance. The method computes
    the best epsilon radius.

    Parameters
    ----------
    trainset
        Tuple, `(X_train, y_train)`, consisting of the training data instances with the corresponding labels.
    protoset
        Tuple, `(Z, )`, consisting of the dataset to choose the prototypes from. If `Z` is not provided
        (i.e., ``protoset=None``), the prototypes will be selected from the training dataset `X`. Otherwise, if `Z`
        is provided, the dataset to be summarised is still `X`, but it is summarised by prototypes belonging to
        the dataset `Z`. Note that the argument is passed as a tuple with a single element for consistency reasons.
    valset
        Optional tuple `(X_val, y_val)` consisting of validation data instances with the corresponding
        validation labels. 1-KNN classifier is evaluated on the validation dataset to obtain the best epsilon radius.
        In case ``valset=None``, then `n-splits` cross-validation is performed on the `trainset`.
    num_prototypes
        The number of prototypes to be selected.
    eps_grid
        Optional grid of values to select the epsilon radius from. If not specified, the search grid is
        automatically proposed based on the inter-distances between `X` and `Z`. The distances are filtered
        by considering only values in between the `quantiles` values. The minimum and maximum distance values are
        used to define the range of values to search the epsilon radius. The interval is discretized in `grid_size`
        equidistant bins.
    quantiles
        Quantiles, `(q_min, q_max)`, to be used to filter the range of values of the epsilon radius. The expected
        quantile values are in `[0, 1]` and clipped to `[0, 1]` if outside the range. See `eps_grid` for usage.
        If not specified, no filtering is applied. Only used if ``eps_grid=None``.
    grid_size
        The number of equidistant bins to be used to discretize the `eps_grid` automatically proposed interval.
        Only used if ``eps_grid=None``.
    batch_size
        Batch size to be used for kernel matrix computation.
    preprocess_fn
        Preprocessing function to be applied to the data instance before applying the kernel.
    protoselect_kw
        Keyword arguments passed to :py:meth:`alibi.prototypes.protoselect.ProtoSelect.__init__`.
    knn_kw
        Keyword arguments passed to `sklearn.neighbors.KNeighborsClassifier`. The `n_neighbors` will be
        set automatically to 1 and the `metric` will be set to ``'euclidean``. See parameters description:
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    kfold_kw
        Keyword arguments passed to `sklearn.model_selection.KFold`. See parameters description:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html

    Returns
    -------
    Dictionary containing
     - ``'best_eps'``: ``float`` - the best epsilon radius according to the accuracy of a 1-KNN classifier.
     - ``'meta'``: ``dict`` - dictionary containing argument and data gather throughout cross-validation.
    """
    if protoselect_kw is None:
        protoselect_kw = {}

    if kfold_kw is None:
        kfold_kw = {}

    if knn_kw is None:
        knn_kw = {}
    # ensure that we are training a 1-KNN classifier with Euclidean distance metric
    knn_kw.update({'n_neighbors': 1, 'metric': 'euclidean'})

    X_train, y_train = trainset
    Z = protoset[0] if (protoset is not None) else X_train
    X_val, y_val = valset if (valset is not None) else (None, None)

    if preprocess_fn is not None:
        X_train = _batch_preprocessing(X=X_train, preprocess_fn=preprocess_fn, batch_size=batch_size)
        Z = _batch_preprocessing(X=Z, preprocess_fn=preprocess_fn, batch_size=batch_size)
        if X_val is not None:
            X_val = _batch_preprocessing(X_val, preprocess_fn=preprocess_fn, batch_size=batch_size)

    # propose eps_grid if not specified
    if eps_grid is None:
        dist = batch_compute_kernel_matrix(x=X_train, y=Z, kernel=EuclideanDistance()).reshape(-1)
        if quantiles is not None:
            if quantiles[0] > quantiles[1]:
                raise ValueError('The quantile lower-bound is greater then the quantile upper-bound.')
            quantiles = np.clip(quantiles, a_min=0, a_max=1)  # type: ignore[assignment]
            min_dist, max_dist = np.quantile(a=dist, q=np.array(quantiles))
        else:
            min_dist, max_dist = np.min(dist), np.max(dist)
        # define list of values for eps
        eps_grid = np.linspace(min_dist, max_dist, num=grid_size)

    (X_train, y_train), (X_val, y_val), splits = _get_splits(trainset=(X_train, y_train),
                                                             valset=(X_val, y_val),
                                                             kfold_kw=kfold_kw)
    scores = np.zeros((len(eps_grid), len(splits)))

    for i, (train_index, val_index) in enumerate(splits):
        X_train_i, y_train_i = X_train[train_index], y_train[train_index]
        X_val_i, y_val_i = X_val[val_index], y_val[val_index]

        # define and fit the summariser here, so we don't repeat the kernel matrix computation in the next for loop
        summariser = ProtoSelect(kernel_distance=EuclideanDistance(), eps=0, **protoselect_kw)
        summariser = summariser.fit(X=X_train_i, y=y_train_i, Z=Z)

        for j in range(len(eps_grid)):
            knn = _helper_protoselect_euclidean_1knn(summariser=summariser,
                                                     num_prototypes=num_prototypes,
                                                     eps=eps_grid[j],
                                                     knn_kw=knn_kw)
            if knn is None:
                continue
            scores[j][i] = knn.score(X_val_i, y_val_i)

    return {
        'best_eps': eps_grid[np.argmax(np.mean(scores, axis=-1))],
        'meta': {
            'num_prototypes': num_prototypes,
            'eps_grid': eps_grid,
            'quantiles': quantiles,
            'grid_size': grid_size,
            'n_splits': n_splits,
            'batch_size': batch_size,
            'scores': scores,
        }
    }


def _batch_preprocessing(X: np.ndarray,
                         preprocess_fn: Callable[[np.ndarray], np.ndarray],
                         batch_size: int = 32) -> np.ndarray:
    """
    Preprocess a dataset `X` in batches by applying the preprocessor function.

    Parameters
    ----------
    X
        Dataset to be preprocessed.
    preprocess_fn
        Preprocessor function.
    batch_size
        Batch size to be used for each call to `preprocess_fn`.

    Returns
    -------
    Preprocessed dataset.
    """
    X_ft = []
    num_iter = int(np.ceil(len(X) / batch_size))

    for i in range(num_iter):
        istart, iend = batch_size * i, min(batch_size * (i + 1), len(X))
        X_ft.append(preprocess_fn(X[istart:iend]))

    return np.concatenate(X_ft, axis=0)


def _imscatterplot(x: np.ndarray,
                   y: np.ndarray,
                   images: np.ndarray,
                   ax: Optional[plt.Axes] = None,
                   fig_kw: Optional[dict] = None,
                   image_size: Tuple[int, int] = (28, 28),
                   zoom: Optional[np.ndarray] = None,
                   zoom_lb: float = 1.0,
                   zoom_ub=2.0,
                   sort_by_zoom: bool = True) -> plt.Axes:
    """
    2D image scatter plot.

    Parameters
    ----------
    x
        Images x-coordinates.
    y
        Images y-coordinates.
    images
        Array of images to be placed at coordinates `(x, y)`.
    ax
        A `matplotlib` axes object to plot on.
    fig_kw
        Keyword arguments passed to the `fig.set` function.
    image_size
        Size of the generated output image as `(rows, cols)`.
    zoom
        Images zoom to be used.
    zoom_lb
        Zoom lower bound. The zoom values will be scaled linearly between `[zoom_lb, zoom_up]`.
    zoom_ub
        Zoom upper bound. The zoom values will be scaled linearly between `[zoom_lb, zoom_up]`.
    """
    if fig_kw is None:
        fig_kw = {}

    if zoom is None:
        zoom = np.ones(len(images))
    else:
        zoom_min, zoom_max = np.min(zoom), np.max(zoom)
        zoom = (zoom - zoom_min) / (zoom_max - zoom_min) * (zoom_ub - zoom_lb) + zoom_lb

        if sort_by_zoom:
            idx = np.argsort(zoom)[::-1]  # type: ignore
            zoom = zoom[idx]  # type: ignore
            x, y, images = x[idx], y[idx], images[idx]

    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        fig = ax.figure

    resized_imgs = [resize(images[i], image_size) for i in range(len(images))]
    imgs = [OffsetImage(img, zoom=zoom[i], cmap='gray') for i, img in enumerate(resized_imgs)]  # type: ignore
    artists = []

    for i in range(len(imgs)):
        x0, y0, im = x[i], y[i], imgs[i]
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    fig.set(**fig_kw)
    return ax


def compute_prototype_importances(summary: 'Explanation',
                                  trainset: Tuple[np.ndarray, np.ndarray],
                                  preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                                  knn_kw: Optional[dict] = None) -> Dict[str, Optional[np.ndarray]]:

    """
    Computes the importance of each prototype. The importance of a prototype is the number of assigned
    training instances correctly classified according to the 1-KNN classifier
    (Bien and Tibshirani (2012): https://arxiv.org/abs/1202.5933).

    Parameters
    ----------
    summary
        An `Explanation` object produced by a call to the
        :py:meth:`alibi.prototypes.protoselect.ProtoSelect.summarise` method.
    trainset
        Tuple, `(X_train, y_train)`, consisting of the training data instances with the corresponding labels.
    preprocess_fn
        Optional preprocessor function. If ``preprocess_fn=None``, no preprocessing is applied.
    knn_kw
        Keyword arguments passed to `sklearn.neighbors.KNeighborsClassifier`. The `n_neighbors` will be
        set automatically to 1, but the `metric` has to be specified according to the kernel distance used.
        If the `metric` is not specified, it will be set by default to ``'euclidean'``.
        See parameters description:
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

    Returns
    -------
    A dictionary containing:

     - ``'prototype_indices'`` - an array of the prototype indices.

     - ``'prototype_importances'`` - an array of prototype importances.

     - ``'X_protos'`` - an array of raw prototypes.

     - ``'X_protos_ft'`` - an optional array of preprocessed prototypes. If the ``preprocess_fn=None``, \
     no preprocessing is applied and ``None`` is returned instead.
    """
    if knn_kw is None:
        knn_kw = {}

    if knn_kw.get('metric') is None:
        knn_kw.update({'metric': 'euclidean'})
        logger.warning("KNN metric was not specified. Automatically setting `metric='euclidean'`.")

    X_train, y_train = trainset
    X_protos = summary.data['prototypes']
    y_protos = summary.data['prototype_labels']

    # preprocess the dataset
    X_train_ft = _batch_preprocessing(X=X_train, preprocess_fn=preprocess_fn) \
        if (preprocess_fn is not None) else X_train
    X_protos_ft = _batch_preprocessing(X=X_protos, preprocess_fn=preprocess_fn) \
        if (preprocess_fn is not None) else X_protos

    # train knn classifier
    knn = KNeighborsClassifier(n_neighbors=1, **knn_kw)
    knn = knn.fit(X=X_protos_ft, y=y_protos)

    # get neighbors indices for each training instance
    neigh_idx = knn.kneighbors(X=X_train_ft, n_neighbors=1, return_distance=False).reshape(-1)

    # compute how many correct labeled instances each prototype covers
    idx, counts = np.unique(neigh_idx[y_protos[neigh_idx] == y_train], return_counts=True)
    return {
        'prototype_indices': idx,
        'prototype_importances': counts,
        'X_protos': X_protos[idx],
        'X_protos_ft': None if (preprocess_fn is None) else X_protos_ft[idx]
    }


def visualize_image_prototypes(summary: 'Explanation',
                               trainset: Tuple[np.ndarray, np.ndarray],
                               reducer: Callable[[np.ndarray], np.ndarray],
                               preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                               knn_kw: Optional[dict] = None,
                               ax: Optional[plt.Axes] = None,
                               fig_kw: Optional[dict] = None,
                               image_size: Tuple[int, int] = (28, 28),
                               zoom_lb: float = 1.0,
                               zoom_ub: float = 3.0) -> plt.Axes:
    """
    Plot the images of the prototypes at the location given by the `reducer` representation.
    The size of each prototype is proportional to the logarithm of the number of assigned training instances correctly
    classified according to the 1-KNN classifier (Bien and Tibshirani (2012): https://arxiv.org/abs/1202.5933).

    Parameters
    ----------
    summary
        An `Explanation` object produced by a call to the
        :py:meth:`alibi.prototypes.protoselect.ProtoSelect.summarise` method.
    trainset
        Tuple, `(X_train, y_train)`, consisting of the training data instances with the corresponding labels.
    reducer
        2D reducer. Reduces the input feature representation to 2D. Note that the reducer operates directly on the
        input instances if ``preprocess_fn=None``. If the `preprocess_fn` is specified, the reducer will be called
        on the feature representation obtained after passing the input instances through the `preprocess_fn`.
    preprocess_fn
        Optional preprocessor function. If ``preprocess_fn=None``, no preprocessing is applied.
    knn_kw
        Keyword arguments passed to `sklearn.neighbors.KNeighborsClassifier`. The `n_neighbors` will be
        set automatically to 1, but the `metric` has to be specified according to the kernel distance used.
        If the `metric` is not specified, it will be set by default to ``'euclidean'``.
        See parameters description:
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    ax
        A `matplotlib` axes object to plot on.
    fig_kw
        Keyword arguments passed to the `fig.set` function.
    image_size
        Shape to which the prototype images will be resized. A zoom of 1 will display the image having the shape
        `image_size`.
    zoom_lb
        Zoom lower bound. The zoom will be scaled linearly between `[zoom_lb, zoom_ub]`.
    zoom_ub
        Zoom upper bound. The zoom will be scaled linearly between `[zoom_lb, zoom_ub]`.
    """
    # compute how many correct labeled instances each prototype covers
    protos_importance = compute_prototype_importances(summary=summary,
                                                      trainset=trainset,
                                                      preprocess_fn=preprocess_fn,
                                                      knn_kw=knn_kw)

    # unpack values
    counts = protos_importance['prototype_importances']
    X_protos = protos_importance['X_protos']
    X_protos_ft = protos_importance['X_protos_ft'] if (protos_importance['X_protos_ft'] is not None) else X_protos

    # compute image zoom
    zoom = np.log(counts)  # type: ignore[arg-type]

    # compute 2D embedding
    protos_2d = reducer(X_protos_ft)  # type: ignore[arg-type]
    x, y = protos_2d[:, 0], protos_2d[:, 1]

    # plot images
    return _imscatterplot(x=x, y=y,
                          images=X_protos,  # type: ignore[arg-type]
                          ax=ax, fig_kw=fig_kw,
                          image_size=image_size,
                          zoom=zoom, zoom_lb=zoom_lb, zoom_ub=zoom_ub)
