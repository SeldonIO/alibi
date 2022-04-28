import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from tqdm import tqdm
from copy import deepcopy
from typing import Callable, Optional, Dict, List, Union, Tuple
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from skimage.transform import resize

from alibi.utils.distance import batch_compute_kernel_matrix
from alibi.api.interfaces import Explainer, Explanation, FitMixin
from alibi.api.defaults import DEFAULT_META_PROTOSELECT, DEFAULT_DATA_PROTOSELECT
from alibi.utils.kernel import EuclideanDistance

logger = logging.getLogger(__name__)


class ProtoSelect(Explainer, FitMixin):
    def __init__(self,
                 kernel_distance: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 eps: float,
                 lambda_penalty: Optional[float] = None,
                 batch_size: int = int(1e10),
                 preprocess_fn: Optional[Callable[[Union[list, np.ndarray]], np.ndarray]] = None,
                 verbose: bool = False):
        """
        Prototype selection for dataset distillation and interpretable classification.

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
            :math:`\\lambda` in the paper's notation. If not specified, the default value is set to `1 / N` where
            `N` represents the number of reference instances passed to the
            :py:meth:`alibi.prototypes.protoselect.ProtoSelect.fit` method.
        batch_size
            Batch size to be used for kernel matrix computation.
        preprocess_fn
            Preprocessing function used for kernel matrix computation. The preprocessing function takes the input in
            a raw format or as a `numpy` array and transforms it into a `numpy` array which is then fed to the
            `kernel_distance` function. The use of `preprocess_fn` allows the method to be applied to any data modality.
        verbose
             Whether to display progression bar while computing prototypes points.
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
            X_ref: Union[list, np.ndarray],
            Y_ref: Optional[np.ndarray] = None,
            X: Optional[Union[list, np.ndarray]] = None) -> 'ProtoSelect':
        """
        Fit the explainer by setting the reference dataset. This step form the kernel matrix in memory
        which has a shape of `Nx x Ny`, where `Nx` are the number of instances in `X` and `Ny` are the
        number of instances in `Y`.

        Parameters
        ---------
        X_ref
            Reference dataset to be summarized.
        Y_ref
            Labels of the reference dataset. The labels are expected to be represented as integers `[0, 1, ..., L-1]`,
            where `L` is the number of classes in the reference dataset.
        X
            Dataset to choose the prototypes from. If ``None``, the prototypes will be selected from the reference
            dataset `X_ref`.

        Returns
        -------
        self
            Reference to itself.
        """
        self.X_ref = X_ref
        # if the `X_labels` are not provided, then consider that all elements belong to the same class. This means
        # that loss term which tries to avoid including in an epsilon ball elements belonging to other classes
        # will always be 0. Still the first term of the loss tries to cover as many examples as possible with
        # minimal overlap between the epsilon balls corresponding to the other prototypes.
        self.Y_ref = Y_ref.astype(np.int32) if (Y_ref is not None) else np.zeros((len(X_ref),), dtype=np.int32)
        # if the set of prototypes is not provided, then find the prototypes belonging to the reference dataset.
        self.X = X if (X is not None) else self.X_ref
        # initialize penalty for adding a prototype
        if self.lambda_penalty is None:
            self.lambda_penalty = 1 / len(self.X_ref)
            self.meta['params'].update({'lambda_penalty': self.lambda_penalty})

        self.max_label = np.max(self.Y_ref)
        self.kmatrix = batch_compute_kernel_matrix(x=self.X,
                                                   y=self.X_ref,
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
            Number of maximum prototypes to be selected.

        Returns
        -------
        An `Explanation` object containing the prototypes, prototype indices and prototype labels with additional \
        metadata as attributes
        """
        if num_prototypes > len(self.X):
            num_prototypes = len(self.X)
            logger.warning('The number of prototypes requested is larger than the number of elements from '
                           f'the prototypes selection set. Automatically setting `num_prototypes={num_prototypes}`.')

        # dictionary of prototypes indices for each class
        protos = {l: [] for l in range(self.max_label + 1)}  # type: Dict[int, List[int]]  # noqa: E741
        # set of available prototypes indices. Note that initially we start with the entire set of Y,
        # but as the algorithm progresses, we remove the indices of the prototypes that we already selected.
        available_indices = set(range(len(self.X)))
        # matrix of size `[NY, NX]`, where `NY = len(Y)` and `NX = len(X)`
        # represents a mask which indicates for each element `y` in `Y` what are the elements of `X` that are in an
        # epsilon ball centered in `y`.
        B = (self.kmatrix <= self.eps).astype(np.int32)
        # matrix of size `[L, NX]`, where `L` is the number of labels
        # each row `l` indicates the elements from `X` that are covered by prototypes belonging to class `l`
        B_P = np.zeros((self.max_label + 1, len(self.X_ref)), dtype=np.int32)
        # matrix of size `[L, NX]`. Each row `l` indicates which elements form `X` are labeled as `l`
        Xl = np.concatenate([(self.Y_ref == l).reshape(1, -1)
                             for l in range(self.max_label + 1)], axis=0).astype(np.int32)  # noqa: E741

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
        # other classes different from `l` will be covered if we add the prototype `y`.
        delta_nu_summed = np.sum(delta_nu_all, axis=-1)
        # compute the tradeoff score - each prototype tries to cover as many new elements as possible
        # belonging to the same class, while trying to avoid covering elements belonging to another class
        scores_all = delta_xi_summed - delta_nu_summed - self.lambda_penalty

        for _ in tqdm(range(num_prototypes), disable=(not self.verbose)):
            j = np.array(list(available_indices)).astype(np.int32)
            scores = scores_all[j]

            # stopping criterion. The number of the returned prototypes might be lower than
            # the number of requested prototypes
            if np.all(scores < 0):
                break

            # find the index `i` of the best prototype and the class `l` that it covers
            row, col = np.unravel_index(np.argmax(scores), scores.shape)
            i, l = j[row.item()], col.item()  # noqa: E741

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
        data['prototypes_labels'] = np.concatenate([[l] * len(protos[l])
                                                    for l in protos]).astype(np.int32)  # noqa: E741
        data['prototypes'] = self.X[data['prototypes_indices']]
        return Explanation(meta=self.meta, data=data)

    def save(self, path: Union[str, os.PathLike]) -> None:
        super().save(path)

    @classmethod
    def load(cls, path: Union[str, os.PathLike]) -> "Explainer":  # type: ignore[override]
        return super().load(path, predictor=None)


def _helper_protoselect_euclidean_1knn(explainer: ProtoSelect,
                                       num_prototypes: int,
                                       eps: float) -> Optional[KNeighborsClassifier]:
    """
    Helper function to fit a 1-KNN classifier on the prototypes returned by the explainer.
    Sets the epsilon radius to be used.

    Parameters
    ----------
    explainer
        Fitted explainer.
    num_prototypes
        Number of requested prototypes.
    eps
        Epsilon radius to be set and used for the computation of prototypes.
     preprocess_fn
        Preprocessing function to be applied to the data instance before fitting the 1-KNN classifier

    Returns
    -------
    Fitted KNN-classifier.
    """
    # update explainer eps and get explanation
    explainer.eps = eps
    explanation = explainer.explain(num_prototypes=num_prototypes)

    # train 1-knn classifier
    proto, proto_labels = explanation.data['prototypes'], explanation.data['prototypes_labels']
    if len(proto) == 0:
        return None

    knn = KNeighborsClassifier(n_neighbors=1)
    return knn.fit(X=proto, y=proto_labels)


def cv_protoselect_euclidean(refset: Tuple[np.ndarray, np.ndarray],
                             protoset: Optional[Tuple[np.ndarray, ]] = None,
                             valset: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                             num_prototypes: int = 1,
                             eps_grid: Optional[np.ndarray] = None,
                             quantiles: Optional[Tuple[float, float]] = None,
                             grid_size: int = 25,
                             n_splits: int = 2,
                             batch_size: int = int(1e10),
                             preprocess_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                             **kwargs) -> dict:
    """
    Cross-validation parameter selection for ProtoSelect with Euclidean distance. The method computes
    the best epsilon radius.

    Parameters
    ----------
    refset
        Tuple, `(X_ref, X_ref_labels)`, consisting of the reference data instances with the corresponding reference
        labels.
    protoset
        Tuple, `(X_proto, )`, consisting of the prototypes selection set. Note that the argument is passed as a tuple
        with a single element for consistency reasons.
    valset
        Optional tuple `(X_val, X_val_labels)` consisting of validation data instances with the corresponding
        validation labels. 1-KNN classifier is evaluated on the validation dataset to obtain the best epsilon radius.
        In case ``valset=None``, then `n-splits` cross-validation is performed on the `refset`.
    num_prototypes
        The number of prototypes to be selected.
    eps_grid
        Optional grid of values to select the epsilon radius from. If not specified, the search grid is
        automatically proposed based on the inter-distances between `X_ref` and `X_proto`. The distances are filtered
        by considering only values in between the `quantiles` values. The minimum and maximum distance values are
        used to define the range of values to search the epsilon radius. The interval is discretized in `grid_size`
        equi-distant bins.
    quantiles
        Quantiles, `(q_min, q_max)`, to be used to filter the range of values of the epsilon radius. The expected
        quantile values are in `[0, 1]` and clipped to `[0, 1]` if outside the range. See `eps_grid` for usage.
        If not specified, no filtering is applied. Only used if ``eps_grid=None``.
    grid_size
        The number of equal-distant bins to be used to discretize the `eps_grid` proposed interval. Only used if
        ``eps_grid=None``.
    n_splits
        The number of cross-validation splits to be used. Default value 2. Only used if ``valset=None``.
    batch_size
        Batch size to be used for kernel matrix computation.
    preprocess_fn
        Preprocessing function to be applied to the data instance before applying the kernel.

    Returns
    -------
    Dictionary containing
     - ``'best_eps'``: ``float`` - best epsilon radius according to the accuracy of a 1-KNN classifier.
     - ``'meta'``: ``dict`` - dictionary containing argument and data gather throughout cross-validation.
    """
    X_ref, Y_ref = refset
    X = protoset[0] if (protoset is not None) else X_ref

    if preprocess_fn is not None:
        X_ref = _batch_preprocessing(X=X_ref, preprocess_fn=preprocess_fn, batch_size=batch_size)
        X = _batch_preprocessing(X=X, preprocess_fn=preprocess_fn, batch_size=batch_size)

    # propose eps_grid if not specified
    if eps_grid is None:
        dist = batch_compute_kernel_matrix(x=X_ref, y=X, kernel=EuclideanDistance()).reshape(-1)
        if quantiles is not None:
            if quantiles[0] > quantiles[1]:
                raise ValueError('The quantile lower-bound is greater then the quantile upper-bound.')

            quantiles = np.clip(quantiles, a_min=0, a_max=1)
            min_dist, max_dist = np.quantile(a=dist, q=quantiles)
        else:
            min_dist, max_dist = np.min(dist), np.max(dist)
        # define list of values for eps
        eps_grid = np.linspace(min_dist, max_dist, num=grid_size)

    if valset is None:
        kf = KFold(n_splits=n_splits)
        scores = np.zeros((len(eps_grid), n_splits))

        for i, (train_index, val_index) in enumerate(kf.split(X=X_ref, y=Y_ref)):
            X_ref_i, Y_ref_i = X_ref[train_index], Y_ref[train_index]
            X_val, Y_val = X_ref[val_index], Y_ref[val_index]

            # define and fit explainer here, so we don't repeat the kernel matrix computation in the next for loop
            explainer = ProtoSelect(kernel_distance=EuclideanDistance(), eps=0, **kwargs)
            explainer = explainer.fit(X_ref=X_ref_i, Y_ref=Y_ref_i, X=X)

            for j in range(len(eps_grid)):
                knn = _helper_protoselect_euclidean_1knn(explainer=explainer,
                                                         num_prototypes=num_prototypes,
                                                         eps=eps_grid[j])
                if knn is None:
                    continue
                scores[j][i] = knn.score(X_val, Y_val)
        best_eps = eps_grid[np.argmax(np.mean(scores, axis=-1))]
    else:
        scores = np.zeros(len(eps_grid))
        X_val, Y_val = valset
        if preprocess_fn is not None:
            X_val = _batch_preprocessing(X=X_val, preprocess_fn=preprocess_fn, batch_size=batch_size)

        # define and fit explainer, so we don't repeat the kernel matrix computation
        explainer = ProtoSelect(kernel_distance=EuclideanDistance(), eps=0, **kwargs)
        explainer = explainer.fit(X_ref=X_ref, Y_ref=Y_ref, X=X)

        for j in range(len(eps_grid)):
            knn = _helper_protoselect_euclidean_1knn(explainer=explainer,
                                                     num_prototypes=num_prototypes,
                                                     eps=eps_grid[j])
            if knn is None:
                continue
            scores[j] = knn.score(X_val, Y_val)
        best_eps = eps_grid[np.argmax(scores)]

    return {
        'best_eps': best_eps,
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
        Images' x-coordinates.
    y
        Images' y-coordinates.
    images
        Array of images to be placed at coordinates `(x, y)`.
    ax
        A `matplotlib` axes object to plot on.
    fig_kw
        Keyword arguments passed to the `fig.set` function.
    image_size
        Size of the generated output image as `(rows, cols)`.
    zoom
        Images' zoom to be used.
    zoom_lb
        Zoom lower bound. The zoom values will be scaled linearly between `[zoom_lb, zoom_up]`.
    zoom_ub
        Zoom upper bound. The zoom values will be scaled linearly between `[zoom_lb, zoom_up]`.
    """
    if fig_kw is None:
        fig_kw = {}

    if zoom is None:
        zoom = np.ones(len(images))

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


def visualize_prototypes(explanation: 'Explanation',
                         refset: Tuple[np.ndarray, np.ndarray],
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
    The size of each prototype is proportional to the log of the number of correct-class training images covered
    by that prototype (Bien and Tibshiran (2012): https://arxiv.org/abs/1202.5933).

    Parameters
    ----------
    explanation
        An `Explanation` object produced by a call to the
        :py:meth:`alibi.prototypes.protoselect.ProtoSelect.explain` method.
    refset
        Tuple, `(X_ref, X_ref_labels)`, consisting of the reference data instances with the corresponding reference
        labels.
    reducer
        2D reducer. Reduces the input feature representation to 2D. Note that the reducer operates directly on the
        input instances if ``preprocess_fn=None``. If the `preprocess_fn` is specified, the reducer will be called
        on the feature representation obtained after calling `preprocess_fn` on the input instances.
    preprocess_fn
        Preprocessor function.
    knn_kw
        Keyword arguments passed to `sklearn KNNClassifier` constructor.
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
    if knn_kw is None:
        knn_kw = {}
    if knn_kw.get('metric') is None:
        knn_kw.update({'metric': 'euclidean'})

    X_ref, Y_ref = refset
    X = explanation.data['prototypes']
    Y = explanation.data['prototypes_labels']

    # preprocess the dataset
    X_ref_ft = _batch_preprocessing(X=X_ref, preprocess_fn=preprocess_fn) \
        if (preprocess_fn is not None) else X_ref
    X_ft = _batch_preprocessing(X=X, preprocess_fn=preprocess_fn) \
        if (preprocess_fn is not None) else X

    # train knn classifier
    knn = KNeighborsClassifier(n_neighbors=1, **knn_kw)
    knn = knn.fit(X=X_ft, y=Y)

    # get neighbors indices for each training instance
    neigh_idx = knn.kneighbors(X=X_ref_ft, n_neighbors=1)[1].reshape(-1)

    # compute how many training instances each prototype covers
    idx, counts = np.unique(neigh_idx, return_counts=True)
    covered = {i: c for i, c in zip(idx, counts)}

    # compute how many correct labeled instances each prototype covers
    idx, counts = np.unique(neigh_idx[Y[neigh_idx] == Y_ref], return_counts=True)
    correct = {i: c for i, c in zip(idx, counts)}

    # compute zoom
    zoom = np.log([correct.get(i, 0) for i in covered])

    # compute 2D embedding
    X_protos_2d = reducer(X_ft)
    x, y = X_protos_2d[:, 0], X_protos_2d[:, 1]

    # plot images
    return _imscatterplot(x=x, y=y, images=X, ax=ax, fig_kw=fig_kw, image_size=image_size,
                          zoom=zoom, zoom_lb=zoom_lb, zoom_ub=zoom_ub)
