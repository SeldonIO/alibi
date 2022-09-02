import warnings
from enum import Enum
from typing import List, Optional, Tuple, Union

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.pyplot import axis, figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ndarray


# the following code was borrowed from the captum library in
# https://github.com/pytorch/captum/blob/master/captum/attr/_utils/visualization.py
class ImageVisualizationMethod(Enum):
    heat_map = 1
    blended_heat_map = 2
    original_image = 3
    masked_image = 4
    alpha_scaling = 5


class VisualizeSign(Enum):
    positive = 1
    absolute_value = 2
    negative = 3
    all = 4


def _prepare_image(attr_visual: ndarray):
    return np.clip(attr_visual.astype(int), 0, 255)


def _normalize_scale(attr: ndarray, scale_factor: float):
    if abs(scale_factor) < 1e-5:
        warnings.warn(
            "Attempting to normalize by value approximately 0, skipping normalization."
            "This likely means that attribution values are all close to 0."
        )
        return np.clip(attr, -1, 1)
    attr_norm = attr / scale_factor
    return np.clip(attr_norm, -1, 1)


def _cumulative_sum_threshold(values: ndarray, percentile: Union[int, float]):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]


def _normalize_image_attr(
        attr: ndarray, sign: str, outlier_perc: Union[int, float] = 2
):
    attr_combined = np.sum(attr, axis=2)
    # Choose appropriate signed values and rescale, removing given outlier percentage.
    if VisualizeSign[sign] == VisualizeSign.all:
        threshold = _cumulative_sum_threshold(np.abs(attr_combined), 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.positive:
        attr_combined = (attr_combined > 0) * attr_combined
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)  # type: ignore
    elif VisualizeSign[sign] == VisualizeSign.negative:
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold(
            np.abs(attr_combined), 100 - outlier_perc
        )
    elif VisualizeSign[sign] == VisualizeSign.absolute_value:
        attr_combined = np.abs(attr_combined)
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)  # type: ignore
    else:
        raise AssertionError("Visualize Sign type is not valid.")
    return _normalize_scale(attr_combined, threshold)  # type: ignore


def visualize_image_attr(
        attr: ndarray,
        original_image: Union[None, ndarray] = None,
        method: str = "heat_map",
        sign: str = "absolute_value",
        plt_fig_axis: Union[None, Tuple[figure, axis]] = None,
        outlier_perc: Union[int, float] = 2,
        cmap: Union[None, str] = None,
        alpha_overlay: float = 0.5,
        show_colorbar: bool = False,
        title: Union[None, str] = None,
        fig_size: Tuple[int, int] = (6, 6),
        use_pyplot: bool = True,
):
    """
    Visualizes attribution for a given image by normalizing attribution values of the desired sign
    (``'positive'`` | ``'negative'`` | ``'absolute_value'`` | ``'all'``) and displaying them using the desired mode
    in a `matplotlib` figure.

    Parameters
    ----------
    attr
        `Numpy` array corresponding to attributions to be visualized. Shape must be in the form `(H, W, C)`, with
        channels as last dimension. Shape must also match that of the original image if provided.
    original_image
        `Numpy` array corresponding to original image. Shape must be in the form `(H, W, C)`, with channels as the
        last dimension. Image can be provided either with `float` values in range 0-1 or `int` values between 0-255.
        This is a necessary argument for any visualization method which utilizes the original image.
    method
        Chosen method for visualizing attribution. Supported options are:

         - ``'heat_map'`` - Display heat map of chosen attributions

         - ``'blended_heat_map'`` - Overlay heat map over greyscale version of original image. Parameter alpha_overlay \
        corresponds to alpha of heat map.

         - ``'original_image'`` - Only display original image.

         - ``'masked_image``' - Mask image (pixel-wise multiply) by normalized attribution values.

         - ``'alpha_scaling'`` - Sets alpha channel of each pixel to be equal to normalized attribution value.

        Default: ``'heat_map'``.
    sign
        Chosen sign of attributions to visualize. Supported options are:

         - ``'positive'`` - Displays only positive pixel attributions.

         - ``'absolute_value'`` - Displays absolute value of attributions.

         - ``'negative'`` - Displays only negative pixel attributions.

         - ``'all'`` - Displays both positive and negative attribution values. This is not supported for \
         ``'masked_image'`` or ``'alpha_scaling'`` modes, since signed information cannot be represented \
         in these modes.

    plt_fig_axis
        Tuple of `matplotlib.pyplot.figure` and `axis` on which to visualize. If ``None`` is provided, then a new
        figure and axis are created.
    outlier_perc
        Top attribution values which correspond to a total of `outlier_perc` percentage of the total attribution are
        set to 1 and scaling is performed using the minimum of these values. For ``sign='all'``, outliers and scale
        value are computed using absolute value of attributions.
    cmap
        String corresponding to desired colormap for heatmap visualization. This defaults to ``'Reds'`` for negative
        sign, ``'Blues'`` for absolute value, ``'Greens'`` for positive sign, and a spectrum from red to green for all.
        Note that this argument is only used for visualizations displaying heatmaps.
    alpha_overlay
        Visualizes attribution for a given image by normalizing attribution values of the desired sign (positive,
        negative, absolute value, or all) and displaying them using the desired mode in a matplotlib figure.
    show_colorbar
        Displays colorbar for heatmap below the visualization. If given method does not use a heatmap,
        then a colormap axis is created and hidden. This is necessary for appropriate alignment when visualizing
        multiple plots, some with colorbars and some without.
    title
        The title for the plot. If ``None``, no title is set.
    fig_size
        Size of figure created.
    use_pyplot
        If ``True``, uses pyplot to create and show figure and displays the figure after creating. If ``False``,
        uses `matplotlib` object-oriented API and simply returns a figure object without showing.

    Returns
    -------
    2-element tuple of consisting of
     - `figure` : ``matplotlib.pyplot.figure`` - Figure object on which visualization is created. If `plt_fig_axis` \
     argument is given, this is the same figure provided.

     - `axis` : ``matplotlib.pyplot.axis`` - Axis object on which visualization is created. If `plt_fig_axis` argument \
     is given, this is the same axis provided.

    """
    # Create plot if figure, axis not provided
    if plt_fig_axis is not None:
        plt_fig, plt_axis = plt_fig_axis
    else:
        if use_pyplot:
            plt_fig, plt_axis = plt.subplots(figsize=fig_size)
        else:
            plt_fig = Figure(figsize=fig_size)
            plt_axis = plt_fig.subplots()

    if original_image is not None:
        if np.max(original_image) <= 1.0:
            original_image = _prepare_image(original_image * 255)
    else:
        assert (
                ImageVisualizationMethod[method] == ImageVisualizationMethod.heat_map
        ), "Original Image must be provided for any visualization other than heatmap."

    # Remove ticks and tick labels from plot.
    plt_axis.xaxis.set_ticks_position("none")
    plt_axis.yaxis.set_ticks_position("none")
    plt_axis.set_yticklabels([])
    plt_axis.set_xticklabels([])

    heat_map = None
    # Show original image
    if ImageVisualizationMethod[method] == ImageVisualizationMethod.original_image:
        plt_axis.imshow(original_image)
    else:
        # Choose appropriate signed attributions and normalize.
        norm_attr = _normalize_image_attr(attr, sign, outlier_perc)

        # Set default colormap and bounds based on sign.
        if VisualizeSign[sign] == VisualizeSign.all:
            default_cmap = LinearSegmentedColormap.from_list(
                "RdWhGn", ["red", "white", "green"]
            )
            vmin, vmax = -1, 1
        elif VisualizeSign[sign] == VisualizeSign.positive:
            default_cmap = "Greens"
            vmin, vmax = 0, 1
        elif VisualizeSign[sign] == VisualizeSign.negative:
            default_cmap = "Reds"
            vmin, vmax = 0, 1
        elif VisualizeSign[sign] == VisualizeSign.absolute_value:
            default_cmap = "Blues"
            vmin, vmax = 0, 1
        else:
            raise AssertionError("Visualize Sign type is not valid.")
        cmap = cmap if cmap is not None else default_cmap

        # Show appropriate image visualization.
        if ImageVisualizationMethod[method] == ImageVisualizationMethod.heat_map:
            heat_map = plt_axis.imshow(norm_attr, cmap=cmap, vmin=vmin, vmax=vmax)
        elif (
                ImageVisualizationMethod[method]
                == ImageVisualizationMethod.blended_heat_map
        ):
            plt_axis.imshow(np.mean(original_image, axis=2), cmap="gray")  # type: ignore[arg-type]
            heat_map = plt_axis.imshow(
                norm_attr, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha_overlay
            )
        elif ImageVisualizationMethod[method] == ImageVisualizationMethod.masked_image:
            assert VisualizeSign[sign] != VisualizeSign.all, (
                "Cannot display masked image with both positive and negative "
                "attributions, choose a different sign option."
            )
            plt_axis.imshow(
                _prepare_image(original_image * np.expand_dims(norm_attr, 2))
            )
        elif ImageVisualizationMethod[method] == ImageVisualizationMethod.alpha_scaling:
            assert VisualizeSign[sign] != VisualizeSign.all, (
                "Cannot display alpha scaling with both positive and negative "
                "attributions, choose a different sign option."
            )
            plt_axis.imshow(
                np.concatenate(
                    [
                        original_image,
                        _prepare_image(np.expand_dims(norm_attr, 2) * 255),
                    ],
                    axis=2,
                )
            )
        else:
            raise AssertionError("Visualize Method type is not valid.")

    # Add colorbar. If given method is not a heatmap and no colormap is relevant,
    # then a colormap axis is created and hidden. This is necessary for appropriate
    # alignment when visualizing multiple plots, some with heatmaps and some
    # without.
    if show_colorbar:
        axis_separator = make_axes_locatable(plt_axis)
        colorbar_axis = axis_separator.append_axes("bottom", size="5%", pad=0.1)
        if heat_map:
            plt_fig.colorbar(heat_map, orientation="horizontal", cax=colorbar_axis)
        else:
            colorbar_axis.axis("off")
    if title:
        plt_axis.set_title(title)

    if use_pyplot:
        plt.show()

    return plt_fig, plt_axis


def _create_heatmap(data: np.ndarray,
                    xticklabels: List[str],
                    yticklabels: List[str],
                    linewidths: float = 3,
                    linecolor: str = 'w',
                    cbar: bool = True,
                    cbar_kws: Optional[dict] = None,
                    cbar_ax: Optional['plt.Axes'] = None,
                    cbar_label: str = "",
                    ax: Optional['plt.Axes'] = None,
                    **kwargs) -> 'plt.Axes':
    """
    Create a heatmap from a `numpy` array and two lists of labels. The code is adapted from `matplotlib tutorials`_.

    .. _matplotlib tutorials:
        https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    data
        A 2D `numpy` array of shape `M x N`.
    xticklabels
        A list or array of length `N` with the labels for the columns.
    yticklabels
        A list or array of length `M` with the labels for the rows.
    linewidths
        Width of the lines that will divide each cell. Default 3.
    linecolor
        Color of the lines that will divide each cell. Default ``'w'``.
    cbar
        Boolean flag whether to draw a colorbar.
    cbar_label
        Optional label for the colorbar.
    cbar_ax
        Optional axes in which to draw the colorbar, otherwise take space from the main axes.
    cbar_kws
        An optional dictionary with arguments to `matplotlib.figure.Figure.colorbar`_.

        .. _matplotlib.figure.Figure.colorbar:
            https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.colorbar
    ax
        Optional `matplotlib.axes.Axes` instance to which the heatmap is plotted. If not provided, use current
        axes or create a new one.
    **kwargs
        All other keyword arguments are passed to `matplotlib.axes.Axes.imshow`_.

        .. _matplotlib.axes.Axes.imshow:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html
    """
    if cbar_kws is None:
        cbar_kws = {}

    if not ax:
        ax = plt.gca()

    # plot the heatmap
    im = ax.imshow(data, **kwargs)

    # create colorbar
    if cbar:
        if cbar_ax is None:
            cbar_ax = ax
        cbar_obj = ax.figure.colorbar(im, ax=cbar_ax, **cbar_kws)
        cbar_obj.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

    # show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels(yticklabels)

    # let the horizontal axes labeling appear on top.
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    # turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color=linecolor, linestyle='-', linewidth=linewidths)
    ax.tick_params(which="minor", bottom=False, left=False)

    return ax


def _annotate_heatmap(im: matplotlib.image.AxesImage,
                      data: Optional[np.ndarray] = None,
                      fmt: Union[str, matplotlib.ticker.Formatter] = '{x:.2f}',
                      textcolors: Tuple[str, str] = ('black', 'white'),
                      threshold: Optional[float] = None,
                      **kwargs):
    """
    A function to annotate a heatmap. The code is adapted from `matplotlib tutorials`_.

    .. _matplotlib tutorials:
        https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    im
        The `matplotlib.image.AxesImage` to be labeled.
    data
        Optional 2D `numpy` array of shape `M x N` used to annotate the cells.  If ``None``, the image's data is used.
    fmt
        Format of the annotations inside the heatmap. This should either use the string format method,
        e.g. ``"{x:.2f}"``, or be a `matplotlib.ticker.Formatter`. Default ``"{x:.2f}"``.
    textcolors
        A tuple of `matplotlib` colors. The first is used for values below a threshold, the second for those above.
        Default ``('black', 'white')``.
    threshold
        Optional value in data units according to which the colors from `textcolors` are applied.
        If ``None`` (the default) uses the middle of the colormap as separation.
    **kwargs
        All other keyword arguments are passed to `matplotlib.axes.Axes.text`_.

        .. _matplotlib.axes.Axes.text:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(np.max(data)) / 2.

    # set default alignment to center, but allow it to be overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(kwargs)

    # get the formatter in case a string is supplied
    if isinstance(fmt, str):
        fmt = matplotlib.ticker.StrMethodFormatter(fmt)

    # loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, fmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def heatmap(data: np.ndarray,
            xticklabels: List[str],
            yticklabels: List[str],
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
            cmap: Union[str, matplotlib.colors.Colormap] = 'magma',
            robust: Optional[bool] = False,
            annot: Optional[bool] = True,
            linewidths: float = 3,
            linecolor: str = 'w',
            cbar: bool = True,
            cbar_label: str = '',
            cbar_ax: Optional['plt.Axes'] = None,
            cbar_kws: Optional[dict] = None,
            fmt: Union[str, matplotlib.ticker.Formatter] = '{x:.2f}',
            textcolors: Tuple[str, str] = ('white', 'black'),
            threshold: Optional[float] = None,
            text_kws: Optional[dict] = None,
            ax: Optional['plt.Axes'] = None,
            **kwargs) -> 'plt.Axes':
    """
    Constructs a heatmap with annotation.

    Parameters
    ----------
    data
        A 2D `numpy` array of shape `M x N`.
    yticklabels
        A list or array of length `M` with the labels for the rows.
    xticklabels
        A list or array of length `N` with the labels for the columns.
    vmin, vmax
       When using scalar data and no explicit norm, `vmin` and `vmax` define the data range that the colormap covers.
       By default, the colormap covers the complete value range of the supplied data. It is an error to use
       `vmin/vmax` when norm is given. When using RGB(A) data, parameters `vmin/vmax` are ignored.
    cmap
        The Colormap instance or registered colormap name used to map scalar data to colors. This parameter is
        ignored for RGB(A) data.
    robust
        If ``True`` and `vmin` or `vmax` are absent, the colormap range is computed with robust quantiles
        instead of the extreme values. Uses `numpy.nanpercentile`_ with `q` values set to 2 and 98, respectively.

        .. _numpy.nanpercentile:
            https://numpy.org/doc/stable/reference/generated/numpy.nanpercentile.html

    annot
        Boolean flag whether to annotate the heatmap. Default ``True``.
    linewidths
        Width of the lines that will divide each cell. Default 3.
    linecolor
        Color of the lines that will divide each cell. Default ``"w"``.
    cbar
        Boolean flag whether to draw a colorbar.
    cbar_label
        Optional label for the colorbar.
    cbar_ax
        Optional axes in which to draw the colorbar, otherwise take space from the main axes.
    cbar_kws
        An optional dictionary with arguments to `matplotlib.figure.Figure.colorbar`_.

        .. _matplotlib.figure.Figure.colorbar:
            https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.colorbar

    fmt
        Format of the annotations inside the heatmap. This should either use the string format method,
        e.g. ``"{x:.2f}"``, or be a `matplotlib.ticker.Formatter`_. Default ``"{x:.2f}"``.

        .. _matplotlib.ticker.Formatter:
            https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.Formatter

    textcolors
        A tuple of `matplotlib` colors. The first is used for values below a threshold,
        the second for those above. Default ``("black", "white")``.
    threshold
        Optional value in data units according to which the colors from textcolors are
        applied. If ``None`` (the default) uses the middle of the colormap as
        separation.
    text_kws
        An optional dictionary with arguments to `matplotlib.axes.Axes.text`_.

        .. _matplotlib.axes.Axes.text:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html

    ax
        Axes in which to draw the plot, otherwise use the currently-active axes.
    kwargs
        All other keyword arguments are passed to `matplotlib.axes.Axes.imshow`_.

        .. _matplotlib.axes.Axes.imshow:
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html

    Returns
    -------
    Axes object with the heatmap.
    """

    # get heatmap min value
    if vmin is None:
        vmin = np.nanpercentile(data, 2) if robust else np.nanmin(data)

    # get heatmap max value
    if vmax is None:
        vmax = np.nanpercentile(data, 98) if robust else np.nanmax(data)

    # create the heatmap
    ax = _create_heatmap(data=data,
                         yticklabels=yticklabels,
                         xticklabels=xticklabels,
                         ax=ax,
                         vmin=vmin,
                         vmax=vmax,
                         cmap=cmap,
                         linewidths=linewidths,
                         linecolor=linecolor,
                         cbar=cbar,
                         cbar_kws=cbar_kws,
                         cbar_ax=cbar_ax,
                         cbar_label=cbar_label,
                         **kwargs)

    if annot:
        if text_kws is None:
            text_kws = {}

        # annotate the heatmap
        _annotate_heatmap(im=ax.get_images()[0],
                          fmt=fmt,
                          textcolors=textcolors,
                          threshold=threshold,
                          **text_kws)
    return ax
