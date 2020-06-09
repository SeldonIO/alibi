import numpy as np
import warnings
from enum import Enum
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure, axis
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import ndarray
from typing import Union, Tuple


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
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    elif VisualizeSign[sign] == VisualizeSign.negative:
        attr_combined = (attr_combined < 0) * attr_combined
        threshold = -1 * _cumulative_sum_threshold(
            np.abs(attr_combined), 100 - outlier_perc
        )
    elif VisualizeSign[sign] == VisualizeSign.absolute_value:
        attr_combined = np.abs(attr_combined)
        threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
    else:
        raise AssertionError("Visualize Sign type is not valid.")
    return _normalize_scale(attr_combined, threshold)


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
    r"""
        Visualizes attribution for a given image by normalizing attribution values
        of the desired sign (positive, negative, absolute value, or all) and displaying
        them using the desired mode in a matplotlib figure.

        Parameters
        ----------

            attr
                Numpy array corresponding to attributions to be
                visualized. Shape must be in the form (H, W, C), with
                channels as last dimension. Shape must also match that of
                the original image if provided.
            original_image
                Numpy array corresponding to
                original image. Shape must be in the form (H, W, C), with
                channels as the last dimension. Image can be provided either
                with float values in range 0-1 or int values between 0-255.
                This is a necessary argument for any visualization method
                which utilizes the original image.
            method
                Chosen method for visualizing attribution.
                Supported options are:
                1. `heat_map` - Display heat map of chosen attributions
                2. `blended_heat_map` - Overlay heat map over greyscale
                version of original image. Parameter alpha_overlay
                corresponds to alpha of heat map.
                3. `original_image` - Only display original image.
                4. `masked_image` - Mask image (pixel-wise multiply)
                by normalized attribution values.
                5. `alpha_scaling` - Sets alpha channel of each pixel
                to be equal to normalized attribution value.
                Default: `heat_map`
            sign
                Chosen sign of attributions to visualize. Supported
                options are:
                1. `positive` - Displays only positive pixel attributions.
                2. `absolute_value` - Displays absolute value of
                attributions.
                3. `negative` - Displays only negative pixel attributions.
                4. `all` - Displays both positive and negative attribution
                values. This is not supported for `masked_image` or
                `alpha_scaling` modes, since signed information cannot
                be represented in these modes.

            plt_fig_axis
                Tuple of matplotlib.pyplot.figure and axis
                on which to visualize. If None is provided, then a new figure
                and axis are created.

            outlier_perc
                Top attribution values which
                correspond to a total of outlier_perc percentage of the
                total attribution are set to 1 and scaling is performed
                using the minimum of these values. For sign=`all`, outliers a
                nd scale value are computed using absolute value of
                attributions.

            cmap
                String corresponding to desired colormap for
                heatmap visualization. This defaults to "Reds" for negative
                sign, "Blues" for absolute value, "Greens" for positive sign,
                and a spectrum from red to green for all. Note that this
                argument is only used for visualizations displaying heatmaps.

            alpha_overlay
                Alpha to set for heatmap when using
                `blended_heat_map` visualization mode, which overlays the
                heat map over the greyscaled original image.

            show_colorbar
                Displays colorbar for heatmap below
                the visualization. If given method does not use a heatmap,
                then a colormap axis is created and hidden. This is
                necessary for appropriate alignment when visualizing
                multiple plots, some with colorbars and some without.

            title
                Title string for plot. If None, no title is set.

            fig_size
                Size of figure created.

            use_pyplot
                If true, uses pyplot to create and show
                figure and displays the figure after creating. If False,
                uses Matplotlib object oriented API and simply returns a
                figure object without showing.

        Returns
        -------
            2-element tuple of **figure**, **axis**:
            - **figure** (*matplotlib.pyplot.figure*):
                        Figure object on which visualization
                        is created. If plt_fig_axis argument is given, this is the
                        same figure provided.
            - **axis** (*matplotlib.pyplot.axis*):
                        Axis object on which visualization
                        is created. If plt_fig_axis argument is given, this is the
                        same axis provided.

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
            plt_axis.imshow(np.mean(original_image, axis=2), cmap="gray")
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
