# `alibi.utils.visualization`
## `ImageVisualizationMethod`

_Inherits from:_ `Enum`

An enumeration.

## `VisualizeSign`

_Inherits from:_ `Enum`

An enumeration.

## Functions
### `heatmap`

```python
heatmap(data: numpy.ndarray, xticklabels: List[str], yticklabels: List[str], vmin: Optional[float] = None, vmax: Optional[float] = None, cmap: Union[str, matplotlib.colors.Colormap] = 'magma', robust: Optional[bool] = False, annot: Optional[bool] = True, linewidths: float = 3, linecolor: str = 'w', cbar: bool = True, cbar_label: str = '', cbar_ax: Optional[matplotlib.axes._axes.Axes] = None, cbar_kws: Optional[dict] = None, fmt: Union[str, matplotlib.ticker.Formatter] = '{x:.2f}', textcolors: Tuple[str, str] = ('white', 'black'), threshold: Optional[float] = None, text_kws: Optional[dict] = None, ax: Optional[matplotlib.axes._axes.Axes] = None, kwargs) -> matplotlib.axes._axes.Axes
```

Constructs a heatmap with annotation.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data` | `numpy.ndarray` |  | A 2D `numpy` array of shape `M x N`. |
| `xticklabels` | `List[str]` |  | A list or array of length `N` with the labels for the columns. |
| `yticklabels` | `List[str]` |  | A list or array of length `M` with the labels for the rows. |
| `vmin` | `Optional[float]` | `None` |  |
| `vmax` | `Optional[float]` | `None` |  |
| `cmap` | `Union[str, matplotlib.colors.Colormap]` | `'magma'` | The Colormap instance or registered colormap name used to map scalar data to colors. This parameter is ignored for RGB(A) data. |
| `robust` | `Optional[bool]` | `False` | If ``True`` and `vmin` or `vmax` are absent, the colormap range is computed with robust quantiles instead of the extreme values. Uses `numpy.nanpercentile`_ with `q` values set to 2 and 98, respectively. .. _numpy.nanpercentile: https://numpy.org/doc/stable/reference/generated/numpy.nanpercentile.html |
| `annot` | `Optional[bool]` | `True` | Boolean flag whether to annotate the heatmap. Default ``True``. |
| `linewidths` | `float` | `3` | Width of the lines that will divide each cell. Default 3. |
| `linecolor` | `str` | `'w'` | Color of the lines that will divide each cell. Default ``"w"``. |
| `cbar` | `bool` | `True` | Boolean flag whether to draw a colorbar. |
| `cbar_label` | `str` | `''` | Optional label for the colorbar. |
| `cbar_ax` | `Optional[matplotlib.axes._axes.Axes]` | `None` | Optional axes in which to draw the colorbar, otherwise take space from the main axes. |
| `cbar_kws` | `Optional[dict]` | `None` | An optional dictionary with arguments to `matplotlib.figure.Figure.colorbar`_. .. _matplotlib.figure.Figure.colorbar: https://matplotlib.org/stable/api/figure_api.html#matplotlib.figure.Figure.colorbar |
| `fmt` | `Union[str, matplotlib.ticker.Formatter]` | `'{x:.2f}'` | Format of the annotations inside the heatmap. This should either use the string format method, e.g. ``"{x:.2f}"``, or be a `matplotlib.ticker.Formatter`_. Default ``"{x:.2f}"``. .. _matplotlib.ticker.Formatter: https://matplotlib.org/stable/api/ticker_api.html#matplotlib.ticker.Formatter |
| `textcolors` | `Tuple[str, str]` | `('white', 'black')` | A tuple of `matplotlib` colors. The first is used for values below a threshold, the second for those above. Default ``("black", "white")``. |
| `threshold` | `Optional[float]` | `None` | Optional value in data units according to which the colors from textcolors are applied. If ``None`` (the default) uses the middle of the colormap as separation. |
| `text_kws` | `Optional[dict]` | `None` | An optional dictionary with arguments to `matplotlib.axes.Axes.text`_. .. _matplotlib.axes.Axes.text: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.text.html |
| `ax` | `Optional[matplotlib.axes._axes.Axes]` | `None` | Axes in which to draw the plot, otherwise use the currently-active axes. |
| `kwargs` |  |  | All other keyword arguments are passed to `matplotlib.axes.Axes.imshow`_. .. _matplotlib.axes.Axes.imshow: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html |
| `vmin,` | `vmax` |  | When using scalar data and no explicit norm, `vmin` and `vmax` define the data range that the colormap covers. By default, the colormap covers the complete value range of the supplied data. It is an error to use `vmin/vmax` when norm is given. When using RGB(A) data, parameters `vmin/vmax` are ignored. |

**Returns**
- Type: `matplotlib.axes._axes.Axes`

### `visualize_image_attr`

```python
visualize_image_attr(attr: numpy.ndarray, original_image: Optional[numpy.ndarray] = None, method: str = 'heat_map', sign: str = 'absolute_value', plt_fig_axis: Optional[Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]] = None, outlier_perc: Union[int, float] = 2, cmap: Optional[str] = None, alpha_overlay: float = 0.5, show_colorbar: bool = False, title: Optional[str] = None, fig_size: Tuple[int, int] = (6, 6), use_pyplot: bool = True) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
```

Visualizes attribution for a given image by normalizing attribution values of the desired sign
(``'positive'`` | ``'negative'`` | ``'absolute_value'`` | ``'all'``) and displaying them using the desired mode
in a `matplotlib` figure.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `attr` | `numpy.ndarray` |  | `Numpy` array corresponding to attributions to be visualized. Shape must be in the form `(H, W, C)`, with channels as last dimension. Shape must also match that of the original image if provided. |
| `original_image` | `Optional[numpy.ndarray]` | `None` | `Numpy` array corresponding to original image. Shape must be in the form `(H, W, C)`, with channels as the last dimension. Image can be provided either with `float` values in range 0-1 or `int` values between 0-255. This is a necessary argument for any visualization method which utilizes the original image. |
| `method` | `str` | `'heat_map'` | Chosen method for visualizing attribution. Supported options are: - ``'heat_map'`` - Display heat map of chosen attributions - ``'blended_heat_map'`` - Overlay heat map over greyscale version of original image. Parameter alpha_overlay         corresponds to alpha of heat map. - ``'original_image'`` - Only display original image. - ``'masked_image``' - Mask image (pixel-wise multiply) by normalized attribution values. - ``'alpha_scaling'`` - Sets alpha channel of each pixel to be equal to normalized attribution value. Default: ``'heat_map'``. |
| `sign` | `str` | `'absolute_value'` | Chosen sign of attributions to visualize. Supported options are: - ``'positive'`` - Displays only positive pixel attributions. - ``'absolute_value'`` - Displays absolute value of attributions. - ``'negative'`` - Displays only negative pixel attributions. - ``'all'`` - Displays both positive and negative attribution values. This is not supported for          ``'masked_image'`` or ``'alpha_scaling'`` modes, since signed information cannot be represented          in these modes. |
| `plt_fig_axis` | `Optional[Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]]` | `None` | Tuple of `matplotlib.pyplot.figure` and `axis` on which to visualize. If ``None`` is provided, then a new figure and axis are created. |
| `outlier_perc` | `Union[int, float]` | `2` | Top attribution values which correspond to a total of `outlier_perc` percentage of the total attribution are set to 1 and scaling is performed using the minimum of these values. For ``sign='all'``, outliers and scale value are computed using absolute value of attributions. |
| `cmap` | `Optional[str]` | `None` | String corresponding to desired colormap for heatmap visualization. This defaults to ``'Reds'`` for negative sign, ``'Blues'`` for absolute value, ``'Greens'`` for positive sign, and a spectrum from red to green for all. Note that this argument is only used for visualizations displaying heatmaps. |
| `alpha_overlay` | `float` | `0.5` | Visualizes attribution for a given image by normalizing attribution values of the desired sign (positive, negative, absolute value, or all) and displaying them using the desired mode in a matplotlib figure. |
| `show_colorbar` | `bool` | `False` | Displays colorbar for heatmap below the visualization. If given method does not use a heatmap, then a colormap axis is created and hidden. This is necessary for appropriate alignment when visualizing multiple plots, some with colorbars and some without. |
| `title` | `Optional[str]` | `None` | The title for the plot. If ``None``, no title is set. |
| `fig_size` | `Tuple[int, int]` | `(6, 6)` | Size of figure created. |
| `use_pyplot` | `bool` | `True` | If ``True``, uses pyplot to create and show figure and displays the figure after creating. If ``False``, uses `matplotlib` object-oriented API and simply returns a figure object without showing. |

**Returns**
- Type: `Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]`
