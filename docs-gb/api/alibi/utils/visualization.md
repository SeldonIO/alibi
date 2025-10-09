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

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `data` | `numpy.ndarray` |  |  |
| `xticklabels` | `List[str]` |  |  |
| `yticklabels` | `List[str]` |  |  |
| `vmin` | `Optional[float]` | `None` |  |
| `vmax` | `Optional[float]` | `None` |  |
| `cmap` | `Union[str, matplotlib.colors.Colormap]` | `'magma'` |  |
| `robust` | `Optional[bool]` | `False` |  |
| `annot` | `Optional[bool]` | `True` |  |
| `linewidths` | `float` | `3` |  |
| `linecolor` | `str` | `'w'` |  |
| `cbar` | `bool` | `True` |  |
| `cbar_label` | `str` | `''` |  |
| `cbar_ax` | `Optional[matplotlib.axes._axes.Axes]` | `None` |  |
| `cbar_kws` | `Optional[dict]` | `None` |  |
| `fmt` | `Union[str, matplotlib.ticker.Formatter]` | `'{x:.2f}'` |  |
| `textcolors` | `Tuple[str, str]` | `('white', 'black')` |  |
| `threshold` | `Optional[float]` | `None` |  |
| `text_kws` | `Optional[dict]` | `None` |  |
| `ax` | `Optional[matplotlib.axes._axes.Axes]` | `None` |  |

**Returns**
- Type: `matplotlib.axes._axes.Axes`

### `visualize_image_attr`

```python
visualize_image_attr(attr: numpy.ndarray, original_image: Optional[numpy.ndarray] = None, method: str = 'heat_map', sign: str = 'absolute_value', plt_fig_axis: Optional[Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]] = None, outlier_perc: Union[int, float] = 2, cmap: Optional[str] = None, alpha_overlay: float = 0.5, show_colorbar: bool = False, title: Optional[str] = None, fig_size: Tuple[int, int] = (6, 6), use_pyplot: bool = True) -> Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
```

Visualizes attribution for a given image by normalizing attribution values of the desired sign

(``'positive'`` | ``'negative'`` | ``'absolute_value'`` | ``'all'``) and displaying them using the desired mode
in a `matplotlib` figure.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `attr` | `numpy.ndarray` |  |  |
| `original_image` | `Optional[numpy.ndarray]` | `None` |  |
| `method` | `str` | `'heat_map'` |  |
| `sign` | `str` | `'absolute_value'` |  |
| `plt_fig_axis` | `Optional[Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]]` | `None` |  |
| `outlier_perc` | `Union[int, float]` | `2` |  |
| `cmap` | `Optional[str]` | `None` |  |
| `alpha_overlay` | `float` | `0.5` |  |
| `show_colorbar` | `bool` | `False` |  |
| `title` | `Optional[str]` | `None` |  |
| `fig_size` | `Tuple[int, int]` | `(6, 6)` |  |
| `use_pyplot` | `bool` | `True` |  |

**Returns**
- Type: `Tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]`
