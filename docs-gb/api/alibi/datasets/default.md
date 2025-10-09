# `alibi.datasets.default`
## Functions
### `fetch_adult`

```python
fetch_adult(features_drop: Optional[list] = None, return_X_y: bool = False, url_id: int = 0) -> Union[alibi.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]
```

Downloads and pre-processes 'adult' dataset.

More info: http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `features_drop` | `Optional[list]` | `None` |  |
| `return_X_y` | `bool` | `False` |  |
| `url_id` | `int` | `0` |  |

**Returns**
- Type: `Union[alibi.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]`

### `fetch_imagenet`

```python
fetch_imagenet(category: str = 'Persian cat', nb_images: int = 10, target_size: tuple = (299, 299), min_std: float = 10.0, seed: int = 42, return_X_y: bool = False) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `category` | `str` | `'Persian cat'` |  |
| `nb_images` | `int` | `10` |  |
| `target_size` | `tuple` | `(299, 299)` |  |
| `min_std` | `float` | `10.0` |  |
| `seed` | `int` | `42` |  |
| `return_X_y` | `bool` | `False` |  |

**Returns**
- Type: `None`

### `fetch_imagenet_10`

```python
fetch_imagenet_10(url_id: int = 0) -> Dict
```

Sample dataset extracted from imagenet in a dictionary format.

The train set contains 1000 random samples, 100 for each of the following 10 selected classes:

* stingray
* trilobite
* centipede
* slug
* snail
* Rhodesian ridgeback
* beagle
* golden retriever
* sea lion
* espresso

The test set contains 50 random samples, 5 for each of the classes above.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `url_id` | `int` | `0` |  |

**Returns**
- Type: `Dict`

### `fetch_movie_sentiment`

```python
fetch_movie_sentiment(return_X_y: bool = False, url_id: int = 0) -> Union[alibi.utils.data.Bunch, Tuple[list, list]]
```

The movie review dataset, equally split between negative and positive reviews.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `return_X_y` | `bool` | `False` |  |
| `url_id` | `int` | `0` |  |

**Returns**
- Type: `Union[alibi.utils.data.Bunch, Tuple[list, list]]`

### `load_cats`

```python
load_cats(target_size: tuple = (299, 299), return_X_y: bool = False) -> Union[alibi.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]
```

A small sample of Imagenet-like public domain images of cats used primarily for examples.

The images were hand-collected using flickr.com by searching for various cat types, filtered by images
in the public domain.

Parameters

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `target_size` | `tuple` | `(299, 299)` |  |
| `return_X_y` | `bool` | `False` |  |

**Returns**
- Type: `Union[alibi.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]`
