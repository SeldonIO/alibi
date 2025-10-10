# `alibi.datasets.default`
## Constants
### `logger`
```python
logger: logging.Logger = <Logger alibi.datasets.default (WARNING)>
```
Instances of the Logger class represent a single logging channel. A
"logging channel" indicates an area of an application. Exactly how an
"area" is defined is up to the application developer. Since an
application can have any number of areas, logging channels are identified
by a unique string. Application areas can be nested (e.g. an area
of "input processing" might include sub-areas "read CSV files", "read
XLS files" and "read Gnumeric files"). To cater for this natural nesting,
channel names are organized into a namespace hierarchy where levels are
separated by periods, much like the Java or Python package namespace. So
in the instance given above, channel names might be "input" for the upper
level, and "input.csv", "input.xls" and "input.gnu" for the sub-levels.
There is no arbitrary limit to the depth of nesting.

### `ADULT_URLS`
```python
ADULT_URLS: list = ['https://storage.googleapis.com/seldon-datasets/adult/adult.data', 'https://...
```
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### `MOVIESENTIMENT_URLS`
```python
MOVIESENTIMENT_URLS: list = ['https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-pola...
```
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### `IMAGENET_URLS`
```python
IMAGENET_URLS: list = ['https://storage.googleapis.com/seldon-datasets/imagenet10/imagenet10.tar.gz']
```
Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

## Functions
### `fetch_adult`

```python
fetch_adult(features_drop: Optional[list] = None, return_X_y: bool = False, url_id: int = 0) -> Union[alibi.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]
```

Downloads and pre-processes 'adult' dataset.
More info: http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `features_drop` | `Optional[list]` | `None` | List of features to be dropped from dataset, by default drops ``["fnlwgt", "Education-Num"]``. |
| `return_X_y` | `bool` | `False` | If ``True``, return features `X` and labels `y` as `numpy` arrays. If ``False`` return a `Bunch` object. |
| `url_id` | `int` | `0` | Index specifying which URL to use for downloading. |

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

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `url_id` | `int` | `0` | Index specifying which URL to use for downloading. |

**Returns**
- Type: `Dict`

### `fetch_movie_sentiment`

```python
fetch_movie_sentiment(return_X_y: bool = False, url_id: int = 0) -> Union[alibi.utils.data.Bunch, Tuple[list, list]]
```

The movie review dataset, equally split between negative and positive reviews.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `return_X_y` | `bool` | `False` | If ``True``, return features `X` and labels `y` as `Python` lists. If ``False`` return a `Bunch` object. |
| `url_id` | `int` | `0` | Index specifying which URL to use for downloading |

**Returns**
- Type: `Union[alibi.utils.data.Bunch, Tuple[list, list]]`

### `load_cats`

```python
load_cats(target_size: tuple = (299, 299), return_X_y: bool = False) -> Union[alibi.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]
```

A small sample of Imagenet-like public domain images of cats used primarily for examples.
The images were hand-collected using flickr.com by searching for various cat types, filtered by images
in the public domain.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `target_size` | `tuple` | `(299, 299)` | Size of the returned images, used to crop images for a specified model input size. |
| `return_X_y` | `bool` | `False` | If ``True``, return features `X` and labels `y` as `numpy` arrays. If ``False`` return a `Bunch` object |

**Returns**
- Type: `Union[alibi.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]`
