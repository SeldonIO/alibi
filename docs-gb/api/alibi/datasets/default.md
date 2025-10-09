# `alibi.datasets.default`
## Functions
### `fetch_adult`

```python
fetch_adult(features_drop: Optional[list] = None, return_X_y: bool = False, url_id: int = 0) -> Union[alibi.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]
```

Downloads and pre-processes 'adult' dataset.

More info: http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/

Parameters
----------
features_drop
    List of features to be dropped from dataset, by default drops ``["fnlwgt", "Education-Num"]``.
return_X_y
    If ``True``, return features `X` and labels `y` as `numpy` arrays. If ``False`` return a `Bunch` object.
url_id
    Index specifying which URL to use for downloading.

Returns
-------
Bunch
    Dataset, labels, a list of features and a dictionary containing a list with the potential categories
    for each categorical feature where the key refers to the feature column.
(data, target)
    Tuple if ``return_X_y=True``

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
----------
url_id
    Index specifying which URL to use for downloading.

Returns
-------
Dictionary with the following keys:

    * trainset - train set tuple (X_train, y_train)
    * testset - test set tuple (X_test, y_test)
    * int_to_str_labels - map from target to target name
    * str_to_int_labels -  map from target name to target

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
----------
return_X_y
    If ``True``, return features `X` and labels `y` as `Python` lists. If ``False`` return a `Bunch` object.
url_id
    Index specifying which URL to use for downloading

Returns
-------
Bunch
    Movie reviews and sentiment labels (0 means 'negative' and 1 means 'positive').
(data, target)
    Tuple if ``return_X_y=True``.

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
----------
target_size
    Size of the returned images, used to crop images for a specified model input size.
return_X_y
    If ``True``, return features `X` and labels `y` as `numpy` arrays. If ``False`` return a `Bunch` object

Returns
-------
Bunch
    Bunch object with fields 'data', 'target' and 'target_names'. Both `targets` and `target_names` are taken from
    the original Imagenet.
(data, target)
    Tuple if ``return_X_y=True``.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `target_size` | `tuple` | `(299, 299)` |  |
| `return_X_y` | `bool` | `False` |  |

**Returns**
- Type: `Union[alibi.utils.data.Bunch, Tuple[numpy.ndarray, numpy.ndarray]]`
