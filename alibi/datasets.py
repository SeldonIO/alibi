import logging
import pkgutil
import tarfile
from io import BytesIO, StringIO
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import PIL
import requests
import tensorflow.keras as keras
from requests import RequestException
from sklearn.preprocessing import LabelEncoder

from alibi.utils.data import Bunch

logger = logging.getLogger(__name__)

__all__ = ['fetch_adult',
           'fetch_fashion_mnist',
           'fetch_imagenet',
           'fetch_movie_sentiment',
           'load_cats']

ADULT_URLS = ['https://storage.googleapis.com/seldon-datasets/adult/adult.data',
              'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
              'http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data']

MOVIESENTIMENT_URLS = ['https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz',
                       'http://www.cs.cornell.edu/People/pabo/movie-review-data/rt-polaritydata.tar.gz']


def load_cats(target_size: tuple = (299, 299), return_X_y: bool = False) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """
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
    """
    tar = tarfile.open(fileobj=BytesIO(pkgutil.get_data(__name__, "data/cats.tar.gz")),  # type: ignore[arg-type]
                       mode='r:gz')
    images = []
    target = []
    target_names = []
    for member in tar.getmembers():
        # data
        img = tar.extractfile(member).read()  # type: ignore[union-attr]
        img = PIL.Image.open(BytesIO(img))
        img = np.expand_dims(img.resize(target_size), axis=0)
        images.append(img)

        # labels
        name = member.name.split('_')
        target.append(int(name.pop(1)))
        target_names.append('_'.join(name).split('.')[0])
    tar.close()

    images = np.concatenate(images, axis=0)  # type: ignore[assignment]
    targets = np.asarray(target)
    if return_X_y:
        return images, targets  # type: ignore[return-value] # TODO: allow redefiniton
    else:
        return Bunch(data=images, target=targets, target_names=target_names)


# deprecated
def fetch_imagenet(category: str = 'Persian cat', nb_images: int = 10, target_size: tuple = (299, 299),
                   min_std: float = 10., seed: int = 42, return_X_y: bool = False) -> None:
    import warnings
    warnings.warn("""The Imagenet API is no longer publicly available, as a result `fetch_imagenet` is deprecated.
To download images from Imagenet please follow instructions on http://image-net.org/download""")


def fetch_movie_sentiment(return_X_y: bool = False, url_id: int = 0) -> Union[Bunch, Tuple[list, list]]:
    """
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
    """
    url = MOVIESENTIMENT_URLS[url_id]
    try:
        resp = requests.get(url, timeout=2)
        resp.raise_for_status()
    except RequestException:
        logger.exception("Could not connect, URL may be out of service")
        raise

    tar = tarfile.open(fileobj=BytesIO(resp.content), mode="r:gz")
    data = []
    labels = []
    for i, member in enumerate(tar.getnames()[1:]):
        f = tar.extractfile(member)
        for line in f.readlines():  # type: ignore[union-attr]
            try:
                line.decode('utf8')
            except UnicodeDecodeError:
                continue
            data.append(line.decode('utf8').strip())
            labels.append(i)
    tar.close()
    if return_X_y:
        return data, labels

    target_names = ['negative', 'positive']
    return Bunch(data=data, target=labels, target_names=target_names)


def fetch_adult(features_drop: Optional[list] = None, return_X_y: bool = False, url_id: int = 0) -> \
        Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """
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
    """
    if features_drop is None:
        features_drop = ["fnlwgt", "Education-Num"]

    # download data
    dataset_url = ADULT_URLS[url_id]
    raw_features = ['Age', 'Workclass', 'fnlwgt', 'Education', 'Education-Num', 'Marital Status',
                    'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss',
                    'Hours per week', 'Country', 'Target']
    try:
        resp = requests.get(dataset_url)
        resp.raise_for_status()
    except RequestException:
        logger.exception("Could not connect, URL may be out of service")
        raise

    raw_data = pd.read_csv(StringIO(resp.text), names=raw_features, delimiter=', ', engine='python').fillna('?')

    # get labels, features and drop unnecessary features
    labels = (raw_data['Target'] == '>50K').astype(int).values
    features_drop += ['Target']
    data = raw_data.drop(features_drop, axis=1)
    features = list(data.columns)

    # map categorical features
    education_map = {
        '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
            'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
            'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
        'Some-college': 'High School grad', 'Masters': 'Masters',
        'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
        'Assoc-voc': 'Associates'
    }
    occupation_map = {
        "Adm-clerical": "Admin", "Armed-Forces": "Military",
        "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
        "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
            "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
            "Service", "Priv-house-serv": "Service", "Prof-specialty":
            "Professional", "Protective-serv": "Other", "Sales":
            "Sales", "Tech-support": "Other", "Transport-moving":
            "Blue-Collar"
    }
    country_map = {
        'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
            'China', 'Columbia': 'South-America', 'Cuba': 'Other',
        'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
        'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
        'France': 'Euro_1', 'Germany': 'Euro_1', 'Greece': 'Euro_2',
        'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
        'Holand-Netherlands': 'Euro_1', 'Honduras': 'Latin-America',
        'Hong': 'China', 'Hungary': 'Euro_2', 'India':
            'British-Commonwealth', 'Iran': 'Other', 'Ireland':
            'British-Commonwealth', 'Italy': 'Euro_1', 'Jamaica':
            'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
            'Latin-America', 'Nicaragua': 'Latin-America',
        'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
            'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_2',
        'Portugal': 'Euro_2', 'Puerto-Rico': 'Latin-America', 'Scotland':
            'British-Commonwealth', 'South': 'Euro_2', 'Taiwan': 'China',
        'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
        'United-States': 'United-States', 'Vietnam': 'SE-Asia'
    }
    married_map = {
        'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
        'Married-civ-spouse': 'Married', 'Married-spouse-absent':
            'Separated', 'Separated': 'Separated', 'Divorced':
            'Separated', 'Widowed': 'Widowed'
    }
    mapping = {'Education': education_map, 'Occupation': occupation_map, 'Country': country_map,
               'Marital Status': married_map}

    data_copy = data.copy()
    for f, f_map in mapping.items():
        data_tmp = data_copy[f].values
        for key, value in f_map.items():
            data_tmp[data_tmp == key] = value
        data[f] = data_tmp

    # get categorical features and apply labelencoding
    categorical_features = [f for f in features if data[f].dtype == 'O']
    category_map = {}
    for f in categorical_features:
        le = LabelEncoder()
        data_tmp = le.fit_transform(data[f].values)
        data[f] = data_tmp
        category_map[features.index(f)] = list(le.classes_)

    # only return data values
    data = data.values
    target_names = ['<=50K', '>50K']

    if return_X_y:
        return data, labels

    return Bunch(data=data, target=labels, feature_names=features, target_names=target_names, category_map=category_map)


def fetch_fashion_mnist(return_X_y: bool = False
                        ) -> Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """
    Loads the Fashion MNIST dataset.

    Parameters
    ----------
    return_X_y:
        If ``True``, an `N x M x P` array of data points and `N`-array of labels are returned
        instead of a dict.

    Returns
    -------
    If ``return_X_y=False``, a Bunch object with fields 'data', 'targets' and 'target_names'
    is returned. Otherwise an array with data points and an array of labels is returned.
    """

    target_names = {
        0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
        5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot',
    }

    data, labels = keras.datasets.fashion_mnist.load_data()[0]

    if return_X_y:
        return data, labels

    return Bunch(data=data, target=labels, target_names=target_names)
