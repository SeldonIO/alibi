from bs4 import BeautifulSoup
import PIL
from io import BytesIO, StringIO
import json
import numpy as np
import pandas as pd
import pkgutil
import random
import requests
from requests import RequestException
from sklearn.preprocessing import LabelEncoder
import tarfile
from typing import Tuple, Union
import logging
from alibi.utils.data import Bunch

import tensorflow.keras as keras

logger = logging.getLogger(__name__)

__all__ = ['fetch_adult',
           'fetch_fashion_mnist',
           'fetch_imagenet',
           'fetch_movie_sentiment']

ADULT_URLS = ['https://storage.googleapis.com/seldon-datasets/adult/adult.data',
              'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
              'http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data']

MOVIESENTIMENT_URLS = ['https://storage.googleapis.com/seldon-datasets/sentence_polarity_v1/rt-polaritydata.tar.gz',
                       'http://www.cs.cornell.edu/People/pabo/movie-review-data/rt-polaritydata.tar.gz']


# deprecated functions
def imagenet(category: str = 'Persian cat', nb_images: int = 10, target_size: tuple = (299, 299),
             min_std: float = 10., seed: int = 42):
    import warnings
    warnings.warn('`imagenet` is deprecated and will be removed soon, use `fetch_imagenet` instead',
                  DeprecationWarning, stacklevel=2)
    return fetch_imagenet(category=category, nb_images=nb_images, target_size=target_size, min_std=min_std,
                          seed=seed, return_X_y=True)


def movie_sentiment():
    import warnings
    warnings.warn('`movie_sentiment` is deprecated and will be removed soon, use `fetch_movie_sentiment` instead',
                  DeprecationWarning, stacklevel=2)
    return fetch_movie_sentiment(return_X_y=True)


def adult(features_drop=None):
    if features_drop is None:
        features_drop = ["fnlwgt", "Education-Num"]
    import warnings
    warnings.warn('`adult` is deprecated and will be removed soon, use `fetch_adult` instead', DeprecationWarning,
                  stacklevel=2)
    bunch = fetch_adult(features_drop)
    return bunch.data, bunch.target, bunch.feature_names, bunch.category_map


def fetch_imagenet(category: str = 'Persian cat', nb_images: int = 10, target_size: tuple = (299, 299),
                   min_std: float = 10., seed: int = 42, return_X_y: bool = False) -> Union[Bunch, Tuple[np.ndarray,
                                                                                                         np.ndarray]]:
    """
    Retrieve imagenet images from specified category which needs to be in the mapping dictionary.

    Parameters
    ----------
    category
        Imagenet category class name.
        Must be one of keys present in alibi/data/imagenet_class_names_to_id.json
    nb_images
        Number of images to be retrieved
    target_size
        Size of the returned images
    min_std
        Min standard deviation of image pixels. Images that are no longer available can be returned
        without content which is undesirable. Having a min std cutoff resolves this.
    seed
        Random seed
    return_X_y
        If true, return features X and labels y as numpy arrays, if False return a Bunch object

    Returns
    -------
    Bunch
        List with images and the labels from imagenet.
    (data, target)
        Tuple if ``return_X_y`` is true
    """
    # load the mappings
    class_names_to_id = json.loads(pkgutil.get_data(__name__, "data/imagenet_class_names_to_id.json"))
    class_names_to_label_idx = json.loads(pkgutil.get_data(__name__, "data/imagenet_class_names_to_label_idx.json"))

    url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=' + class_names_to_id[category]
    try:
        page = requests.get(url)
        page.raise_for_status()
    except RequestException:
        logger.exception('Imagenet API down')
        raise
    soup = BeautifulSoup(page.content, 'html.parser')
    img_urls = str(soup).split('\r\n')  # list of url's
    random.seed(seed)
    random.shuffle(img_urls)  # shuffle image list
    data = []
    nb = 0
    for img_url in img_urls:
        try:
            resp = requests.get(img_url, timeout=2)
            resp.raise_for_status()
        except RequestException:
            continue
        try:
            image = PIL.Image.open(BytesIO(resp.content)).convert('RGB')
        except OSError:
            continue
        image = np.expand_dims(image.resize(target_size), axis=0)
        if np.std(image) < min_std:  # do not include empty images
            continue
        data.append(image)
        nb += 1
        if nb == nb_images:
            break
    data = np.concatenate(data, axis=0)

    label_idx = class_names_to_label_idx[category]
    labels = np.array([label_idx for _ in range(nb_images)])

    if return_X_y:
        return data, labels

    target_names = [category for _ in range(nb_images)]
    return Bunch(data=data, target=labels, target_names=target_names)


def fetch_movie_sentiment(return_X_y: bool = False, url_id: int = 0) -> Union[Bunch, Tuple[list, list]]:
    """
    The movie review dataset, equally split between negative and positive reviews.

    Parameters
    ----------
    return_X_y
        If true, return features X and labels y as Python lists, if False return a Bunch object
    url_id
        Index specifying which URL to use for downloading

    Returns
    -------
    Bunch
        Movie reviews and sentiment labels (0 means 'negative' and 1 means 'positive').
    (data, target)
        Tuple if ``return_X_y`` is true
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
        for line in f.readlines():
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


def fetch_adult(features_drop: list = None, return_X_y: bool = False, url_id: int = 0) -> \
        Union[Bunch, Tuple[np.ndarray, np.ndarray]]:
    """
    Downloads and pre-processes 'adult' dataset.
    More info: http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/

    Parameters
    ----------
    features_drop
        List of features to be dropped from dataset, by default drops ["fnlwgt", "Education-Num"]
    return_X_y
        If true, return features X and labels y as numpy arrays, if False return a Bunch object
    url_id
        Index specifying which URL to use for downloading

    Returns
    -------
    Bunch
        Dataset, labels, a list of features and a dictionary containing a list with the potential categories
        for each categorical feature where the key refers to the feature column.
    (data, target)
        Tuple if ``return_X_y`` is true
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


def fetch_fashion_mnist(return_X_y: bool = False):
    """
    Loads the Fashion MNIST dataset.

    Parameters
    ----------
    return_X_y:
        If True, an NxMxP array of data points and N-array of labels are returned
        instead of a dict.

    Returns
    -------
    If return_X_y is False, a Bunch object with fields 'data', 'targets' and 'target_names'
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
