import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Union


# TODO: This should inherit from collections.UserDict not dict


class Bunch(dict):
    """
    Container object for internal datasets
    Dictionary-like object that exposes its keys as attributes.
    """

    def __init__(self, **kwargs):
        super().__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __dir__(self):
        return self.keys()

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


def gen_category_map(data: Union[pd.DataFrame, np.ndarray],
                     categorical_columns: Union[List[int], List[str], None] = None) -> Dict[int, list]:
    """

    Parameters
    ----------
    data
        2-dimensional pandas dataframe or numpy array.
    categorical_columns
        A list of columns indicating categorical variables. Optional if passing a pandas dataframe as inference will
        be used based on dtype 'O'. If passing a numpy array this is compulsory.

    Returns
    -------
    category_map
        A dictionary with keys being the indices of the categorical columns and values being lists of categories for
        that column. Implicitly each category is mapped to the index of its position in the list.

    """
    if data.ndim != 2:
        raise TypeError('Expected a 2-dimensional dataframe or array')
    n_features = data.shape[1]

    if isinstance(data, np.ndarray):
        # if numpy array, we need categorical_columns, otherwise impossible to infer
        if categorical_columns is None:
            raise ValueError('If passing a numpy array, `categorical_columns` is required')
        elif not all(isinstance(ix, int) for ix in categorical_columns):
            raise ValueError('If passing a numpy array, `categorical_columns` must be a list of integers')
        data = pd.DataFrame(data)

    # infer categorical columns
    if categorical_columns is None:
        try:
            categorical_columns = [i for i in range(n_features) if data.iloc[:, i].dtype == 'O']  # NB: 'O'
        except AttributeError:
            raise

    # create the map
    category_map = {}
    for col in categorical_columns:
        if not isinstance(col, int):
            col = int(data.columns.get_loc(col))
        le = LabelEncoder()
        try:
            _ = le.fit_transform(data.iloc[:, col])
        except (AttributeError, IndexError):
            raise

        category_map[col] = list(le.classes_)

    return category_map
