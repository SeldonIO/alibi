import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Dict


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


def get_category_map(data: pd.DataFrame,
                     categorical_columns: list = None) -> Dict[int, list]:
    # TODO: support passing np.ndarray
    assert data.ndim == 2, 'Expected 2-dimensional dataframe'

    n_features = data.shape[1]
    if categorical_columns is None:
        # infer categorical columns
        categorical_columns = [i for i in range(n_features) if data.iloc[:, i].dtype == 'O']

    # create the map
    category_map = dict.fromkeys(categorical_columns)
    for col in categorical_columns:
        le = LabelEncoder()
        _ = le.fit_transform(data.iloc[:, col])
        category_map[col] = list(le.classes_)

    return category_map
