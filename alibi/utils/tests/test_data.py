import pytest
import pandas as pd
from alibi.utils.data import get_category_map

CAT_COLUMNS = [1, 3]
data = {'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'c', 'd', 'f'],
        'col3': [1., 2., 3., 4., 5.],
        'col4': ['a', 'a', 'b', 'b', 'c']}

df = pd.DataFrame(data=data)


@pytest.mark.parametrize('categorical_columns', [None, [1, 3]])
def test_get_category_map(categorical_columns):
    category_map = get_category_map(df, categorical_columns=categorical_columns)
    assert list(category_map.keys()) == CAT_COLUMNS

    for C in CAT_COLUMNS:
        assert len(set(df.iloc[:, C].values)) == len(category_map[C])
