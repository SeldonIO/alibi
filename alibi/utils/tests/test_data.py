import pytest
import pandas as pd
from alibi.utils.data import gen_category_map

CAT_COLUMNS_IX = [1, 3]
CAT_COLUMNS = ['col2', 'col4']
data = {'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'c', 'd', 'f'],
        'col3': [1., 2., 3., 4., 5.],
        'col4': ['a', 'a', 'b', 'b', 'c']}

expected_category_map = {1: ['a', 'b', 'c', 'd', 'f'],
                         3: ['a', 'b', 'c']}

dframe = pd.DataFrame(data=data)
arr = dframe.values


@pytest.mark.parametrize('df', [dframe, arr])
@pytest.mark.parametrize('categorical_columns', [None, [1, 3], ['col2', 'col4']])
def test_get_category_map(categorical_columns, df):
    # test numpy case with no categorical_columns raises error
    if df is arr and (categorical_columns is None or categorical_columns == ['col2', 'col4']):
        with pytest.raises(ValueError):
            _ = gen_category_map(df)

    else:
        category_map = gen_category_map(df, categorical_columns=categorical_columns)
        assert category_map == expected_category_map
