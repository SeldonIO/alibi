import numpy as np
import alibi.utils.mapping as mp

X_ohe = np.array([[0, 1, 0.1, 1, 0, 0.2]]).astype(np.float32)
shape_ohe = X_ohe.shape
cat_vars_ohe = {0: 2, 3: 2}
is_ohe = True

X_ord = np.array([[1., 0.1, 0., 0.2]]).astype(np.float32)
shape_ord = X_ord.shape
cat_vars_ord = {0: 2, 2: 2}

dist = {0: np.array([.3, .4]),
        2: np.array([.5, .6, .7])}
X_num = np.array([[.4, .1, .5, .2]]).astype(np.float32)


def test_mapping_fn():
    # ohe_to_ord_shape
    assert mp.ohe_to_ord_shape(shape_ohe, cat_vars_ohe, is_ohe=True) == shape_ord

    # ohe_to_ord
    X_ohe_to_ord, cat_vars_ohe_to_ord = mp.ohe_to_ord(X_ohe, cat_vars_ohe)
    assert (X_ohe_to_ord == X_ord).all() and cat_vars_ohe_to_ord == cat_vars_ord

    # ord_to_ohe
    X_ord_to_ohe, cat_vars_ord_to_ohe = mp.ord_to_ohe(X_ord, cat_vars_ord)
    assert (X_ord_to_ohe == X_ohe).all() and cat_vars_ohe == cat_vars_ord_to_ohe

    # ord_to_num
    X_ord_to_num = mp.ord_to_num(X_ord, dist)
    assert (X_num == X_ord_to_num).all()

    # num_to_ord
    X_num_to_ord = mp.num_to_ord(X_num, dist)
    assert (X_ord == X_num_to_ord).all()
