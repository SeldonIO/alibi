
import pytest
import numpy as np
from typing import Any, Dict, Tuple, List


def tf_models(tf_models: Tuple[str]):
    """
    This fixture loads a list of pre-trained test-models by name from the
    alibi-testing helper package.
    """
    from alibi_testing.loading import load
    return [load(name) for name in tf_models]


def tf_keras_iris_explainer(models: List[Any], iris_data: Dict[str, Any], use_kdtree: bool, c_init: float, c_steps: int):
    X_train = iris_data['X_train']
    model, ae, enc = models

    if use_kdtree:
        ae, enc = None, None

    shape = (1, 4)
    feature_range = (X_train.min(axis=0).reshape(shape), X_train.max(axis=0).reshape(shape))

    from alibi.explainers import CounterfactualProto
    cf_explainer = CounterfactualProto(
        model, 
        shape, 
        gamma=100, theta=100,
        ae_model=ae, enc_model=enc, 
        use_kdtree=use_kdtree, max_iterations=1000, 
        c_init=c_init, c_steps=c_steps,
        feature_range=feature_range
    )
    
    return model, cf_explainer

@pytest.mark.parametrize('tf_model_names', [('iris-ffn-tf2.18.0.h5', 'iris-ae-tf2.18.0.h5', 'iris-enc-tf2.18.0.h5')])
@pytest.mark.parametrize(
    'use_kdtree, c_init, c_steps, k',
    [
        (False, 0., 1, None),
        (False, 1., 3, None),
        (False, 0., 1, 2),
        (False, 1., 3, 2),
        (True, 0., 1, None),
        (True, 1., 3, None),
        (True, 0., 1, 2),
        (True, 1., 3, 2)
    ],
)
def test_tf_keras_iris_explainer(set_env_variables, tf_model_names, use_kdtree, c_init, c_steps, k):
    # load data
    from alibi_testing.data import get_iris_data
    iris_data = get_iris_data()

    # load models
    models = tf_models(tf_model_names)

    # load explainer
    model, cf = tf_keras_iris_explainer(
        models=models,
        iris_data=iris_data,
        use_kdtree=use_kdtree,
        c_init=c_init,
        c_steps=c_steps
    )

    # instance to be explained
    X_train = iris_data['X_train']
    x = X_train[0].reshape(1, -1)

    predictions = model.predict(x)
    pred_class = np.argmax(predictions)
    not_pred_class = np.argmin(predictions)

    # Run the fit method inside the session
    cf.fit(X_train)
    
    if use_kdtree:  # k-d trees
        assert len(cf.kdtrees) == cf.classes  # each class has a k-d tree
        assert cf.kdtrees[pred_class].query(x, k=1)[0] == 0.  # nearest distance to own class equals 0
        assert cf.score(x, not_pred_class, pred_class) == 0.  # test score fn

        n_by_class = np.sum([cf.X_by_class[c].shape[0] for c in range(cf.classes)])
        assert n_by_class == X_train.shape[0]  # all training instances are stored in the trees
    else:  # encoder
        assert len(list(cf.class_proto.keys())) == cf.classes
        assert [True for _ in range(cf.classes)] == [v.shape == (1, 2) for _, v in cf.class_proto.items()]

        n_by_class = np.sum([cf.class_enc[c].shape[0] for c in range(cf.classes)])
        assert n_by_class == X_train.shape[0]  # all training instances encoded

    # test explanation
    explanation = cf.explain(x, k=k)
    assert cf.id_proto != pred_class
    assert np.argmax(model.predict(explanation.cf['X'])) == explanation.cf['class']
    assert explanation.cf['grads_num'].shape == explanation.cf['grads_graph'].shape == x.shape

    from alibi.api.defaults import DEFAULT_META_CFP, DEFAULT_DATA_CFP
    assert explanation.meta.keys() == DEFAULT_META_CFP.keys()
    assert explanation.data.keys() == DEFAULT_DATA_CFP.keys()

    # test gradient shapes
    y = np.zeros((1, cf.classes))
    np.put(y, pred_class, 1)
    cf.predict = cf.predict.predict  # make model black box
    grads = cf.get_gradients(x, y, x.shape[1:], cf.cat_vars_ord)
    assert grads.shape == x.shape


def tf_keras_adult_explainer(models: List[Any], adult_data: Dict[str, Any], use_kdtree: bool, c_init: float, c_steps: int):
    shape = (1, 57)
    cat_vars_ohe = adult_data['metadata']['cat_vars_ohe']

    from alibi.explainers import CounterfactualProto
    cf_explainer = CounterfactualProto(
        models[0], 
        shape, 
        beta=.01, 
        cat_vars=cat_vars_ohe, ohe=True,
        use_kdtree=use_kdtree, max_iterations=1000,
        c_init=c_init, c_steps=c_steps,
        feature_range=(-1 * np.ones((1, 12)), np.ones((1, 12)))
    )
    
    return models[0], cf_explainer


@pytest.mark.parametrize('tf_model_names', [('adult-ffn-tf2.18.0.h5',)])
@pytest.mark.parametrize(
    'use_kdtree, c_init, c_steps, k, d_type',
    [
        (False, 1., 3, None, 'mvdm'),
        (True, 1., 3, 2, 'mvdm'),
        (True, 1., 3, 2, 'abdm'),
    ]
)
def test_tf_keras_adult_explainer(set_env_variables, tf_model_names, use_kdtree, c_init, c_steps, k, d_type):
    # load data
    from alibi_testing.data import get_adult_data
    adult_data = get_adult_data()

    # load models
    models = tf_models(tf_model_names)

    # load explainer
    model, cf = tf_keras_adult_explainer(
        models=models,
        adult_data=adult_data,
        use_kdtree=use_kdtree,
        c_init=c_init,
        c_steps=c_steps
    )

    # instance to be explained
    X_train = adult_data['preprocessor'].transform(adult_data['X_train']).toarray()
    x = X_train[0].reshape(1, -1)
    pred_class = np.argmax(model.predict(x))

    # test fit
    cf.fit(X_train, d_type=d_type)

    # checked ragged tensor shape
    n_cat = len(list(cf.cat_vars_ord.keys()))
    max_key = max(cf.cat_vars_ord, key=cf.cat_vars_ord.get)
    max_cat = cf.cat_vars_ord[max_key]
    assert cf.d_abs_ragged.shape == (n_cat, max_cat)

    if use_kdtree:  # k-d trees
        assert len(cf.kdtrees) == cf.classes  # each class has a k-d tree
        n_by_class = 0
        for c in range(cf.classes):
            n_by_class += cf.X_by_class[c].shape[0]
        assert n_by_class == X_train.shape[0]  # all training instances are stored in the trees

    # test explanation
    num_shape = (1, 12)
    explanation = cf.explain(x, k=k)
    
    if use_kdtree:
        assert cf.id_proto != pred_class

    assert np.argmax(model.predict(explanation.cf['X'])) == explanation.cf['class']
    assert explanation.cf['grads_num'].shape == explanation.cf['grads_graph'].shape == num_shape

    from alibi.api.defaults import DEFAULT_META_CFP, DEFAULT_DATA_CFP
    assert explanation.meta.keys() == DEFAULT_META_CFP.keys()
    assert explanation.data.keys() == DEFAULT_DATA_CFP.keys()

    # test gradient shapes
    y = np.zeros((1, cf.classes))
    np.put(y, pred_class, 1)
    cf.predict = cf.predict.predict  # make model black box

    # convert instance to numerical space
    from alibi.utils.mapping import ohe_to_ord, ord_to_num
    x_ord = ohe_to_ord(x, cf.cat_vars)[0]
    x_num = ord_to_num(x_ord, cf.d_abs)
    
    # check gradients
    grads = cf.get_gradients(x_num, y, num_shape[1:], cf.cat_vars_ord)
    assert grads.shape == num_shape
