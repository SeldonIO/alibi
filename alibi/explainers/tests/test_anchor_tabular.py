# flake8: noqa E731

from alibi.explainers import AnchorTabular
import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


@pytest.mark.parametrize("predict_type", ("proba", "class"))
@pytest.mark.parametrize("threshold", (0.9, 0.95))
def test_iris(predict_type, threshold):
    # load iris dataset
    dataset = load_iris()
    feature_names = dataset.feature_names

    # define train and test set
    idx = 145
    X_train, Y_train = dataset.data[:idx, :], dataset.target[:idx]
    X_test, Y_test = dataset.data[idx + 1 :, :], dataset.target[idx + 1 :]  # noqa F841

    # fit random forest to training data
    np.random.seed(0)
    clf = RandomForestClassifier(n_estimators=50)
    clf.fit(X_train, Y_train)

    # define prediction function
    if predict_type == "proba":
        predict_fn = lambda x: clf.predict_proba(x)
    elif predict_type == "class":
        predict_fn = lambda x: clf.predict(x)

    # test explainer initialization
    explainer = AnchorTabular(predict_fn, feature_names)
    assert explainer.predict_fn(X_test[0].reshape(1, -1)).shape == (1,)

    # test explainer fit: shape and binning of ordinal features
    explainer.fit(X_train, disc_perc=[25, 50, 75])
    assert explainer.train_data.shape == explainer.d_train_data.shape == (145, 4)
    assert (np.unique(explainer.d_train_data) == np.array([0.0, 1.0, 2.0, 3.0])).all()
    assert explainer.categorical_features == explainer.ordinal_features

    # test sampling function
    sample_fn, mapping = explainer.get_sample_fn(X_test[0], desired_label=None)
    nb_samples = 5
    raw_data, data, labels = sample_fn(mapping, nb_samples)
    assert len(mapping) == data.shape[1]

    # test mapping dictionary used for sampling
    dsc = explainer.disc.discretize(raw_data)[0, :]
    for f, d in enumerate(dsc):
        m = 0
        while mapping[m][0] < f:
            m += 1
        if m + d >= len(mapping):
            assert mapping[len(mapping) - 1][1] == "geq"
            break
        assert mapping[m + d][1] == "leq"

    # test explanation
    explanation = explainer.explain(X_test[0], threshold=threshold)
    assert explanation["precision"] >= threshold
    assert explanation["coverage"] >= 0.05
