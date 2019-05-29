from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from alibi.confidence.model_linearity import linearity_measure


def test_linearity_measure():

	iris = load_iris()
	X_train = iris.data
	y_train = iris.target
	x = X_train[0]

	lg = LogisticRegression()
	lg.fit(X_train, y_train)

	predict_fn = lambda x: lg.predict_proba(x)

	lin = linearity_measure(predict_fn, x, X_train)

	assert lin >= 0, 'Linearity measure must be >= 0'
