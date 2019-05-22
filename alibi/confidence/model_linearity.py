import logging
import numpy as np
from sklearn.neighbors import NearestNeighbors
from functools import reduce

logger = logging.getLogger(__name__)


def _flatten_features(X_train):
	return X_train.reshape((X_train.shape[0],reduce( lambda x, y: x * y, X_train.shape[1:]))), X_train.shape[1:]


def _reshape_features(X, features_shape):
	return X.reshape((X.shape[0],) + features_shape)


def _calculate_linearity_measure(predict_fn, samples, alphas, verbose=False):

	ps = [predict_fn(samples[i: i + 1]) for i in range(len(alphas))]
	outs = [np.log(p + 1e-10) for p in ps]
	summ = reduce(lambda x, y: x + y, [alphas[i] * samples[i] for i in range(len(alphas))])
	summ = summ.reshape((1,) + summ.shape)
	out_sum = np.log(predict_fn(summ) + 1e-10)
	sum_out = reduce(lambda x, y: x + y, [alphas[i] * outs[i] for i in range(len(alphas))])

	if verbose:
		logger.debug(out_sum.shape)
		logger.debug(sum_out.shape)

	linearity_score = ((out_sum - sum_out) ** 2).sum()

	return out_sum, sum_out, linearity_score


def _sample_train(x, X_train, nb_samples=10, order=2, superposition='uniform', verbose=False):

	X_train, _ = _flatten_features(X_train)
	X_stack = np.stack([x for i in range(X_train.shape[0])], axis=0)

	X_stack, features_shape = _flatten_features(X_stack)
	nbrs = NearestNeighbors(n_neighbors=nb_samples, algorithm='ball_tree').fit(X_train)
	distances, indices = nbrs.kneighbors(X_stack)
	distances, indices = distances[0], indices[0]

	X_sampled = X_train[indices]

	X_sampled = _reshape_features(X_sampled, features_shape)
	x = x.reshape((1,) + x.shape)
	if verbose:
		logger.debug(x.shape)
		logger.debug(X_sampled.shape)

	X_pairs = [np.vstack((x, X_sampled[i:i + order - 1])) for i in range(X_sampled.shape[0])]

	if superposition == 'uniform':
		alphas = [1 / float(order) for j in range(order)]
	else:
		logs = np.asarray([np.random.rand() + np.random.randint(1) for i in range(order)])
		alphas = np.exp(logs) / np.exp(logs).sum()
	if verbose:
		logger.debug([X_tmp.shape for X_tmp in X_pairs])
		logger.debug(len(alphas))

	return X_pairs, alphas


def linearity_measure(predict_fn, x, X_train, nb_samples=10, order=2, superposition='uniform', verbose=False):

	X_pairs, alphas = _sample_train(x, X_train, nb_samples=nb_samples, order=order,
									superposition=superposition, verbose=verbose)
	scores = []
	for pair in X_pairs:
		out_sum, sum_out, score = _calculate_linearity_measure(predict_fn, pair, alphas, verbose=verbose)
		scores.append(score)
	return np.asarray(scores).mean()
