import numpy as np
import copy
import collections
import logging
from typing import Callable, Tuple, Set, Dict, Sequence

logger = logging.getLogger(__name__)


def matrix_subset(matrix: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Parameters
    ----------
    matrix
        Matrix to sample from
    n_samples
        Number of samples returned

    Returns
    -------
    Sample of the input matrix.
    """
    if matrix.shape[0] == 0:
        return matrix
    n_samples = min(matrix.shape[0], n_samples)
    return matrix[np.random.choice(matrix.shape[0], n_samples, replace=False)]


class AnchorBaseBeam(object):

    def __init__(self) -> None:
        """
        Initialize the anchor beam search class.
        """
        pass

    @staticmethod
    def kl_bernoulli(p: float, q: float) -> float:
        """
        Compute KL-divergence between 2 probabilities p and q.

        Parameters
        ----------
        p
            Probability
        q
            Probability

        Returns
        -------
        KL-divergence
        """
        p = min(0.9999999999999999, max(0.0000001, p))
        q = min(0.9999999999999999, max(0.0000001, q))
        return (p * np.log(float(p) / q) + (1 - p) *
                np.log(float(1 - p) / (1 - q)))

    @staticmethod
    def dup_bernoulli(p: float, level: float, n_iter: int = 17) -> float:
        """
        Update upper precision bound for a candidate anchor dependent on the KL-divergence.

        Parameters
        ----------
        p
            Precision of candidate anchor
        level
            beta / nb of samples
        n_iter
            Number of iterations during lower bound update

        Returns
        -------
        Updated upper precision bound
        """
        # TODO: where does 17x sampling come from?
        lm = p
        um = min(min(1, p + np.sqrt(level / 2.)), 1)
        for j in range(1, n_iter):
            qm = (um + lm) / 2.
            if AnchorBaseBeam.kl_bernoulli(p, qm) > level:
                um = qm
            else:
                lm = qm
        return um

    @staticmethod
    def dlow_bernoulli(p: float, level: float, n_iter: int = 17) -> float:
        """
        Update lower precision bound for a candidate anchor dependent on the KL-divergence.

        Parameters
        ----------
        p
            Precision of candidate anchor
        level
            beta / nb of samples
        n_iter
            Number of iterations during lower bound update

        Returns
        -------
        Updated lower precision bound
        """
        um = p
        lm = max(min(1, p - np.sqrt(level / 2.)), 0)  # lower bound
        for j in range(1, n_iter):
            qm = (um + lm) / 2.
            if AnchorBaseBeam.kl_bernoulli(p, qm) > level:  # KL-divergence > threshold level
                lm = qm
            else:
                um = qm
        return lm

    @staticmethod
    def compute_beta(n_features: int, t: int, delta: float) -> float:
        """
        Parameters
        ----------
        n_features
            Number of candidate anchors
        t
            Iteration number
        delta

        Returns
        -------
        Level used to update upper and lower precision bounds.
        """
        # TODO: where do magic numbers come from?
        alpha = 1.1
        k = 405.5
        temp = np.log(k * n_features * (t ** alpha) / delta)
        return temp + np.log(temp)

    @staticmethod
    def lucb(sample_fns: list, initial_stats: dict, epsilon: float, delta: float, batch_size: int, top_n: int,
             verbose: bool = False, verbose_every: int = 1) -> Sequence:
        """
        Parameters
        ----------
        sample_fns
            List with sample functions for each candidate anchor
        initial_stats
            Dictionary with lists containing nb of samples used and where sample predictions equal the desired label
        epsilon
            Precision bound tolerance for convergence
        delta
            Used to compute beta
        batch_size
            Number of samples
        top_n
            Min of beam width size or number of candidate anchors
        verbose
            Whether to print intermediate output
        verbose_every
            Whether to print intermediate output every verbose_every steps

        Returns
        -------
        Indices of best anchor options. Number of indices equals min of beam width or nb of candidate anchors.
        """
        # n_features equals to the nb of candidate anchors
        n_features = len(sample_fns)

        # initiate arrays for number of samples, positives (sample prediction equals desired label), ...
        # ... upper and lower precision bounds for each anchor candidate
        n_samples = np.array(initial_stats['n_samples'])
        positives = np.array(initial_stats['positives'])
        ub = np.zeros(n_samples.shape)
        lb = np.zeros(n_samples.shape)
        for f in np.where(n_samples == 0)[0]:
            n_samples[f] += 1  # set min samples for each anchor candidate to 1
            positives[f] += sample_fns[f](1)  # add labels.sum() for the anchor candidate

        if n_features == top_n:  # return all options b/c of beam search width
            return range(n_features)

        means = positives / n_samples  # fraction sample predictions equal to desired label
        t = 1

        def update_bounds(t: int) -> Tuple[np.ndarray, np.ndarray]:
            """
            Parameters
            ----------
            t
                Iteration number

            Returns
            -------
            Upper and lower precision bound indices.
            """
            sorted_means = np.argsort(means)  # ascending sort of anchor candidates by precision

            beta = AnchorBaseBeam.compute_beta(n_features, t, delta)

            # J = the beam width top anchor candidates with highest precision
            # not_J = the rest
            J = sorted_means[-top_n:]
            not_J = sorted_means[:-top_n]

            for f in not_J:  # update upper bound for lowest precision anchor candidates
                ub[f] = AnchorBaseBeam.dup_bernoulli(means[f], beta / n_samples[f])

            for f in J:  # update lower bound for highest precision anchor candidates
                lb[f] = AnchorBaseBeam.dlow_bernoulli(means[f], beta / n_samples[f])

            # for the low precision anchor candidates, compute the upper precision bound and keep the index ...
            # ... of the anchor candidate with the highest upper precision value -> ut
            # for the high precision anchor candidates, compute the lower precision bound and keep the index ...
            # ... of the anchor candidate with the lowest lower precision value -> lt
            ut = not_J[np.argmax(ub[not_J])]
            lt = J[np.argmin(lb[J])]
            return ut, lt

        # keep updating the upper and lower precision bounds until the difference between the best upper ...
        # ... precision bound of the low precision anchors and the worst lower precision bound of the high ...
        # ... precision anchors is smaller than eps
        ut, lt = update_bounds(t)
        B = ub[ut] - lb[lt]
        verbose_count = 0
        while B > epsilon:
            verbose_count += 1
            if verbose and verbose_count % verbose_every == 0:
                print('Best: %d (mean:%.10f, n: %d, lb:%.4f)' %
                      (lt, means[lt], n_samples[lt], lb[lt]), end=' ')
                print('Worst: %d (mean:%.4f, n: %d, ub:%.4f)' %
                      (ut, means[ut], n_samples[ut], ub[ut]), end=' ')
                print('B = %.2f' % B)
            n_samples[ut] += batch_size
            positives[ut] += sample_fns[ut](batch_size)  # sample new batch of data
            means[ut] = positives[ut] / n_samples[ut]
            n_samples[lt] += batch_size
            positives[lt] += sample_fns[lt](batch_size)  # sample new batch of data
            means[lt] = positives[lt] / n_samples[lt]
            t += 1
            ut, lt = update_bounds(t)
            B = ub[ut] - lb[lt]
        sorted_means = np.argsort(means)
        return sorted_means[-top_n:]

    @staticmethod
    def make_tuples(previous_best: list, state: dict) -> list:
        """
        Parameters
        ----------
        previous_best
            List with tuples of anchor candidates
        state
            Dictionary with the relevant metrics like coverage and samples for candidate anchors

        Returns
        -------
        List with tuples of candidate anchors with additional metadata.
        """
        # compute some variables used later on
        normalize_tuple = lambda x: tuple(sorted(set(x)))  # noqa E731
        all_features = range(state['n_features'])
        coverage_data = state['coverage_data']
        current_idx = state['current_idx']
        data = state['data'][:current_idx]
        labels = state['labels'][:current_idx]

        # initially, every feature separately is an anchor
        if len(previous_best) == 0:
            tuples = [(x,) for x in all_features]
            for x in tuples:
                pres = data[:, x[0]].nonzero()[0]
                state['t_idx'][x] = set(pres)
                state['t_nsamples'][x] = float(len(pres))
                state['t_positives'][x] = float(labels[pres].sum())
                state['t_order'][x].append(x[0])
                state['t_coverage_idx'][x] = set(coverage_data[:, x[0]].nonzero()[0])
                state['t_coverage'][x] = (float(len(state['t_coverage_idx'][x])) / coverage_data.shape[0])
            return tuples

        # create new anchors: add a feature to every anchor in current best
        new_tuples = set()  # type: Set[tuple]
        for f in all_features:
            for t in previous_best:
                new_t = normalize_tuple(t + (f,))
                if len(new_t) != len(t) + 1:
                    continue
                if new_t not in new_tuples:
                    new_tuples.add(new_t)
                    state['t_order'][new_t] = copy.deepcopy(state['t_order'][t])
                    state['t_order'][new_t].append(f)
                    state['t_coverage_idx'][new_t] = (state['t_coverage_idx'][t].intersection(
                        state['t_coverage_idx'][(f,)]))
                    state['t_coverage'][new_t] = (float(len(state['t_coverage_idx'][new_t])) / coverage_data.shape[0])
                    t_idx = np.array(list(state['t_idx'][t]))
                    t_data = state['data'][t_idx]
                    present = np.where(t_data[:, f] == 1)[0]
                    state['t_idx'][new_t] = set(t_idx[present])
                    idx_list = list(state['t_idx'][new_t])
                    state['t_nsamples'][new_t] = float(len(idx_list))
                    state['t_positives'][new_t] = np.sum(state['labels'][idx_list])
        return list(new_tuples)

    @staticmethod
    def get_sample_fns(sample_fn: Callable, tuples: list, state: dict, data_type: str = None) -> list:
        """
        Parameters
        ----------
        sample_fn
            Sample function, returns both raw and categorized data as well as labels
        tuples
            List of anchor candidates
        state
            Dictionary with the relevant metrics like coverage and samples for candidate anchors
        data_type
            Data type for raw data

        Returns
        -------
        List with sample functions for each candidate anchor.
        """
        def complete_sample_fn(t: tuple, n: int) -> int:
            """
            Parameters
            ----------
            t
                Anchor candidate
            n
                Number of samples

            Returns
            -------
            Sum of where sampled data equals desired label of observation to be explained.
            """
            raw_data, data, labels = sample_fn(list(t), n)
            current_idx = state['current_idx']
            idxs = range(current_idx, current_idx + n)
            state['t_idx'][t].update(idxs)
            state['t_nsamples'][t] += n
            state['t_positives'][t] += labels.sum()
            state['data'][idxs] = data
            state['raw_data'][idxs] = raw_data
            state['labels'][idxs] = labels
            state['current_idx'] += n
            if state['current_idx'] >= state['data'].shape[0] - max(1000, n):
                prealloc_size = state['prealloc_size']
                current_idx = data.shape[0]
                state['data'] = np.vstack((state['data'], np.zeros((prealloc_size, data.shape[1]), data.dtype)))
                dtype = data_type if data_type is not None else raw_data.dtype
                state['raw_data'] = np.vstack((state['raw_data'], np.zeros((prealloc_size, raw_data.shape[1]),
                                                                           dtype=dtype)))
                state['labels'] = np.hstack((state['labels'], np.zeros(prealloc_size, labels.dtype)))
            return labels.sum()

        sample_fns = []
        for t in tuples:
            sample_fns.append(lambda n, t=t: complete_sample_fn(t, n))

        return sample_fns

    @staticmethod
    def get_initial_statistics(tuples: list, state: dict) -> dict:
        """
        Parameters
        ----------
        tuples
            Candidate anchors
        state
            Dictionary with the relevant metrics like coverage and samples for candidate anchors

        Returns
        -------
        Dictionary with lists containing nb of samples used and where sample predictions equal the desired label.
        """
        stats = {
            'n_samples': [],
            'positives': []
        }  # type: Dict[str, list]
        for t in tuples:
            stats['n_samples'].append(state['t_nsamples'][t])
            stats['positives'].append(state['t_positives'][t])
        return stats

    @staticmethod
    def get_anchor_from_tuple(t: tuple, state: dict) -> dict:
        """
        Parameters
        ----------
        t
            Anchor
        state
            Dictionary with the relevant metrics like coverage and samples for candidate anchors

        Returns
        -------
        Anchor dictionary with anchor features and additional metadata.
        """
        # TODO: This is wrong, some of the intermediate anchors may not exist.
        anchor = {'feature': [], 'mean': [], 'precision': [],
                  'coverage': [], 'examples': [], 'all_precision': 0}  # type: dict
        anchor['num_preds'] = state['data'].shape[0]
        normalize_tuple = lambda x: tuple(sorted(set(x)))  # noqa E731
        current_t = tuple()  # type: tuple
        for f in state['t_order'][t]:
            current_t = normalize_tuple(current_t + (f,))
            mean = (state['t_positives'][current_t] / state['t_nsamples'][current_t])
            anchor['feature'].append(f)
            anchor['mean'].append(mean)
            anchor['precision'].append(mean)
            anchor['coverage'].append(state['t_coverage'][current_t])

            # add examples where anchor does or does not hold
            raw_idx = list(state['t_idx'][current_t])
            raw_data = state['raw_data'][raw_idx]
            covered_true = (state['raw_data'][raw_idx][state['labels'][raw_idx] == 1])
            covered_false = (state['raw_data'][raw_idx][state['labels'][raw_idx] == 0])
            exs = {}
            exs['covered'] = matrix_subset(raw_data, 10)
            exs['covered_true'] = matrix_subset(covered_true, 10)
            exs['covered_false'] = matrix_subset(covered_false, 10)
            exs['uncovered_true'] = np.array([])
            exs['uncovered_false'] = np.array([])
            anchor['examples'].append(exs)
        return anchor

    @staticmethod
    def anchor_beam(sample_fn: Callable, delta: float = 0.05, epsilon: float = 0.1, batch_size: int = 10,
                    desired_confidence: float = 1, beam_size: int = 1, verbose: bool = False,
                    epsilon_stop: float = 0.05, min_samples_start: int = 0, max_anchor_size: int = None,
                    verbose_every: int = 1, stop_on_first: bool = False, coverage_samples: int = 10000,
                    data_type: str = None) -> dict:
        """
        Parameters
        ----------
        sample_fn
            Function used to sample from training set which returns (raw) data and labels
        delta
            Used to compute beta
        epsilon
            Precision bound tolerance for convergence
        batch_size
            Number of samples
        desired_confidence
            Desired level of precision, tau in paper
        beam_size
            Beam width
        verbose
            Whether to print intermediate output
        epsilon_stop
            Confidence bound margin around desired precision
        min_samples_start
            Min number of initial samples
        max_anchor_size
            Max number of features in anchor
        verbose_every
            Whether to print intermediate output every verbose_every steps
        stop_on_first
            Stop on first valid anchor found
        coverage_samples
            Number of samples used to compute coverage
        data_type
            Data type for raw data

        Returns
        -------
        Explanation dictionary containing anchors with metadata like coverage and precision.
        """
        # initiate empty anchor dict
        anchor = {'feature': [], 'mean': [], 'precision': [],
                  'coverage': [], 'examples': [], 'all_precision': 0}

        # random (b/c first argument is empty) sample nb of coverage_samples from training data
        _, coverage_data, _ = sample_fn([], coverage_samples, compute_labels=False)

        # sample by default 1 or min_samples_start more random value(s)
        raw_data, data, labels = sample_fn([], max(1, min_samples_start))

        # mean = fraction of labels sampled data that equals the label of the instance to be explained ...
        # ... and is equivalent to prec(A) in paper (eq.2)
        # get lower precision bound lb
        mean = labels.mean()
        beta = np.log(1. / delta)
        lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / data.shape[0])

        # while prec(A) > tau (precision constraint) for A=[] and prec_lb(A) < tau - eps ...
        # ... (lower precision bound below tau with margin eps), keep sampling data until lb is high enough
        while mean > desired_confidence and lb < desired_confidence - epsilon:
            nraw_data, ndata, nlabels = sample_fn([], batch_size)
            data = np.vstack((data, ndata))
            raw_data = np.vstack((raw_data, nraw_data))
            labels = np.hstack((labels, nlabels))
            mean = labels.mean()
            lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / data.shape[0])

        # if prec_lb(A) > tau for A=[] then the empty anchor already satisfies the constraints ...
        # ... and an empty anchor is returned
        if lb > desired_confidence:
            anchor['num_preds'] = data.shape[0]
            anchor['all_precision'] = mean
            return anchor

        # initialize variables
        prealloc_size = batch_size * 10000
        current_idx = data.shape[0]
        data = np.vstack((data, np.zeros((prealloc_size, data.shape[1]), data.dtype)))
        dtype = data_type if data_type is not None else raw_data.dtype
        raw_data = np.vstack((raw_data, np.zeros((prealloc_size, raw_data.shape[1]), dtype=dtype)))
        labels = np.hstack((labels, np.zeros(prealloc_size, labels.dtype)))
        n_features = data.shape[1]
        state = {'t_idx': collections.defaultdict(lambda: set()),
                 't_nsamples': collections.defaultdict(lambda: 0.),
                 't_positives': collections.defaultdict(lambda: 0.),
                 'data': data,
                 'prealloc_size': prealloc_size,
                 'raw_data': raw_data,
                 'labels': labels,
                 'current_idx': current_idx,
                 'n_features': n_features,
                 't_coverage_idx': collections.defaultdict(lambda: set()),
                 't_coverage': collections.defaultdict(lambda: 0.),
                 'coverage_data': coverage_data,
                 't_order': collections.defaultdict(lambda: list())
                 }
        current_size = 1
        best_of_size = {0: []}  # type: Dict[int, list]
        best_coverage = -1
        best_tuple = ()
        if max_anchor_size is None:
            max_anchor_size = n_features

        # find best anchor using beam search until max anchor size
        while current_size <= max_anchor_size:

            # create new candidate anchors by adding features to current best anchors
            tuples = AnchorBaseBeam.make_tuples(best_of_size[current_size - 1], state)

            # goal is to max coverage given precision constraint P(prec(A) > tau) > 1 - delta (eq.4)
            # so keep tuples with higher coverage than current best coverage
            tuples = [x for x in tuples if state['t_coverage'][x] > best_coverage]

            # if no better coverage found with added features -> break
            if len(tuples) == 0:
                break

            # build sample functions for each tuple in tuples list
            # these functions sample randomly for all features except for the ones in the candidate anchors
            # for the features in the anchor it uses the same category (categorical features) or samples from ...
            # ... the same bin (discretized numerical features) as the feature in the observation that is explained
            sample_fns = AnchorBaseBeam.get_sample_fns(sample_fn, tuples, state, data_type=dtype)

            # for each tuple, get initial nb of samples used and prec(A)
            initial_stats = AnchorBaseBeam.get_initial_statistics(tuples, state)

            # apply KL-LUCB and return anchor options (nb of options = beam width)
            # anchor options are in the form of indices of candidate anchors
            chosen_tuples = AnchorBaseBeam.lucb(sample_fns,
                                                initial_stats,
                                                epsilon, delta,
                                                batch_size,
                                                min(beam_size, len(tuples)),
                                                verbose=verbose,
                                                verbose_every=verbose_every)

            # store best anchors for the given anchor size (nb of features in the anchor)
            best_of_size[current_size] = [tuples[x] for x in chosen_tuples]
            if verbose:
                print('Best of size ', current_size, ':')

            # for each candidate anchor:
            # update precision, lower and upper bounds until precision constraints are met
            # update best anchor if coverage is larger than current best coverage
            stop_this = False
            for i, t in zip(chosen_tuples, best_of_size[current_size]):

                # choose at most (beam_size - 1) tuples at each step with at most n_feature steps
                beta = np.log(1. / (delta / (1 + (beam_size - 1) * n_features)))

                # get precision, lower and upper bounds, and coverage for candidate anchor
                mean = state['t_positives'][t] / state['t_nsamples'][t]
                lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / state['t_nsamples'][t])
                ub = AnchorBaseBeam.dup_bernoulli(mean, beta / state['t_nsamples'][t])
                coverage = state['t_coverage'][t]

                if verbose:
                    print(i, mean, lb, ub)

                # while prec(A) >= tau and prec_lb(A) < tau - eps or prec(A) < tau and prec_ub(A) > tau + eps
                # sample more data and update lower and upper precision bounds ...
                # ... b/c respectively either prec_lb(A) or prec(A) needs to improve
                while ((mean >= desired_confidence and lb < desired_confidence - epsilon_stop) or
                       (mean < desired_confidence and ub >= desired_confidence + epsilon_stop)):
                    # sample a batch of data, get new precision, lb and ub values
                    sample_fns[i](batch_size)
                    mean = state['t_positives'][t] / state['t_nsamples'][t]
                    lb = AnchorBaseBeam.dlow_bernoulli(mean, beta / state['t_nsamples'][t])
                    ub = AnchorBaseBeam.dup_bernoulli(mean, beta / state['t_nsamples'][t])

                if verbose:
                    print('%s mean = %.2f lb = %.2f ub = %.2f coverage: %.2f n: %d' %
                          (t, mean, lb, ub, coverage, state['t_nsamples'][t]))

                # if prec(A) > tau and prec_lb(A) > tau - eps then we found an eligible anchor
                if mean >= desired_confidence and lb > desired_confidence - epsilon_stop:

                    if verbose:
                        print('Found eligible anchor ', t, 'Coverage:',
                              coverage, 'Is best?', coverage > best_coverage)

                    # coverage eligible anchor needs to be bigger than current best coverage
                    if coverage > best_coverage:
                        best_coverage = coverage
                        best_tuple = t
                        if best_coverage == 1 or stop_on_first:
                            stop_this = True
            if stop_this:
                break
            current_size += 1

        # if no anchor is found, choose highest precision of best anchor candidate from every round
        if best_tuple == ():
            logger.warning('Could not find an anchor satisfying the {} precision constraint. Now returning '
                           'the best non-eligible anchor.'.format(desired_confidence))
            tuples = []
            for i in range(0, current_size):
                tuples.extend(best_of_size[i])
            sample_fns = AnchorBaseBeam.get_sample_fns(sample_fn, tuples, state, data_type=dtype)
            initial_stats = AnchorBaseBeam.get_initial_statistics(tuples, state)
            chosen_tuples = AnchorBaseBeam.lucb(sample_fns, initial_stats, epsilon,
                                                delta, batch_size, 1, verbose=verbose)
            best_tuple = tuples[chosen_tuples[0]]

        # return explanation dictionary
        return AnchorBaseBeam.get_anchor_from_tuple(best_tuple, state)
