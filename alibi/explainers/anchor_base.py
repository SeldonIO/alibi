import numpy as np
import copy
import collections
import logging
from collections import namedtuple
from operator import truediv
from typing import Callable, NamedTuple, Tuple, Set, Dict, Sequence

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

        # TODO: Add short comments regarding what data is stored here...
        self.state = {'t_idx': collections.defaultdict(lambda: set()),
                      't_nsamples': collections.defaultdict(lambda: 0.),
                      't_positives': collections.defaultdict(lambda: 0.),
                      'data': None,
                      'prealloc_size': None,
                      'raw_data': None,
                      'labels': None,
                      'current_idx': None,
                      'n_features': None,
                      't_coverage_idx': collections.defaultdict(lambda: set()),
                      't_coverage': collections.defaultdict(lambda: 0.),
                      'coverage_data': None,
                      't_order': collections.defaultdict(lambda: list())
                      }

        self.data_type = None  # data type for sampled data

    @staticmethod
    def kl_bernoulli(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Compute KL-divergence between 2 probabilities p and q. len(p) divergences are calculated
        simultaneously.

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

        # TODO: Review casting to float?
        p = np.clip(p, 0.0000001, 0.9999999999999999).astype(float)
        q = np.clip(q, 0.0000001, 0.9999999999999999).astype(float)
        return p * np.log(p / q) + (1. - p) * np.log((1. - p) / (1. - q))

    @staticmethod
    def dup_bernoulli(p: np.ndarray, level: np.ndarray, n_iter: int = 17) -> np.ndarray:
        """
        Update upper precision bound for a candidate anchors dependent on the KL-divergence.

        Parameters
        ----------
        p
            Precision of candidate anchors
        level
            beta / nb of samples for each anchor
        n_iter
            Number of iterations during lower bound update

        Returns
        -------
        Updated upper precision bound
        """
        # TODO: where does 17x sampling come from?
        lm = p
        # TODO: Review this line of code as it diverges from current implementation - what's the point of min(min()) in
        #   orig?
        um = np.clip(p + np.sqrt(level / 2.), 0., 1.0)  # upper bound
        for j in range(1, n_iter):
            qm = (um + lm) / 2.
            kl_gt_idx = AnchorBaseBeam.kl_bernoulli(p, qm) > level
            kl_lt_idx = np.logical_not(kl_gt_idx)
            um[kl_gt_idx] = qm[kl_gt_idx]
            lm[kl_lt_idx] = qm[kl_lt_idx]

        return um

    @staticmethod
    def dlow_bernoulli(p: np.ndarray, level: np.ndarray, n_iter: int = 17) -> np.ndarray:
        """
        Update lower precision bound for a candidate anchors dependent on the KL-divergence.

        Parameters
        ----------
        p
            Precision of candidate anchors
        level
            beta / nb of samples for each anchor
        n_iter
            Number of iterations during lower bound update

        Returns
        -------
        Updated lower precision bound
        """
        um = p
        lm = np.clip(p - np.sqrt(level / 2.), 0., 1.0)  # lower bound
        for _ in range(1, n_iter):
            qm = (um + lm) / 2.
            kl_gt_idx = AnchorBaseBeam.kl_bernoulli(p, qm) > level  # KL-divergence > threshold level
            kl_lt_idx = np.logical_not(kl_gt_idx)
            lm[kl_gt_idx] = qm[kl_gt_idx]
            um[kl_lt_idx] = qm[kl_lt_idx]

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

    def lucb(self, anchors: list, sample_fcn: Callable, initial_stats: dict, epsilon: float, delta: float, batch_size: int, top_n: int,
             verbose: bool = False, verbose_every: int = 1) -> np.ndarray:
        """
        Parameters
        ----------
        anchors:
            A list of anchors from which two critical anchors are selected (see Kaufmann and Kalyanakrishnan, 2013)
        sample_fcn
            A function that returns a sample from the dataset for the specified set of anchors
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

        def update_bounds(means: np.ndarray, ub: np.ndarray, lb: np.ndarray, n_samples: np.ndarray, delta: float,
                          t: int) -> NamedTuple:
            """
            Determines a set of two anchors by updating the upper bound for low emprical precision anchors and
            the lower bound for anchors with high empirical precision.

            Parameters
            ----------
            means
                Empirical mean anchor precisions
            ub
                Upper bound on anchor precisions
            lb
                Lower bound on anchor precisions
            n_samples
                The number of samples drawn for each candidate anchor
            delta
                Confidence budget, candidate anchors have close to optimal precisions with prob. 1 - delta
            t
                Iteration number

            Returns
            -------
            Upper and lower precision bound indices.
            """

            critical_anchors = namedtuple('critical_anchors', 'ut lt')

            sorted_means = np.argsort(means)  # ascending sort of anchor candidates by precision
            beta = AnchorBaseBeam.compute_beta(len(means), t, delta)

            # J = the beam width top anchor candidates with highest precision
            # not_J = the rest
            J = sorted_means[-top_n:]
            not_J = sorted_means[:-top_n]

            # update upper bound for lowest precision anchor candidates
            ub[not_J] = AnchorBaseBeam.dup_bernoulli(means[not_J], beta / n_samples[not_J])
            # update lower bound for highest precision anchor candidates
            lb[J] = AnchorBaseBeam.dup_bernoulli(means[J], beta / n_samples[J])

            # for the low precision anchor candidates, compute the upper precision bound and keep the index ...
            # ... of the anchor candidate with the highest upper precision value -> ut
            # for the high precision anchor candidates, compute the lower precision bound and keep the index ...
            # ... of the anchor candidate with the lowest lower precision value -> lt
            ut = not_J[np.argmax(ub[not_J])]
            lt = J[np.argmin(lb[J])]

            return critical_anchors._make((ut, lt))

        # n_features equals to the nb of candidate anchors
        n_features = len(anchors)

        # initiate arrays for number of samples, positives (sample prediction equals desired label), ...
        # ... upper and lower precision bounds for each anchor candidate
        n_samples = np.array(initial_stats['n_samples'])
        positives = np.array(initial_stats['positives'])
        ub = np.zeros(n_samples.shape)
        lb = np.zeros(n_samples.shape)

        # TODO: It is probably a good idea to sample a larger number of examples here? More accurate precision estimates
        # TODO: Experiment with batching these calls/increasing number of samples after initial benchmark
        # should technically mean that the algo stops quicker? Discuss.*
        for f in np.where(n_samples == 0)[0]:
            n_samples[f] += 1  # set min samples for each anchor candidate to 1
            samples = sample_fcn([f], [1])  # add labels.sum() for the anchor candidate
            positives[f] += samples[0][0]

        if n_features == top_n:  # return all options b/c of beam search width
            return np.arange(n_features)

        # keep updating the upper and lower precision bounds until the difference between the best upper ...
        # ... precision bound of the low precision anchors and the worst lower precision bound of the high ...
        # ... precision anchors is smaller than eps
        means = positives / n_samples  # fraction sample predictions equal to desired label
        t = 1
        critical_a_idx = update_bounds(means, ub, lb, n_samples, delta, t)  # indices of critical arms returned
        B = ub[critical_a_idx.ut] - lb[critical_a_idx.lt]
        verbose_count = 0

        while B > epsilon:
            verbose_count += 1
            if verbose and verbose_count % verbose_every == 0:
                lt, ut = critical_a_idx
                print('Best: %d (mean:%.10f, n: %d, lb:%.4f)' %
                      (lt, means[lt], n_samples[lt], lb[lt]), end=' ')
                print('Worst: %d (mean:%.4f, n: %d, ub:%.4f)' %
                      (ut, means[ut], n_samples[ut], ub[ut]), end=' ')
                print('B = %.2f' % B)

            # TODO: turning num_samples to a list to be able to make further improvements, due to which one would sample
            #  different batch sizes for different anchors
            # TODO: If batch sizes become unequal, then it might be worth to reorder the request so that the smallest
            #  batch is done first and returned, and the programm execution can continue and do other stuff while the
            #  other batch is processed

            # draw samples for each critical anchor, update anchors' mean, upper and lower bound precision estimate
            selected_anchors = [anchors[idx] for idx in critical_a_idx]
            samples = sample_fcn(selected_anchors, [batch_size]*len(critical_a_idx))
            sample_stats = [self.update_state(s, anchor) for (s, anchor) in zip(samples, selected_anchors)]
            pos, total = zip(*sample_stats)
            positives[critical_a_idx] += pos
            n_samples[critical_a_idx] += total
            means[critical_a_idx] = list(map(truediv, pos, total))
            t += 1
            critical_a_idx = update_bounds(means, ub, lb, n_samples, delta, t)
            B = ub[critical_a_idx.ut] - lb[critical_a_idx.lt]
        sorted_means = np.argsort(means)
        return sorted_means[-top_n:]

    @staticmethod
    def propose_anchors(previous_best: list, state: dict) -> list:
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
                pres = data[:, x[0]].nonzero()[0] # Select samples whose feat value is = to the anchor value
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
                if len(new_t) != len(t) + 1:  # Avoid repeating the same feature ...
                    continue
                if new_t not in new_tuples:
                    new_tuples.add(new_t)
                    state['t_order'][new_t] = copy.deepcopy(state['t_order'][t])
                    state['t_order'][new_t].append(f)
                    state['t_coverage_idx'][new_t] = (state['t_coverage_idx'][t].intersection(
                        state['t_coverage_idx'][(f,)]))
                    state['t_coverage'][new_t] = (float(len(state['t_coverage_idx'][new_t])) / coverage_data.shape[0])
                    t_idx = np.array(list(state['t_idx'][t]))  # indices of samples where the len-1 anchor applies
                    t_data = state['data'][t_idx]
                    present = np.where(t_data[:, f] == 1)[0]
                    state['t_idx'][new_t] = set(t_idx[present])  # indices of samples where the proposed anchor applies
                    idx_list = list(state['t_idx'][new_t])
                    state['t_nsamples'][new_t] = float(len(idx_list))
                    state['t_positives'][new_t] = np.sum(state['labels'][idx_list])
        return list(new_tuples)

    def update_state(self, samples: tuple, anchor: tuple) -> Tuple[int, int]:
        """
        Updates the explainer state (see __init__ for full state definition).

        Parameters
        ----------

        samples
            a tuple containing raw_data, discretized data, and an array indicating whether the prediction on the sample
                matches the label of the instance to be explained

        anchor
            a tuple containing the feature indices of the anchor

        Returns
        -------
            a tuple containing the number of instances equals desired label of observation to be explained
                and the total number of instances sampled

        """

        raw_data, data, labels = samples  # data is a binary matrix where 1 indicates that a feature has the
                                          # same value as the feature in the anchor
        n_samples = raw_data.shape[0]

        current_idx = self.state['current_idx']
        idxs = range(current_idx, current_idx + n_samples)
        self.state['t_idx'][anchor].update(idxs)
        self.state['t_nsamples'][anchor] += n_samples
        self.state['t_positives'][anchor] += labels.sum()
        self.state['data'][idxs] = data
        self.state['raw_data'][idxs] = raw_data
        self.state['labels'][idxs] = labels
        self.state['current_idx'] += n_samples

        if self.state['current_idx'] >= self.state['data'].shape[0] - max(1000, n_samples):
            prealloc_size = self.state['prealloc_size']
            self.state['data'] = np.vstack((self.state['data'], np.zeros((prealloc_size, data.shape[1]), data.dtype)))
            self.state['raw_data'] = np.vstack((self.state['raw_data'], np.zeros((prealloc_size, raw_data.shape[1]),
                                                                                 dtype=self.data_type)))
            self.state['labels'] = np.hstack((self.state['labels'], np.zeros(prealloc_size, labels.dtype)))

        return labels.sum(), raw_data.shape[0]

    @staticmethod
    def get_initial_statistics(anchors: list, state: dict) -> dict:
        """
        Parameters
        ----------
        anchors
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
        for anchor in anchors:
            stats['n_samples'].append(state['t_nsamples'][anchor])
            stats['positives'].append(state['t_positives'][anchor])
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
    def to_sample(means: np.ndarray, ubs: np.ndarray, lbs: np.ndarray, desired_confidence: float, epsilon_stop: float):
        """
        Given an array of mean anchor precisions and their upper and lower bounds, determines for which anchors
        more samples need to be drawn in order to estimate the anchors precision with desired_confidence and error
        tolerance

        Parameters
        ----------
            means:
                Mean precisions (each element represents a different anchor)
            ubs:
                Precisions' upper bounds (each element represents a different anchor)
            lbs:
                Precisions' lower bounds (each element represents a different anchor)
            desired_confidence:
                Desired level of confidence for precision estimation
            epsilon_stop:
                Tolerance around desired precision

        Returns
        -------
            A boolean array indicating whether more samples are to be drawn for that particular anchor
        """

        return (means >= desired_confidence) & (lbs < desired_confidence - epsilon_stop) | \
               (means < desired_confidence) & (ubs >= desired_confidence + epsilon_stop)

    def anchor_beam(self, sample_fn: Callable, delta: float = 0.05, epsilon: float = 0.1, batch_size: int = 10,
                    desired_confidence: float = 1., beam_size: int = 1, verbose: bool = False,
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

        # random (b/c first argument is empty) sample nb of coverage_samples from training data
        _, coverage_data, _ = sample_fn([], [coverage_samples], compute_labels=False)

        # sample by default 1 or min_samples_start more random value(s)
        raw_data, data, labels = sample_fn([], [max(1, min_samples_start)])

        # Update object state
        prealloc_size = batch_size * 10000
        dtype = data_type if data_type is not None else raw_data.dtype
        self.state['data'] = np.vstack((data, np.zeros((prealloc_size, data.shape[1]), data.dtype)))
        self.state['current_idx'], self.state['n_features'] = data.shape
        self.state['raw_data'] = np.vstack((raw_data, np.zeros((prealloc_size, raw_data.shape[1]), dtype=dtype)))
        self.state['labels'] = np.hstack((labels, np.zeros(prealloc_size, labels.dtype)))
        self.state['prealloc_size'] = prealloc_size
        self.data_type = dtype

        # mean = fraction of labels sampled data that equals the label of the instance to be explained ...
        # ... and is equivalent to prec(A) in paper (eq.2)
        # get lower precision bound lb
        mean = np.array(labels.mean())
        beta = np.log(1. / delta)
        lb = AnchorBaseBeam.dlow_bernoulli(mean, np.array([beta / data.shape[0]]))

        # while prec(A) > tau (precision constraint) for A=[] and prec_lb(A) < tau - eps ...
        # ... (lower precision bound below tau with margin eps), keep sampling data until lb is high enough
        while mean > desired_confidence and lb < desired_confidence - epsilon:
            nraw_data, ndata, nlabels = sample_fn([], batch_size)
            data = np.vstack((data, ndata))
            raw_data = np.vstack((raw_data, nraw_data))
            labels = np.hstack((labels, nlabels))
            mean = np.array([labels.mean()])
            lb = AnchorBaseBeam.dlow_bernoulli(mean, np.array([beta / data.shape[0]]))

        # if prec_lb(A) > tau for A=[] then the empty anchor already satisfies the constraints ...
        # ... and an empty anchor is returned
        if lb > desired_confidence:
            return {'feature': [],
                    'mean': [],
                    'num_preds': data.shape[0],
                    'precision': [],
                    'coverage': [],
                    'examples': [],
                    'all_precision': mean.item(),
                    }

        current_size, best_coverage = 1, -1
        best_of_size = {0: []}  # type: Dict[int, list]
        best_anchor = ()

        if max_anchor_size is None:
            max_anchor_size = self.state['n_features']

        # find best anchor using beam search until max anchor size
        while current_size <= max_anchor_size:

            # create new candidate anchors by adding features to current best anchors
            anchors = AnchorBaseBeam.propose_anchors(best_of_size[current_size - 1], self.state)

            # goal is to max coverage given precision constraint P(prec(A) > tau) > 1 - delta (eq.4)
            # so keep tuples with higher coverage than current best coverage
            anchors = [anchor for anchor in anchors if self.state['t_coverage'][anchor] > best_coverage]

            # if no better coverage found with added features -> break
            if len(anchors) == 0:
                break

            # for each anchor, get initial nb of samples used and prec(A)
            initial_stats = AnchorBaseBeam.get_initial_statistics(anchors, self.state)

            # apply KL-LUCB and return anchor options (nb of options = beam width)in the form of indices
            candidate_anchors = self.lucb(anchors,
                                          sample_fn,
                                          initial_stats,
                                          epsilon,
                                          delta,
                                          batch_size,
                                          min(beam_size, len(anchors)),
                                          verbose=verbose,
                                          verbose_every=verbose_every,
                                          )
            # store best anchors for the given anchor size (nb of features in the anchor)
            best_of_size[current_size] = [anchors[index] for index in candidate_anchors]

            # for each candidate anchor:
            # update precision, lower and upper bounds until precision constraints are met
            # update best anchor if coverage is larger than current best coverage
            stop_this = False
            sz = candidate_anchors.shape
            kl_constraints, means, coverages = np.zeros(sz), np.zeros(sz), np.zeros(sz)
            beta = np.log(1. / (delta / (1 + (beam_size - 1) * self.state['n_features'])))

            for idx, t in enumerate(best_of_size[current_size]):

                means[idx] = self.state['t_positives'][t] / self.state['t_nsamples'][t]
                kl_constraints[idx] = beta / self.state['t_nsamples'][t]
                coverages[idx] = self.state['t_coverage'][t]

            lbs = AnchorBaseBeam.dlow_bernoulli(means, kl_constraints)
            ubs = AnchorBaseBeam.dup_bernoulli(means, kl_constraints)

            if verbose:
                print('Best of size ', current_size, ':')
                for i, mean, lb, ub in zip(candidate_anchors, means, lbs, ubs):
                    print(i, mean, lb, ub)

            continue_sampling = AnchorBaseBeam.to_sample(means, ubs, lbs, desired_confidence, epsilon_stop)
            remaining_anchors_idx = candidate_anchors[continue_sampling]

            while remaining_anchors_idx.size > 0:
                selected_anchors = [anchors[idx] for idx in remaining_anchors_idx]
                samples = sample_fn(selected_anchors, [batch_size]*len(selected_anchors))
                # TODO: Can we do better than to wait and update the state sequentially if sampling in parallel?
                for (s, anchor, anchor_idx) in zip(samples, selected_anchors, remaining_anchors_idx):
                    self.update_state(s, anchor)
                    means[anchor_idx] = self.state['t_positives'][anchor] / self.state['t_nsamples'][anchor]
                    kl_constraints[anchor_idx] = beta / self.state['t_nsamples'][anchor]

                lbs[continue_sampling] = AnchorBaseBeam.dlow_bernoulli(means[continue_sampling],
                                                                       kl_constraints[continue_sampling])
                ubs[continue_sampling] = AnchorBaseBeam.dup_bernoulli(means[continue_sampling],
                                                                      kl_constraints[continue_sampling])
                continue_sampling = AnchorBaseBeam.to_sample(means, ubs, lbs, desired_confidence, epsilon_stop)
                remaining_anchors_idx = candidate_anchors[continue_sampling]

            if verbose:
                for i, mean, lb, ub, coverage in zip(candidate_anchors, means, lbs, ubs, coverages):
                    t = anchors[i]
                    print('%s mean = %.2f lb = %.2f ub = %.2f coverage: %.2f n: %d' %
                          (t, mean, lb, ub, coverage, self.state['t_nsamples'][t]))

            # find anchors who meet the precision setting and have better coverage than the best anchors so far
            valid_anchors = (means >= desired_confidence) & (lb > desired_confidence - epsilon_stop)
            better_anchors = (valid_anchors & (coverages > best_coverage)).nonzero()[0]

            if verbose:
                for idx, coverage in zip(candidate_anchors[valid_anchors], coverages[valid_anchors]):
                    t = anchors[idx]
                    print('Found eligible anchor ', t, 'Coverage:', coverage, 'Is best?', coverage > best_coverage)

            if better_anchors.size > 0:
                best_anchor_idx = better_anchors[np.argmax(coverages[better_anchors])]
                best_coverage = coverages[best_anchor_idx]
                best_anchor = anchors[candidate_anchors[best_anchor_idx]]
                if best_coverage == 1. or stop_on_first:
                    stop_this = True

            if stop_this:
                break
            current_size += 1

        # if no anchor is found, choose highest precision of best anchor candidate from every round
        if best_anchor == ():
            logger.warning('Could not find an anchor satisfying the {} precision constraint. Now returning '
                           'the best non-eligible anchor.'.format(desired_confidence))
            anchors = []
            for i in range(0, current_size):
                anchors.extend(best_of_size[i])
            initial_stats = AnchorBaseBeam.get_initial_statistics(anchors, self.state)
            candidate_anchors = self.lucb(anchors, sample_fn, initial_stats, epsilon, delta, batch_size, 1,
                                          verbose=verbose)
            best_anchor = anchors[candidate_anchors[0]]

        # return explanation dictionary
        return AnchorBaseBeam.get_anchor_from_tuple(best_anchor, self.state)

        # TODO: Discuss logging strategy
