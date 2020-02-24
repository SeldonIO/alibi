import copy
import logging
import numpy as np
from collections import defaultdict, namedtuple
from functools import partial
from typing import Callable, Tuple, Set, Dict, List

from alibi.utils.distributed import ActorPool, RAY_INSTALLED
from alibi.utils.distributions import kl_bernoulli


logger = logging.getLogger(__name__)


# TODO: Discuss logging strategy

class AnchorBaseBeam:

    def __init__(self, samplers: List[Callable], **kwargs) -> None:
        """
        Parameters
        ---------
        samplers
            Objects that can be called with args (result, n_samples) tuple to draw samples.
        """

        self.sample_fcn = samplers[0]
        self.samplers = None  # type: List[Callable]
        # Initial size (in batches) of data/raw data samples cache.
        self.sample_cache_size = kwargs.get('sample_cache_size', 10000)
        # when only the max of self.margin or batch size remain emptpy, the cache is
        # extended to accommodate an additional sample_cache_size batches.
        self.margin = kwargs.get('cache_margin', 1000)

    def _init_state(self, batch_size: int, coverage_data: np.ndarray) -> None:
        """
        Initialises the object state, which is used to compute result precisions & precision bounds
        and provide metadata for explanation objects.

        Parameters
        ----------
        batch_size
            See anchor_beam method.
        coverage_data
            See _get_coverage_samples method.
        sample_cache_size
            See anchor_beam method.
        """

        prealloc_size = batch_size * self.sample_cache_size
        # t_ indicates that the attribute is a dictionary with entries for each anchor
        self.state = {
            't_coverage': defaultdict(lambda: 0.),   # anchors' coverage
            't_coverage_idx': defaultdict(set),      # index of anchors in coverage set
            't_covered_true': defaultdict(None),     # samples with same pred as instance where t_ applies
            't_covered_false': defaultdict(None),    # samples with dif pred to instance where t_ applies
            't_idx': defaultdict(set),               # row idx in sample cache where the anchors apply
            't_nsamples': defaultdict(lambda: 0.),   # total number of samples drawn for the anchors
            't_order': defaultdict(list),            # anchors are sorted to avoid exploring permutations
                                                     # this is the order in which anchors were found
            't_positives': defaultdict(lambda: 0.),  # nb of samples where result pred = pred on instance
            'prealloc_size': prealloc_size,          # samples caches size
            'data': np.zeros((prealloc_size, coverage_data.shape[1]), coverage_data.dtype),  # samples caches
            'labels': np.zeros(prealloc_size, ),     # clf pred labels on raw_data
            'current_idx': 0,
            'n_features': coverage_data.shape[1],    # data set dim after encoding
            'coverage_data': coverage_data,          # coverage data
        }  # type: dict
        self.state['t_order'][()] = ()  # Trivial order for the empty result

    @staticmethod
    def _sort(x: tuple, allow_duplicates=False) -> tuple:
        """
        Sorts a tuple, optionally removing duplicates.

        Parameters
        ----------
        x:
            Tuple to be sorted.
        allow_duplicates:
            If True, duplicate entries are kept

        Returns
        -------
            A sorted tuple.
        """

        if allow_duplicates:
            return tuple(sorted(x))

        return tuple(sorted(set(x)))

    @staticmethod
    def dup_bernoulli(p: np.ndarray, level: np.ndarray, n_iter: int = 17) -> np.ndarray:
        """
        Update upper precision bound for a candidate anchors dependent on the KL-divergence.

        Parameters
        ----------
        p
            Precision of candidate anchors.
        level
            beta / nb of samples for each result.
        n_iter
            Number of iterations during lower bound update.

        Returns
        -------
            Updated upper precision bounds array.
        """
        # TODO: where does 17x sampling come from?
        lm = p.copy()
        um = np.minimum(np.minimum(p + np.sqrt(level / 2.), 1.0), 1.0)

        # Perform bisection algorithm to find the largest qm s.t. kl divergence is > level
        for j in range(1, n_iter):
            qm = (um + lm) / 2.
            kl_gt_idx = kl_bernoulli(p, qm) > level
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
            Precision of candidate anchors.
        level
            beta / nb of samples for each result.
        n_iter
            Number of iterations during lower bound update.

        Returns
        -------
            Updated lower precision bounds array.
        """

        um = p.copy()
        lm = np.clip(p - np.sqrt(level / 2.), 0.0, 1.0)  # lower bound

        # Perform bisection algorithm to find the smallest qm s.t. kl divergence is > level
        for _ in range(1, n_iter):
            qm = (um + lm) / 2.
            kl_gt_idx = kl_bernoulli(p, qm) > level
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
            Number of candidate anchors.
        t
            Iteration number.
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

    def _get_coverage_samples(self, coverage_samples: int, samplers: List[Callable] = None) -> np.ndarray:
        """
        Draws samples uniformly at random from the training set.

        Parameters
        ---------
        coverage_samples
            See anchor_beam method.
        samplers
            See __init__ method.

        Returns
        -------
        coverage_data
            binarised samples, where 1 indicates the feature has same value/is in same beam as
            instance to be explained. Used to determine, e.g., which samples an result applies to.
        """

        [coverage_data] = self.sample_fcn((0, ()), coverage_samples, compute_labels=False)

        return coverage_data

    def select_critical_arms(self, means: np.ndarray, ub: np.ndarray, lb: np.ndarray, n_samples: np.ndarray,
                             delta: float, top_n: int, t: int):  # type: ignore
        """
        Determines a set of two anchors by updating the upper bound for low emprical precision anchors and
        the lower bound for anchors with high empirical precision.

        Parameters
        ----------
        means
            Empirical mean result precisions.
        ub
            Upper bound on result precisions.
        lb
            Lower bound on result precisions.
        n_samples
            The number of samples drawn for each candidate result.
        delta
            Confidence budget, candidate anchors have close to optimal precisions with prob. 1 - delta.
        top_n
            Number of arms to be selected.
        t
            Iteration number.

        Returns
        -------
        Upper and lower precision bound indices.
        """

        crit_arms = namedtuple('crit_arms', ['ut', 'lt'])

        sorted_means = np.argsort(means)  # ascending sort of result candidates by precision
        beta = self.compute_beta(len(means), t, delta)

        # J = the beam width top result candidates with highest precision
        # not_J = the rest
        J = sorted_means[-top_n:]
        not_J = sorted_means[:-top_n]

        # update upper bound for lowest precision result candidates
        ub[not_J] = self.dup_bernoulli(means[not_J], beta / n_samples[not_J])
        # update lower bound for highest precision result candidates
        lb[J] = self.dlow_bernoulli(means[J], beta / n_samples[J])

        # for the low precision result candidates, compute the upper precision bound and keep the index ...
        # ... of the result candidate with the highest upper precision value -> ut
        # for the high precision result candidates, compute the lower precision bound and keep the index ...
        # ... of the result candidate with the lowest lower precision value -> lt
        ut = not_J[np.argmax(ub[not_J])]
        lt = J[np.argmin(lb[J])]

        return crit_arms._make((ut, lt))

    def kllucb(self, anchors: list, init_stats: dict, epsilon: float, delta: float, batch_size: int, top_n: int,
               verbose: bool = False, verbose_every: int = 1) -> np.ndarray:
        """
        Implements the KL-LUCB algorithm (Kaufmann and Kalyanakrishnan, 2013).

        Parameters
        ----------
        anchors:
            A list of anchors from which two critical anchors are selected (see Kaufmann and Kalyanakrishnan, 2013).
        init_stats
            Dictionary with lists containing nb of samples used and where sample predictions equal the desired label.
        epsilon
            Precision bound tolerance for convergence.
        delta
            Used to compute beta.
        batch_size
            Number of samples.
        top_n
            Min of beam width size or number of candidate anchors.
        verbose
            Whether to print intermediate output.
        verbose_every
            Whether to print intermediate output every verbose_every steps.

        Returns
        -------
            Indices of best result options. Number of indices equals min of beam width or nb of candidate anchors.
        """

        # n_features equals to the nb of candidate anchors
        n_features = len(anchors)

        # arrays for total number of samples & positives (# samples where prediction equals desired label)
        n_samples, positives = init_stats['n_samples'], init_stats['positives']
        anchors_to_sample, anchors_idx = [], []
        for f in np.where(n_samples == 0)[0]:
            anchors_to_sample.append(anchors[f])
            anchors_idx.append(f)

        if anchors_idx:
            pos, total = self.draw_samples(anchors_to_sample, 1)
            positives[anchors_idx] += pos
            n_samples[anchors_idx] += total

        if n_features == top_n:  # return all options b/c of beam search width
            return np.arange(n_features)

        # update the upper and lower precision bounds until the difference between the best upper ...
        # ... precision bound of the low precision anchors and the worst lower precision bound of the high ...
        # ... precision anchors is smaller than eps
        means = positives / n_samples  # fraction sample predictions equal to desired label
        ub, lb = np.zeros(n_samples.shape), np.zeros(n_samples.shape)
        t = 1
        crit_a_idx = self.select_critical_arms(means, ub, lb, n_samples, delta, top_n, t)
        B = ub[crit_a_idx.ut] - lb[crit_a_idx.lt]
        verbose_count = 0

        while B > epsilon:

            verbose_count += 1
            if verbose and verbose_count % verbose_every == 0:
                ut, lt = crit_a_idx
                print('Best: %d (mean:%.10f, n: %d, lb:%.4f)' %
                      (lt, means[lt], n_samples[lt], lb[lt]), end=' ')
                print('Worst: %d (mean:%.4f, n: %d, ub:%.4f)' %
                      (ut, means[ut], n_samples[ut], ub[ut]), end=' ')
                print('B = %.2f' % B)

            # draw samples for each critical result, update anchors' mean, upper and lower
            # bound precision estimate
            selected_anchors = [anchors[idx] for idx in crit_a_idx]
            pos, total = self.draw_samples(selected_anchors, batch_size)
            idx = list(crit_a_idx)
            positives[idx] += pos
            n_samples[idx] += total
            means = positives / n_samples
            t += 1
            crit_a_idx = self.select_critical_arms(means, ub, lb, n_samples, delta, top_n, t)
            B = ub[crit_a_idx.ut] - lb[crit_a_idx.lt]
        sorted_means = np.argsort(means)

        return sorted_means[-top_n:]

    def draw_samples(self, anchors: list, batch_size: int) -> Tuple[tuple, tuple]:
        """
        Parameters
        ----------
            anchors
                Anchors on which samples are conditioned.
            batch_size
                The number of samples drawn for each result.

        Returns
        -------
            A tuple of positive samples (for which prediction matches desired label)
            and a tuple of total number of samples drawn.
        """

        for anchor in anchors:
            if anchor not in self.state['t_order']:
                self.state['t_order'][anchor] = list(anchor)

        sample_stats, pos, total = [], (), ()  # type: List, Tuple, Tuple
        samples_iter = [self.sample_fcn((i, tuple(self.state['t_order'][anchor])), num_samples=batch_size)
                        for i, anchor in enumerate(anchors)]
        for samples, anchor in zip(samples_iter, anchors):
            covered_true, covered_false, labels, *additionals, _ = samples
            sample_stats.append(self.update_state(covered_true, covered_false, labels, additionals, anchor))
            pos, total = list(zip(*sample_stats))

        return pos, total

    def propose_anchors(self, previous_best: list) -> list:
        """
        Parameters
        ----------
        previous_best
            List with tuples of result candidates.


        Returns
        -------
            List with tuples of candidate anchors with additional metadata.
        """

        # compute some variables used later on
        state = self.state
        all_features = range(state['n_features'])
        coverage_data = state['coverage_data']
        current_idx = state['current_idx']
        data = state['data'][:current_idx]
        labels = state['labels'][:current_idx]

        # initially, every feature separately is an result
        if len(previous_best) == 0:
            tuples = [(x,) for x in all_features]
            for x in tuples:
                pres = data[:, x[0]].nonzero()[0]  # Select samples whose feat value is = to the result value
                state['t_idx'][x] = set(pres)
                state['t_nsamples'][x] = float(len(pres))
                state['t_positives'][x] = float(labels[pres].sum())
                state['t_order'][x].append(x[0])
                state['t_coverage_idx'][x] = set(coverage_data[:, x[0]].nonzero()[0])
                state['t_coverage'][x] = (float(len(state['t_coverage_idx'][x])) / coverage_data.shape[0])
            return tuples

        # create new anchors: add a feature to every result in current best
        new_tuples = set()  # type: Set[tuple]
        for f in all_features:
            for t in previous_best:
                new_t = self._sort(t + (f,), allow_duplicates=False)
                if len(new_t) != len(t) + 1:  # Avoid repeating the same feature ...
                    continue
                if new_t not in new_tuples:
                    new_tuples.add(new_t)
                    state['t_order'][new_t] = copy.deepcopy(state['t_order'][t])
                    state['t_order'][new_t].append(f)
                    state['t_coverage_idx'][new_t] = (state['t_coverage_idx'][t].intersection(
                        state['t_coverage_idx'][(f,)])
                    )
                    state['t_coverage'][new_t] = (float(len(state['t_coverage_idx'][new_t])) / coverage_data.shape[0])
                    t_idx = np.array(list(state['t_idx'][t]))  # indices of samples where the len-1 result applies
                    t_data = state['data'][t_idx]
                    present = np.where(t_data[:, f] == 1)[0]
                    state['t_idx'][new_t] = set(t_idx[present])  # indices of samples where the proposed result applies
                    idx_list = list(state['t_idx'][new_t])
                    state['t_nsamples'][new_t] = float(len(idx_list))
                    state['t_positives'][new_t] = np.sum(state['labels'][idx_list])

        return list(new_tuples)

    def update_state(self, covered_true: np.ndarray, covered_false: np.ndarray, labels: np.ndarray,
                     samples: tuple, anchor: tuple) -> Tuple[int, int]:
        """
        Updates the explainer state (see __init__ for full state definition).

        Parameters
        ----------

        covered_true
            Examples where the result applies and the prediction is the same as on
            the instance to be explained.
        covered_false
            Examples where the result applies and the prediction is the different to
            the instance to be explained.
        samples
            A tuple containing discretized data, coverage and the result sampled.
        labels
            An array indicating whether the prediction on the sample matches the label
            of the instance to be explained.
        anchor
            The result to be updated.

        Returns
        -------
            A tuple containing the number of instances equals desired label of observation
            to be explained the total number of instances sampled, and the result that was sampled
        """

        # data = binary matrix where 1 means a feature has the same value as the feature in the result
        data, coverage = samples
        n_samples = data.shape[0]

        current_idx = self.state['current_idx']
        idxs = range(current_idx, current_idx + n_samples)
        self.state['t_idx'][anchor].update(idxs)
        self.state['t_nsamples'][anchor] += n_samples
        self.state['t_positives'][anchor] += labels.sum()
        if coverage > -1:
            self.state['t_coverage'][anchor] = coverage
        self.state['t_covered_true'][anchor] = covered_true
        self.state['t_covered_false'][anchor] = covered_false
        self.state['data'][idxs] = data
        self.state['labels'][idxs] = labels
        self.state['current_idx'] += n_samples

        if self.state['current_idx'] >= self.state['data'].shape[0] - max(self.margin, n_samples):
            prealloc_size = self.state['prealloc_size']
            self.state['data'] = np.vstack(
                (self.state['data'], np.zeros((prealloc_size, data.shape[1]), data.dtype))
            )
            self.state['labels'] = np.hstack(
                (self.state['labels'], np.zeros(prealloc_size, labels.dtype))
            )

        return labels.sum(), data.shape[0]

    def get_init_stats(self, anchors: list, coverages=False) -> dict:
        """
        Finds the number of samples already drawn for each result in anchors, their
        comparisons with the instance to be explained and, optionally, coverage.

        Parameters
        ----------
        anchors
            Candidate anchors.
        coverages
            If True, the statistics returned contain the coverage of the specified anchors.

        Returns
        -------
            Dictionary with lists containing nb of samples used and where sample predictions equal
            the desired label.
        """

        def array_factory(size: tuple):
            return lambda: np.zeros(size)

        state = self.state
        stats = defaultdict(array_factory((len(anchors),)))  # type: Dict[str, np.ndarray]
        for i, anchor in enumerate(anchors):
            stats['n_samples'][i] = state['t_nsamples'][anchor]
            stats['positives'][i] = state['t_positives'][anchor]
            if coverages:
                stats['coverages'][i] = state['t_coverage'][anchor]

        return stats

    def get_anchor_metadata(self, features: tuple, success, batch_size: int = 100) -> dict:
        """
        Given the features contained in a result, it retrieves metadata such as the precision and
        coverage of the result and partial anchors and examples where the result/partial anchors
        apply and yield the same prediction as on the instance to be explained (covered_true)
        or a different prediction (covered_false).

        Parameters
        ----------
        features
            Sorted indices of features in result.
        success
            Indicates whether an anchor satisfying precision threshold was met or not.
        batch_size
            Number of samples among which positive and negative examples for partial anchors are
            selected if partial anchors have not already been explicitly sampled.

        Returns
        -------
            Anchor dictionary with result features and additional metadata.
            :param success:
        """

        state = self.state
        anchor = {'feature': [], 'mean': [], 'precision': [], 'coverage': [], 'examples': [],
                  'all_precision': 0, 'num_preds': state['data'].shape[0], 'success': success}  # type: dict
        current_t = tuple()  # type: tuple
        # draw pos and negative example where partial result applies if not sampled during search
        to_resample, to_resample_idx = [], []
        for f in state['t_order'][features]:
            current_t = self._sort(current_t + (f,), allow_duplicates=False)
            mean = (state['t_positives'][current_t] / state['t_nsamples'][current_t])
            anchor['feature'].append(f)
            anchor['mean'].append(mean)
            anchor['precision'].append(mean)
            anchor['coverage'].append(state['t_coverage'][current_t])

            # add examples where result does or does not hold
            if current_t in state['t_covered_true']:
                exs = {
                    'covered_true': state['t_covered_true'][current_t],
                    'covered_false': state['t_covered_false'][current_t],
                    'uncovered_true': np.array([]),
                    'uncovered_false': np.array([]),
                }
                anchor['examples'].append(exs)
            else:
                to_resample.append(current_t)
                # sampling process relies on ordering
                state['t_order'][current_t] = list(current_t)
                to_resample_idx.append(len(anchor['examples']))
                anchor['examples'].append('placeholder')
                # if the anchor was not sampled, the coverage is not estimated
                anchor['coverage'][-1] = 'placeholder'

        # If partial anchors have not been sampled, resample to find examples
        if to_resample:

            _, _ = self.draw_samples(to_resample, batch_size)

            while to_resample:
                feats, example_idx = to_resample.pop(), to_resample_idx.pop()
                anchor['examples'][example_idx] = {
                    'covered_true': state['t_covered_true'][feats],
                    'covered_false': state['t_covered_false'][feats],
                    'uncovered_true': np.array([]),
                    'uncovered_false': np.array([]),
                }
                # update result with true coverage
                anchor['coverage'][example_idx] = state['t_coverage'][feats]

        return anchor

    @staticmethod
    def to_sample(means: np.ndarray, ubs: np.ndarray, lbs: np.ndarray, desired_confidence: float, epsilon_stop: float):
        """
        Given an array of mean result precisions and their upper and lower bounds, determines for which anchors
        more samples need to be drawn in order to estimate the anchors precision with desired_confidence and error
        tolerance.

        Parameters
        ----------
            means:
                Mean precisions (each element represents a different result).
            ubs:
                Precisions' upper bounds (each element represents a different result).
            lbs:
                Precisions' lower bounds (each element represents a different result).
            desired_confidence:
                Desired level of confidence for precision estimation.
            epsilon_stop:
                Tolerance around desired precision.

        Returns
        -------
            Boolean array indicating whether more samples are to be drawn for that particular result.
        """

        return ((means >= desired_confidence) & (lbs < desired_confidence - epsilon_stop)) | \
               ((means < desired_confidence) & (ubs >= desired_confidence + epsilon_stop))

    def anchor_beam(self, delta: float = 0.05, epsilon: float = 0.1, desired_confidence: float = 1.,
                    beam_size: int = 1, epsilon_stop: float = 0.05, min_samples_start: int = 100,
                    max_anchor_size: int = None, stop_on_first: bool = False, batch_size: int = 100,
                    coverage_samples: int = 10000,  verbose: bool = False, verbose_every: int = 1,
                    **kwargs) -> dict:

        """
        Uses the KL-LUCB algorithm (Kaufmann and Kalyanakrishnan, 2013) together with additional sampling to search
        feature sets (anchors) that guarantee the prediction made by a classifier model. The search is greedy if
        beam_size=1. Otherwise, at each of the max_anchor_size steps, beam_size solutions are explored. By construction,
        solutions found have high precision (defined as the expected of number of times the classifier makes the same
        prediction when queried with the feature subset combined with arbitrary samples drawn from a noise distribution)
        The algorithm maximises the coverage of the solution found - the frequency of occurrence of records containing
        the feature subset in set of samples.

        Parameters
        ----------
        delta
            Used to compute beta.
        epsilon
            Precision bound tolerance for convergence.
        desired_confidence
            Desired level of precision (tau in paper).
        beam_size
            Beam width.
        epsilon_stop
            Confidence bound margin around desired precision.
        min_samples_start
            Min number of initial samples.
        max_anchor_size
            Max number of features in result.
        stop_on_first
            Stop on first valid result found.
        coverage_samples
            Number of samples from which to build a coverage set.
        batch_size
            Number of samples used for an arm evaluation.
        verbose
            Whether to print intermediate LUCB & anchor selection output.
        verbose_every
            Print intermediate output every verbose_every steps.

        Returns
        -------
            Explanation dictionary containing anchors with metadata like coverage and precision
            and examples.
        """

        # Select coverage set and initialise object state
        coverage_data = self._get_coverage_samples(
            coverage_samples,
            samplers=self.samplers,
        )
        self._init_state(batch_size, coverage_data)

        # sample by default 1 or min_samples_start more random value(s)
        (pos,), (total,) = self.draw_samples([()], min_samples_start)

        # mean = fraction of labels sampled data that equals the label of the instance to be explained, ...
        # ... equivalent to prec(A) in paper (eq.2)
        mean = np.array([pos / total])
        beta = np.log(1. / delta)
        # lower bound on mean precision
        lb = self.dlow_bernoulli(mean, np.array([beta / total]))

        # if lower precision bound below tau with margin eps, keep sampling data until lb is high enough ...
        # or mean falls below precision threshold
        while mean > desired_confidence and lb < desired_confidence - epsilon:
            (n_pos,), (n_total,) = self.draw_samples([()], batch_size)
            pos += n_pos
            total += n_total
            mean = np.array([pos / total])
            lb = self.dlow_bernoulli(mean, np.array([beta / total]))

        # if prec_lb(A) > tau for A=() then the empty result satisfies the constraints ...
        if lb > desired_confidence:
            return {
                'feature': [],
                'mean': [],
                'num_preds': total,
                'precision': [],
                'coverage': [],
                'examples': [],
                'all_precision': mean,
                'success': True,
            }

        current_size, best_coverage = 1, -1
        best_of_size = {0: []}  # type: Dict[int, list]
        best_anchor = ()

        if max_anchor_size is None:
            max_anchor_size = self.state['n_features']

        # find best result using beam search
        while current_size <= max_anchor_size:

            # create new candidate anchors by adding features to current best anchors
            anchors = self.propose_anchors(best_of_size[current_size - 1])
            # goal is to max coverage given precision constraint P(prec(A) > tau) > 1 - delta (eq.4)
            # so keep tuples with higher coverage than current best coverage
            anchors = [anchor for anchor in anchors if self.state['t_coverage'][anchor] > best_coverage]

            # if no better coverage found with added features -> break
            if len(anchors) == 0:
                break

            # for each result, get initial nb of samples used and prec(A)
            stats = self.get_init_stats(anchors)

            # apply KL-LUCB and return result options (nb of options = beam width) in the form of indices
            candidate_anchors = self.kllucb(
                anchors,
                stats,
                epsilon,
                delta,
                batch_size,
                min(beam_size, len(anchors)),
                verbose=verbose,
                verbose_every=verbose_every,
            )
            # store best anchors for the given result size (nb of features in the result)
            best_of_size[current_size] = [anchors[index] for index in candidate_anchors]
            # for each candidate result:
            #   update precision, lower and upper bounds until precision constraints are met
            #   update best result if coverage is larger than current best coverage
            stats = self.get_init_stats(best_of_size[current_size], coverages=True)
            positives, n_samples = stats['positives'], stats['n_samples']
            beta = np.log(1. / (delta / (1 + (beam_size - 1) * self.state['n_features'])))
            kl_constraints = beta / n_samples
            means = stats['positives'] / stats['n_samples']
            lbs = self.dlow_bernoulli(means, kl_constraints)
            ubs = self.dup_bernoulli(means, kl_constraints)

            if verbose:
                print('Best of size ', current_size, ':')
                for i, mean, lb, ub in zip(candidate_anchors, means, lbs, ubs):
                    print(i, mean, lb, ub)

            # draw samples to ensure result meets precision criteria
            continue_sampling = self.to_sample(means, ubs, lbs, desired_confidence, epsilon_stop)
            while continue_sampling.any():
                selected_anchors = [anchors[idx] for idx in candidate_anchors[continue_sampling]]
                pos, total = self.draw_samples(selected_anchors, batch_size)
                positives[continue_sampling] += pos
                n_samples[continue_sampling] += total
                means[continue_sampling] = positives[continue_sampling]/n_samples[continue_sampling]
                kl_constraints[continue_sampling] = beta / n_samples[continue_sampling]
                lbs[continue_sampling] = self.dlow_bernoulli(
                    means[continue_sampling],
                    kl_constraints[continue_sampling],
                )
                ubs[continue_sampling] = self.dup_bernoulli(
                    means[continue_sampling],
                    kl_constraints[continue_sampling],
                )
                continue_sampling = self.to_sample(means, ubs, lbs, desired_confidence, epsilon_stop)

            # anchors who meet the precision setting and have better coverage than the best anchors so far
            coverages = stats['coverages']
            valid_anchors = (means >= desired_confidence) & (lbs > desired_confidence - epsilon_stop)
            better_anchors = (valid_anchors & (coverages > best_coverage)).nonzero()[0]

            if verbose:
                for i, valid, mean, lb, ub, coverage in \
                        zip(candidate_anchors, valid_anchors,  means, lbs, ubs, coverages):
                    t = anchors[i]
                    print(
                        '%s mean = %.2f lb = %.2f ub = %.2f coverage: %.2f n: %d' %
                        (t, mean, lb, ub, coverage, self.state['t_nsamples'][t]))
                    if valid:
                        print(
                            'Found eligible result ', t,
                            'Coverage:', coverage,
                            'Is best?', coverage > best_coverage,
                        )

            if better_anchors.size > 0:
                best_anchor_idx = better_anchors[np.argmax(coverages[better_anchors])]
                best_coverage = coverages[best_anchor_idx]
                best_anchor = anchors[candidate_anchors[best_anchor_idx]]
                if best_coverage == 1. or stop_on_first:
                    break

            current_size += 1

        # if no result is found, choose highest precision of best result candidate from every round
        if not best_anchor:
            success = False  # indicates the method has not found an anchor
            logger.warning('Could not find an result satisfying the {} precision constraint. Now returning '
                           'the best non-eligible result.'.format(desired_confidence))
            anchors = []
            for i in range(0, current_size):
                anchors.extend(best_of_size[i])
            stats = self.get_init_stats(anchors)
            candidate_anchors = self.kllucb(
                anchors,
                stats,
                epsilon,
                delta,
                batch_size,
                1,  # beam size
                verbose=verbose,
            )
            best_anchor = anchors[candidate_anchors[0]]
        else:
            success = True

        return self.get_anchor_metadata(best_anchor, success, batch_size=batch_size)


class DistributedAnchorBaseBeam(AnchorBaseBeam):

    if RAY_INSTALLED:
        import ray
        ray = ray

    def __init__(self, samplers: List[Callable], **kwargs) -> None:

        super().__init__(samplers)
        if 'chunksize' in kwargs:
            self.chunksize = kwargs['chunksize']
        else:
            self.chunksize = 1
        self.sample_fcn = lambda actor, anchor, n_samples, compute_labels=True:\
            actor.__call__.remote(anchor,
                                  n_samples,
                                  compute_labels=compute_labels)
        self.pool = ActorPool(samplers)
        self.samplers = samplers

    def _get_coverage_samples(self, coverage_samples: int, samplers: List[Callable] = None) -> np.ndarray:
        """
        Sends a request for a coverage set to process running sampling tasks.

        Parameters
        ----------
            See superclass implementation.

        Returns
        -------
            See superclass implementation.
        """

        [coverage_data] = DistributedAnchorBaseBeam.ray.get(
            self.sample_fcn(samplers[0], (0, ()), coverage_samples, compute_labels=False)
            )

        return coverage_data

    def draw_samples(self, anchors: list, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Distributes sampling requests among processes running sampling tasks.

        Parameters
        ----------
            See superclass  implementation.

        Returns
        -------
            Same outputs as superclass but of different types.
        """

        # partial anchors not generated by propose_anchors are not in the order dictionary
        for anchor in anchors:
            if anchor not in self.state['t_order']:
                self.state['t_order'][anchor] = list(anchor)

        pos, total = np.zeros((len(anchors),)), np.zeros((len(anchors),))
        order_map = [(i, tuple(self.state['t_order'][anchor])) for i, anchor in enumerate(anchors)]
        samples_iter = self.pool.map_unordered(
            partial(self.sample_fcn, n_samples=batch_size),
            order_map,
            self.chunksize,
        )
        for samples_batch in samples_iter:
            for samples in samples_batch:
                covered_true, covered_false, labels, *additionals, anchor_idx = samples
                positives, n_samples = self.update_state(
                    covered_true,
                    covered_false,
                    labels,
                    additionals,
                    anchors[anchor_idx],
                )
                # return statistics in the same order as the requests
                pos[anchor_idx], total[anchor_idx] = positives, n_samples

        return pos, total
