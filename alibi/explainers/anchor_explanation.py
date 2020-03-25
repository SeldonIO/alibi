import numpy as np


class AnchorExplanation:

    def __init__(self, exp_type: str, exp_map: dict) -> None:
        """
        Class used to unpack the anchors and metadata from the explainer dictionary.

        Parameters
        ----------
        exp_type
            Type of explainer: tabular, text or image
        exp_map
            Dictionary with the anchors and explainer metadata for an observation
        """
        self.type = exp_type
        self.exp_map = exp_map

    def names(self, partial_index: int = None) -> list:
        """
        Parameters
        ----------
        partial_index
            Get the result until a certain index.
            For example, if the result is (A=1,B=2,C=2) and partial_index=1, this will return ["A=1", "B=2"].

        Returns
        -------
        names
            Names with the result conditions
        """
        names = self.exp_map['names']
        if partial_index is not None:
            names = names[:partial_index + 1]
        return names

    def features(self, partial_index: int = None) -> list:
        """
        Parameters
        ----------
        partial_index
            Get the result until a certain index.
            For example, if the result uses segment_labels (1, 2, 3) and partial_index=1, this will return [1, 2].

        Returns
        -------
        segment_labels
            Features used in the result conditions.
        """
        features = self.exp_map['feature']
        if partial_index is not None:
            features = features[:partial_index + 1]
        return features

    def precision(self, partial_index: int = None) -> float:
        """
        Parameters
        ----------
        partial_index
            Get the result precision until a certain index.
            For example, if the result has precisions [0.1, 0.5, 0.95] and partial_index=1, this will return 0.5.

        Returns
        -------
        precision
            Anchor precision
        """
        precision = self.exp_map['precision']
        if len(precision) == 0:
            return self.exp_map['all_precision']
        if partial_index is not None:
            return precision[partial_index]
        else:
            return precision[-1]

    def coverage(self, partial_index: int = None) -> float:
        """
        Parameters
        ----------
        partial_index
            Get the result coverage until a certain index.
            For example, if the result has precisions [0.1, 0.5, 0.95] and partial_index=1, this will return 0.5.

        Returns
        -------
        coverage
            Anchor coverage
        """
        coverage = self.exp_map['coverage']
        if len(coverage) == 0:
            return 1
        if partial_index is not None:
            return coverage[partial_index]
        else:
            return coverage[-1]

    def examples(self, only_different_prediction: bool = False,
                 only_same_prediction: bool = False, partial_index: int = None) -> np.ndarray:
        """
        Parameters
        ----------
        only_different_prediction
            If True, will only return examples where the result makes a different prediction than the original model
        only_same_prediction
            If True, will only return examples where the result makes the same prediction than the original model
        partial_index
            Get the examples from the partial result until a certain index

        Returns
        -------
        Examples covered by result
        """
        if only_different_prediction and only_same_prediction:
            print('Error: you cannot have only_different_prediction and only_same_prediction at the same time')
            return []
        key = 'covered'
        if only_different_prediction:
            key = 'covered_false'
        if only_same_prediction:
            key = 'covered_true'
        size = len(self.exp_map['examples'])
        idx = partial_index if partial_index is not None else size - 1
        if idx < 0 or idx > size:
            return []
        return self.exp_map['examples'][idx][key]
