import numpy as np
import pytest
from alibi.prototypes import ProtoSelect
from alibi.utils.kernel import EuclideanDistance


@pytest.mark.parametrize('blobs_dataset', [
    {'num_blobs': 2, 'size1': 10, 'size2': 100},
    {'num_blobs': 2, 'size1': 100, 'size2': 10},
    {'num_blobs': 2, 'size1': 100, 'size2': 100},
    {'num_blobs': 3, 'size1': 10, 'size2': 10, 'size3': 100},
    {'num_blobs': 3, 'size1': 10, 'size2': 100, 'size3': 10},
    {'num_blobs': 3, 'size1': 100, 'size2': 10, 'size3': 10},
    {'num_blobs': 3, 'size1': 100, 'size2': 100, 'size3': 100}
], indirect=True)
@pytest.mark.parametrize('kernel_distance', [EuclideanDistance()])
@pytest.mark.parametrize('num_prototypes', [3, 10, 100, 100])
@pytest.mark.parametrize('eps', [0.2])
def test_multimodal(blobs_dataset, kernel_distance, num_prototypes, eps):
    """
    Checks if ProtoSelect select an element from each cluster.
    """
    X, Y = blobs_dataset

    # define & fit the explainer
    explainer = ProtoSelect(eps=eps, kernel_distance=kernel_distance)
    explainer = explainer.fit(X=X, X_labels=Y)

    # get prototypes
    explanation = explainer.explain(num_prototypes=num_prototypes)
    protos = explanation.prototypes
    protos_indices = explanation.prototypes_indices
    protos_labels = explanation.prototypes_labels

    assert len(protos) == len(protos_indices) == len(protos_labels)
    assert len(protos) <= num_prototypes
    assert len(np.unique(protos_labels)) == len(np.unique(Y))
