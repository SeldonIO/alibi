import pytest
import numpy as np


@pytest.fixture(scope='module')
def blobs_dataset(request):
    param = request.param
    if param is None:
        param = dict()

    X, Y = [], []

    # get number of blobs
    num_blobs = param.get('num_blobs', 2)
    num_blobs = min(max(num_blobs, 2), 3)

    # get params of the first blob
    mu1 = param.get('mu1', np.array([[0, 0]]))
    std1 = param.get('std1', 0.1)
    size1 = param.get('size', 100)
    X.append(std1 * np.random.randn(size1, 2) + mu1)
    Y.append(np.zeros(size1, dtype=np.uint8))

    # get params of the second blob
    mu2 = param.get('mu2', np.array([[2, 2]]))
    std2 = param.get('std2', 0.1)
    size2 = param.get('size2', 100)
    X.append(std2 * np.random.randn(size2, 2) + mu2)
    Y.append(np.ones(size2, dtype=np.uint8))

    if num_blobs > 2:
        mu3 = param.get('mu3', np.array([[0, 2]]))
        std3 = param.get('std3', 0.1)
        size3 = param.get('size3', 100)
        X.append(std3 * np.random.randn(size3, 2) + mu3)
        Y.append(2 * np.ones(size3, dtype=np.uint8))

    # construct dataset
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y)
    return X, Y

