import numpy as np
import pytest
import torch
import torch.nn as nn
from alibi.models.pytorch.metrics import AccuracyMetric, LossContainer


@pytest.mark.parametrize("num_batches", [5, 10])
def test_loss_container(num_batches):
    """ Test `LossContainer` wrapper for classification task. """
    loss = nn.CrossEntropyLoss(reduction='mean')
    loss_container = LossContainer(loss=loss, name='CrossEntropyLoss')
    cumulated_loss_val, counts = 0, 0

    for i in range(num_batches):
        input = torch.randn(3, 5)
        target = torch.empty(3, dtype=torch.long).random_(5)

        loss_val1 = loss_container(y_pred=input, y_true=target)
        loss_val2 = loss(input, target)
        assert torch.allclose(loss_val1, loss_val2)

        cumulated_loss_val += loss_val2.item()
        counts += 1

    expected_val = cumulated_loss_val / counts
    result = loss_container.result()
    assert np.isclose(expected_val, result['CrossEntropyLoss'])

    loss_container.reset()
    assert np.isclose(loss_container.total, 0)
    assert np.isclose(loss_container.count, 0)


@pytest.mark.parametrize('num_batches', [5, 10])
def test_accuracy_metric(num_batches):
    """ Test accuracy metric for different reductions procedures. """
    acc_metric = AccuracyMetric()
    cumulated_acc, counts = 0, 0

    for i in range(num_batches):
        input = torch.randn(3, 5)
        target = torch.empty(3, dtype=torch.long).random_(5)

        acc_metric.compute_metric(y_pred=input, y_true=target)
        cumulated_acc += torch.sum(torch.argmax(input, dim=-1) == target)
        counts += len(target)

    expected_value = cumulated_acc / counts
    result = acc_metric.result()
    assert np.isclose(expected_value, result['accuracy'])

    acc_metric.reset()
    assert np.isclose(acc_metric.total, 0)
    assert np.isclose(acc_metric.count, 0)
