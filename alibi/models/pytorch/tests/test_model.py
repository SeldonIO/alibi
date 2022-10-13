import re
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from alibi.models.pytorch.model import Model
from alibi.models.pytorch.metrics import AccuracyMetric, LossContainer
from typing import List


class UnimodalModel(Model):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.to(self.device)

    def forward(self, x):
        return self.fc1(x)


class MultimodalModel(Model):
    def __init__(self, input_dim: int, output_dims: List[int]):
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(input_dim, dim) for dim in output_dims])
        self.to(self.device)

    def forward(self, x):
        return [fc(x) for fc in self.fcs]


@pytest.fixture(scope='module')
def unimodal_model(request):
    return UnimodalModel(input_dim=request.param['input_dim'], output_dim=request.param['output_dim'])


@pytest.fixture(scope='module')
def multimodal_model(request):
    return MultimodalModel(input_dim=request.param['input_dim'], output_dims=request.param['output_dims'])


@pytest.mark.parametrize('unimodal_model', [{'input_dim': 10, 'output_dim': 1}], indirect=True)
def test_compile_unimodal(unimodal_model):
    """ Test compile function when a single loss function is passed. """
    unimodal_model.compile(optimizer=optim.Adam(unimodal_model.parameters()),
                           loss=nn.BCELoss(),
                           metrics=[AccuracyMetric()])
    assert isinstance(unimodal_model.loss, LossContainer)


@pytest.mark.parametrize('multimodal_model', [{'input_dim': 10, 'output_dims': [5, 3]}], indirect=True)
def test_compile_multimodal(multimodal_model):
    """ Test compile function when multiple loss functions are passed. """
    multimodal_model.compile(optimizer=optim.Adam(multimodal_model.parameters()),
                             loss=[nn.BCELoss(), nn.MSELoss()],
                             loss_weights=[1.0, 0.5],
                             metrics=[AccuracyMetric()])
    assert isinstance(multimodal_model.loss, list)
    assert all([isinstance(l, LossContainer) for l in multimodal_model.loss])


@pytest.mark.parametrize('multimodal_model', [{'input_dim': 10, 'output_dims': [5, 3]}], indirect=True)
def test_compile_multimodal_mismatch(multimodal_model):
    """ Test compile function raises and error when multiple loss functions are passed but the number \
    of loss weights does not match the number of loss functions. """
    with pytest.raises(ValueError) as err:
        multimodal_model.compile(optimizer=optim.Adam(multimodal_model.parameters()),
                                 loss=[nn.BCELoss(), nn.MSELoss()],
                                 loss_weights=[0.5],
                                 metrics=[AccuracyMetric()])
    assert re.search('The number of loss weights differs from the number of losses', err.value.args[0])


@pytest.mark.parametrize('unimodal_model', [{'input_dim': 10, 'output_dim': 10}], indirect=True)
def test_compute_loss_unimodal(unimodal_model):
    """ Test if the loss computation for a single loss function matches the expectation. """
    loss = nn.MSELoss(reduction="mean")
    unimodal_model.compile(optimizer=optim.Adam(unimodal_model.parameters()), loss=nn.MSELoss(loss))

    y_true = torch.randn(10, 10)
    y_pred = torch.randn(10, 10)

    loss_val, _ = unimodal_model.compute_loss(y_pred=y_pred, y_true=y_true)
    expected_loss_val = loss(input=y_pred, target=y_true).item()
    assert np.allclose(loss_val, expected_loss_val)


@pytest.mark.parametrize('multimodal_model', [{'input_dim': 10, 'output_dims': [5, 3]}], indirect=True)
def test_validate_prediction_labels1(multimodal_model):
    """ Test if an error is raised when multiple loss function were compiled but the model outputs is a tensor
    instead of a list of tensors. """
    multimodal_model.compile(optimizer=optim.Adam(multimodal_model.parameters()),
                             loss=[nn.MSELoss(reduction="mean"), nn.BCELoss(reduction="mean")],
                             loss_weights=[1, 1])
    y_pred = torch.randn(5, 5)
    y_true = [torch.randn(5, 5), torch.randn(5, 3)]
    with pytest.raises(ValueError) as err:
        multimodal_model.validate_prediction_labels(y_pred=y_pred, y_true=y_true)
    assert re.search('The prediction should be a list since list of losses have been passed.', err.value.args[0])


@pytest.mark.parametrize('multimodal_model', [{'input_dim': 10, 'output_dims': [5, 3]}], indirect=True)
def test_validate_prediction_labels2(multimodal_model):
    """ Test if an error is raised when multiple loss functions were compiled but the target is a tensor
    instead of a list of tensors. """
    multimodal_model.compile(optimizer=optim.Adam(multimodal_model.parameters()),
                             loss=[nn.MSELoss(reduction="mean"), nn.BCELoss(reduction="mean")],
                             loss_weights=[1, 1])
    y_pred = [torch.randn(5, 5), torch.randn(5, 3)]
    y_true = torch.randn(5, 5)
    with pytest.raises(ValueError) as err:
        multimodal_model.validate_prediction_labels(y_pred=y_pred, y_true=y_true)
    assert re.search('The label should be a list since list of losses have been passed.', err.value.args[0])


@pytest.mark.parametrize('multimodal_model', [{'input_dim': 10, 'output_dims': [5, 3]}], indirect=True)
def test_validate_prediction_labels3(multimodal_model):
    """ Test if an error is raised when multiple loss functions were compiled but the length of the target and\
    the length of the predictions do not match. """
    multimodal_model.compile(optimizer=optim.Adam(multimodal_model.parameters()),
                             loss=[nn.MSELoss(reduction='mean'), nn.BCELoss(reduction='mean')],
                             loss_weights=[1, 1])

    y_pred = [torch.randn(5, 5), torch.randn(5, 3)]
    y_true = [torch.randn(5, 5)]
    with pytest.raises(ValueError) as err:
        multimodal_model.validate_prediction_labels(y_pred=y_pred, y_true=y_true)
    assert re.search("Number of predictions differs from the number of labels.", err.value.args[0])


@pytest.mark.parametrize('unimodal_model', [{'input_dim': 10, 'output_dim': 10}], indirect=True)
def test_validate_prediction_labels4(unimodal_model):
    """ Test if an error is raised when a single loss function is complied but the prediction
    is a list of tensors. """
    unimodal_model.compile(optimizer=optim.Adam(unimodal_model.parameters()),
                           loss=nn.MSELoss(reduction="mean"))
    y_pred = [torch.randn(5, 5), torch.randn(5, 5)]
    y_true = torch.randn(5, 5)
    with pytest.raises(ValueError) as err:
        multimodal_model.validate_prediction_labels(y_pred=y_pred, y_true=y_true)
    assert re.search("The prediction is a list and should be a tensor since only one loss has been passed",
                     err.value.args[0])



@pytest.mark.parametrize('multimodal_model', [{'input_dim': 10, 'output_dims': [5, 3]}], indirect=True)
def test_compute_loss_multimodal(multimodal_model):
    """ Test if the loss computation for multiple loss functions matches the expectation. """
    loss1 = nn.MSELoss(reduction="mean")
    loss2 = nn.CrossEntropyLoss(reduction="mean")
    loss_weights = [0.25, 0.35]
    multimodal_model.compile(optimizer=optim.Adam(multimodal_model.parameters()),
                             loss=[loss1, loss2],
                             loss_weights=[0.25, 0.35])

    y_pred = [torch.randn(10, 5), torch.randn(10, 3)]
    y_true = [torch.randn(10, 5), torch.randint(low=0, high=3, size=(10,))]

    loss_val, _ = multimodal_model.compute_loss(y_pred=y_pred, y_true=y_true)
    expected_loss1_val = loss1(input=y_pred[0], target=y_true[0]).item()
    expected_loss2_val = loss2(input=y_pred[1], target=y_true[1]).item()
    expected_loss_val = loss_weights[0] * expected_loss1_val + loss_weights[1] * expected_loss2_val
    assert np.allclose(loss_val, expected_loss_val)
