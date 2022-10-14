import os
import re
import tempfile
from typing import List

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from alibi.models.pytorch.metrics import AccuracyMetric, LossContainer
from alibi.models.pytorch.model import Model
from torch.utils.data import DataLoader, TensorDataset


class UnimodalModel(Model):
    """ Simple uni-modal output model. """

    def __init__(self, input_dim: int, output_dim: int):
        """
        Initializer.

        Parameters
        ----------
        input_dim
            Input dimension.
        output_dim
            Output dimension.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        """
        Forward pass

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        Prediction tensor.
        """
        return self.fc1(x)


class MultimodalModel(Model):
    """ Simple multi-modal output model. """

    def __init__(self, input_dim: int, output_dims: List[int]):
        """
        Initializer.

        Parameters
        ----------
        input_dim
            Input dimension.
        output_dims
            List of output dimensions for each modality.
        """
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(input_dim, dim) for dim in output_dims])
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        """
        Forward pass.

        Parameters
        ----------
        x
            Input tensor.

        Returns
        -------
        List of prediction tensors.
        """
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


@pytest.mark.parametrize('multimodal_model', [{'input_dim': 10, 'output_dims': [5, 3]}], indirect=True)
def test_validate_prediction_labels1(multimodal_model):
    """ Test if an error is raised when multiple loss function were compiled but the model outputs is a tensor
    instead of a list of tensors. """
    multimodal_model.compile(optimizer=optim.Adam(multimodal_model.parameters()),
                             loss=[nn.MSELoss(reduction='mean'), nn.CrossEntropyLoss(reduction='mean')],
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
                             loss=[nn.MSELoss(reduction='mean'), nn.BCELoss(reduction='mean')],
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
    assert re.search('Number of predictions differs from the number of labels.', err.value.args[0])


@pytest.mark.parametrize('unimodal_model', [{'input_dim': 10, 'output_dim': 10}], indirect=True)
def test_validate_prediction_labels4(unimodal_model):
    """ Test if an error is raised when a single loss function is complied but the prediction
    is a list of tensors. """
    unimodal_model.compile(optimizer=optim.Adam(unimodal_model.parameters()),
                           loss=nn.MSELoss(reduction='mean'))
    y_pred = [torch.randn(5, 5), torch.randn(5, 5)]
    y_true = torch.randn(5, 5)
    with pytest.raises(ValueError) as err:
        unimodal_model.validate_prediction_labels(y_pred=y_pred, y_true=y_true)
    assert re.search('The prediction is a list and should be a tensor since only one loss has been passed',
                     err.value.args[0])


@pytest.mark.parametrize('unimodal_model', [{'input_dim': 10, 'output_dim': 10}], indirect=True)
def test_compute_loss_unimodal(unimodal_model):
    """ Test if the loss computation for a single loss function matches the expectation. """
    loss = nn.MSELoss(reduction='mean')
    unimodal_model.compile(optimizer=optim.Adam(unimodal_model.parameters()), loss=nn.MSELoss(loss))

    y_true = torch.randn(10, 10)
    y_pred = torch.randn(10, 10)

    loss_val, _ = unimodal_model.compute_loss(y_pred=y_pred, y_true=y_true)
    expected_loss_val = loss(input=y_pred, target=y_true).item()
    assert np.allclose(loss_val, expected_loss_val)


@pytest.mark.parametrize('multimodal_model', [{'input_dim': 10, 'output_dims': [5, 3]}], indirect=True)
def test_compute_loss_multimodal(multimodal_model):
    """ Test if the loss computation for multiple loss functions matches the expectation. """
    loss1 = nn.MSELoss(reduction='mean')
    loss2 = nn.CrossEntropyLoss(reduction='mean')
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


@pytest.mark.parametrize('unimodal_model', [{'input_dim': 10, 'output_dim': 10}], indirect=True)
def test_compute_metrics_unimodal(unimodal_model):
    """ Test if the metric computation for a unimodal model matches the expectation. """
    unimodal_model.compile(optimizer=optim.Adam(unimodal_model.parameters()),
                           loss=nn.CrossEntropyLoss(reduction='mean'),
                           metrics=[AccuracyMetric()])

    y_pred = torch.randn(10, 5)
    y_true = torch.randint(low=0, high=5, size=(10, ))

    expected_acc = torch.mean((y_true == torch.argmax(y_pred, dim=-1)).float()).item()
    result = unimodal_model.compute_metrics(y_pred=y_pred, y_true=y_true)
    assert np.allclose(expected_acc, result['accuracy'])


@pytest.mark.parametrize('multimodal_model', [{'input_dim': 10, 'output_dims': [5, 5]}], indirect=True)
def test_compute_metrics_multimodal(multimodal_model):
    """ Test if the metrics computation for a multimodal model matches the expectation. """
    multimodal_model.compile(optimizer=optim.Adam(multimodal_model.parameters()),
                             loss=[nn.CrossEntropyLoss(reduction='mean'), nn.CrossEntropyLoss(reduction='mean')],
                             loss_weights=[1, 1],
                             metrics={
                                'output_1': AccuracyMetric(),
                                'output_2': AccuracyMetric(),
                             })

    y_pred = [torch.randn(10, 5), torch.randn(10, 5)]
    y_true = [torch.randint(low=0, high=5, size=(10,)), torch.randint(low=0, high=5, size=(10, ))]

    expected_acc1 = torch.mean((y_true[0] == torch.argmax(y_pred[0], dim=-1)).float()).item()
    expected_acc2 = torch.mean((y_true[1] == torch.argmax(y_pred[1], dim=-1)).float()).item()
    results = multimodal_model.compute_metrics(y_pred=y_pred, y_true=y_true)
    assert np.isclose(expected_acc1, results['output_1_accuracy'])
    assert np.isclose(expected_acc2, results['output_2_accuracy'])


@pytest.mark.parametrize('unimodal_model', [{'input_dim': 10, 'output_dim': 5}], indirect=True)
def test_train_step(unimodal_model, mocker):
    """ Test if the train step return the appropriate statistics. """
    unimodal_model.compile(optimizer=optim.Adam(unimodal_model.parameters()),
                           loss=nn.CrossEntropyLoss(reduction='mean'),
                           metrics=[AccuracyMetric()])

    x = torch.randn(10, 10)
    y_true = torch.randint(low=0, high=5, size=(10, ))

    mocker.patch('alibi.models.pytorch.model.nn.Module.train')
    mocker.patch('alibi.models.pytorch.model.Model.validate_prediction_labels')
    mocker.patch('alibi.models.pytorch.model.Model.forward')
    mocker.patch('alibi.models.pytorch.model.Model.compute_loss', return_value=[torch.tensor(0.), {'loss': 0.}])
    mocker.patch('alibi.models.pytorch.model.optim.Adam.zero_grad')
    mocker.patch('alibi.models.pytorch.model.torch.Tensor.backward')
    mocker.patch('alibi.models.pytorch.model.optim.Adam.step')
    mocker.patch('alibi.models.pytorch.model.Model.compute_metrics', return_value={'accuracy': 1.})
    results = unimodal_model.train_step(x, y_true)
    assert np.isclose(results['loss'], 0.)
    assert np.isclose(results['accuracy'], 1.)


@pytest.mark.parametrize('unimodal_model', [{'input_dim': 10, 'output_dim': 10}], indirect=True)
def test_test_step(unimodal_model, mocker):
    """ Test if the test step return the expected statistics. """
    unimodal_model.compile(optimizer=optim.Adam(unimodal_model.parameters()),
                           loss=nn.CrossEntropyLoss(reduction='mean'),
                           metrics=[AccuracyMetric()])

    x = torch.randn(10, 10)
    y_true = torch.randint(low=0, high=5, size=(10, ))

    mocker.patch('alibi.models.pytorch.model.nn.Module.eval')
    mocker.patch('alibi.models.pytorch.model.nn.Module.forward')
    mocker.patch('alibi.models.pytorch.model.Model.validate_prediction_labels')
    mocker.patch('alibi.models.pytorch.model.Model.compute_loss', return_value=[torch.tensor(0.), {'loss': 0.}])
    mocker.patch('alibi.models.pytorch.model.Model.compute_metrics', return_value={'accuracy': 1.})
    results = unimodal_model.test_step(x, y_true)
    assert np.allclose(results['loss'], 0)
    assert np.allclose(results['accuracy'], 1)


@pytest.mark.parametrize('unimodal_model', [{'input_dim': 10, 'output_dim': 10}], indirect=True)
def test_fit(unimodal_model, mocker):
    """ Test if the fit function returns the expected statistics. """
    x = torch.randn(10, 10)
    y = torch.randint(low=0, high=5, size=(10, ))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=len(x))

    unimodal_model.compile(optimizer=optim.Adam(unimodal_model.parameters()),
                           loss=nn.CrossEntropyLoss(reduction='mean'),
                           metrics=[AccuracyMetric()])

    mocker.patch('alibi.models.pytorch.model.Model.train_step', return_value={'loss': 0., 'accuracy': 1.})
    metrics_val = unimodal_model.fit(trainloader=dataloader, epochs=1)
    assert np.allclose(metrics_val['loss'], 0.)
    assert np.allclose(metrics_val['accuracy'], 1.)


@pytest.mark.parametrize('unimodal_model', [{'input_dim': 10, 'output_dim': 10}], indirect=True)
def test_evaluate(unimodal_model, mocker):
    x = torch.randn(10, 10)
    y = torch.randint(low=0, high=5, size=(10, ))
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=len(x))

    unimodal_model.compile(optimizer=optim.Adam(unimodal_model.parameters()),
                           loss=nn.CrossEntropyLoss(reduction='mean'),
                           metrics=[AccuracyMetric()])

    mocker.patch('alibi.models.pytorch.model.Model.test_step', return_value={'loss': 0, 'accuracy': 1.})
    metrics_val = unimodal_model.evaluate(testloader=dataloader)
    assert np.allclose(metrics_val['loss'], 0)
    assert np.allclose(metrics_val['accuracy'], 1)


def test_metrics_to_str():
    """ Test the string representation of the metrics dictionary. """
    metrics = {'metric1': 0.9134, 'metric2': 0.3213}
    expected_metrics_str = "metric1: 0.9134\tmetric2: 0.3213\t"
    metric_str = Model._metrics_to_str(metrics)
    assert expected_metrics_str == metric_str


@pytest.mark.parametrize('unimodal_model', [{'input_dim': 10, 'output_dim': 10}], indirect=True)
def test_reset_loss_unimodal(unimodal_model):
    """ Test if the model is resetting the loss function in the unimodal case."""
    unimodal_model.compile(optimizer=optim.Adam(unimodal_model.parameters()),
                           loss=nn.CrossEntropyLoss(reduction='mean'))

    y_pred = torch.randn(10, 5)
    y_true = torch.randint(low=0, high=5, size=(10, ))
    unimodal_model.loss(y_pred, y_true)

    unimodal_model._reset_loss()
    assert np.isclose(unimodal_model.loss.total, 0)
    assert np.isclose(unimodal_model.loss.count, 0)


@pytest.mark.parametrize('multimodal_model', [{'input_dim': 10, 'output_dims': [5, 5]}], indirect=True)
def test_reset_loss_multimodal(multimodal_model):
    """ Test if the model is resetting the loss function in the multimodal case. """
    multimodal_model.compile(optimizer=optim.Adam(multimodal_model.parameters()),
                             loss=[nn.CrossEntropyLoss(reduction='mean'), nn.CrossEntropyLoss(reduction='mean')],
                             loss_weights=[1., 1.])

    y_pred = [torch.randn(10, 5), torch.randn(10, 5)]
    y_true = [torch.randint(low=0, high=5, size=(10, )), torch.randint(low=0, high=5, size=(10, ))]

    for i in range(2):
        multimodal_model.loss[i](y_true=y_true[i], y_pred=y_pred[i])

    multimodal_model._reset_loss()

    for i in range(2):
        assert np.isclose(multimodal_model.loss[i].total, 0)
        assert np.isclose(multimodal_model.loss[i].count, 0)


@pytest.mark.parametrize('unimodal_model', [{'input_dim': 10, 'output_dim': 10}], indirect=True)
def test_reset_metrics_unimodal(unimodal_model):
    """ Test if the model is resetting the metrics in the unimodal case. """
    unimodal_model.compile(optimizer=optim.Adam(unimodal_model.parameters()),
                           loss=nn.CrossEntropyLoss(reduction='mean'),
                           metrics=[AccuracyMetric()])

    y_pred = torch.randn(10, 5)
    y_true = torch.randint(low=0, high=5, size=(10, ))

    unimodal_model.metrics[0].compute_metric(y_true=y_true, y_pred=y_pred)
    unimodal_model._reset_metrics()
    assert np.isclose(unimodal_model.metrics[0].total, 0)
    assert np.isclose(unimodal_model.metrics[0].count, 0)


@pytest.mark.parametrize('multimodal_model', [{'input_dim': 10, 'output_dims': [5, 5]}], indirect=True)
def test_reset_metrics_multimodal(multimodal_model):
    """ Test if the model is resetting the metrics in the multimodal case. """
    multimodal_model.compile(optimizer=optim.Adam(multimodal_model.parameters()),
                             loss=nn.CrossEntropyLoss(reduction='mean'),
                             metrics = {
                                 'output_1': AccuracyMetric(),
                                 'output_2': AccuracyMetric(),
                             })

    y_pred = [torch.randn(10, 5), torch.randn(10, 5)]
    y_true = [torch.randint(low=0, high=5, size=(10, )), torch.randint(low=0, high=5, size=(10, ))]

    for i in range(2):
        key = f'output_{i + 1}'
        multimodal_model.metrics[key].compute_metric(y_true=y_true[i], y_pred=y_pred[i])

    multimodal_model._reset_metrics()

    for i in range(2):
        key = f'output_{i + 1}'
        assert np.isclose(multimodal_model.metrics[key].total, 0)
        assert np.isclose(multimodal_model.metrics[key].count, 0)


def test_saving():
    """ Test saving functionality. """
    model1 = UnimodalModel(input_dim=5, output_dim=5)
    model2 = UnimodalModel(input_dim=5, output_dim=5)

    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, 'weights.pt')
        model1.save_weights(path)
        model2.load_weights(path)

    for params1, params2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(params1, params2)