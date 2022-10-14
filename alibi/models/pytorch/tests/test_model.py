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
from pytest_lazyfixture import lazy_fixture
from torch.utils.data import DataLoader, TensorDataset


class UnimodalModel(Model):
    """ Simple uni-modal output model. """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        return self.fc1(x)


class MultimodalModel(Model):
    """ Simple multi-modal output model. """

    def __init__(self, input_dim: int, output_dims: List[int]):
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(input_dim, dim) for dim in output_dims])
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        return [fc(x) for fc in self.fcs]


n_instances = 10
input_dim = 5
output_dim = 5
output_dims = [5, 5]

loss_fn = nn.CrossEntropyLoss(reduction='mean')
loss_fns = [nn.CrossEntropyLoss(reduction='mean'), nn.CrossEntropyLoss(reduction='mean')]
loss_weights = [0.3, 0.4]

metric = [AccuracyMetric()]
metrics = {
    'output_1': AccuracyMetric(),
    'output_2': AccuracyMetric(),
}

optimizer_class = optim.Adam


@pytest.fixture(scope='module')
def dataset():
    return {
        'input': torch.randn(n_instances, input_dim),
        'output': {
            'unimodal': {
                'y_pred': torch.randn(n_instances, output_dim),
                'y_true': torch.randint(low=0, high=output_dim, size=(n_instances, ))
            },
            'multimodal': {
                'y_pred': [
                    torch.randn(n_instances, output_dims[0]),
                    torch.randn(n_instances, output_dims[1])
                ],
                'y_true': [
                    torch.randint(low=0, high=output_dims[0], size=(n_instances, )),
                    torch.randint(low=0, high=output_dims[1], size=(n_instances, ))
                ]
            }
        }
    }


@pytest.fixture(scope='module')
def unimodal_model(request):
    """ Creates and optionally compiles a uni-modal output model. """
    model = UnimodalModel(input_dim=request.param['input_dim'],
                          output_dim=request.param['output_dim'])

    if request.param.get('compile', False):
        model.compile(optimizer=request.param['optimizer'](model.parameters()),
                      loss=request.param['loss'],
                      metrics=request.param.get('metrics', None))

    return model


@pytest.fixture(scope='module')
def multimodal_model(request):
    """ Creates and optionally compiles a multi-modal output model. """
    model = MultimodalModel(input_dim=request.param['input_dim'],
                            output_dims=request.param['output_dims'])

    if request.param.get('compile', False):
        model.compile(optimizer=request.param['optimizer'](model.parameters()),
                      loss=request.param['loss'],
                      loss_weights=request.param['loss_weights'],
                      metrics=request.param.get('metrics', None))

    return model


@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optimizer_class,
        'loss': loss_fn,
        'metrics': metric
    }
], indirect=True)
def test_compile_unimodal(unimodal_model):
    """ Test compile function when a single loss function is passed. """
    assert isinstance(unimodal_model.loss, LossContainer)


@pytest.mark.parametrize('multimodal_model', [
    {
        'input_dim': input_dim,
        'output_dims': output_dims,
        'compile': True,
        'optimizer': optimizer_class,
        'loss': loss_fns,
        'loss_weights': loss_weights,
        'metrics': metrics
    }
], indirect=True)
def test_compile_multimodal(multimodal_model):
    """ Test compile function when multiple loss functions are passed. """
    assert isinstance(multimodal_model.loss, list)
    assert all([isinstance(l, LossContainer) for l in multimodal_model.loss])


@pytest.mark.parametrize('multimodal_model', [
    {
        'input_dim': input_dim,
        'output_dims': output_dims,
    }
], indirect=True)
def test_compile_multimodal_mismatch(multimodal_model):
    """ Test compile function raises and error when multiple loss functions are passed but the number \
    of loss weights does not match the number of loss functions. """
    with pytest.raises(ValueError) as err:
        multimodal_model.compile(optimizer=optimizer_class(multimodal_model.parameters()),
                                 loss=loss_fns,
                                 loss_weights=[0.5])

    assert 'The number of loss weights differs from the number of losses' in str(err.value)


@pytest.mark.parametrize('multimodal_model', [
    {
        'input_dim': input_dim,
        'output_dims': output_dims,
        'compile': True,
        'optimizer': optimizer_class,
        'loss': loss_fns,
        'loss_weights': loss_weights
    }
], indirect=True)
def test_validate_prediction_labels1(multimodal_model, dataset):
    """ Test if an error is raised when multiple loss function were compiled but the model outputs is a tensor
    instead of a list of tensors. """
    y_pred = dataset['output']['unimodal']['y_pred']
    y_true = dataset['output']['multimodal']['y_true']

    with pytest.raises(ValueError) as err:
        multimodal_model.validate_prediction_labels(y_pred=y_pred, y_true=y_true)

    assert 'The prediction should be a list since list of losses have been passed.' in str(err.value)


@pytest.mark.parametrize('multimodal_model', [
    {
        'input_dim': input_dim,
        'output_dims': output_dims,
        'compile': True,
        'optimizer': optimizer_class,
        'loss': loss_fns,
        'loss_weights': loss_weights,
    }
], indirect=True)
def test_validate_prediction_labels2(multimodal_model, dataset):
    """ Test if an error is raised when multiple loss functions were compiled but the target is a tensor
    instead of a list of tensors. """
    y_pred = dataset['output']['multimodal']['y_pred']
    y_true = dataset['output']['unimodal']['y_pred']

    with pytest.raises(ValueError) as err:
        multimodal_model.validate_prediction_labels(y_pred=y_pred, y_true=y_true)

    assert 'The label should be a list since list of losses have been passed.' in str(err.value)


@pytest.mark.parametrize('multimodal_model', [
    {
        'input_dim': input_dim,
        'output_dims': output_dims,
        'compile': True,
        'optimizer': optimizer_class,
        'loss': loss_fns,
        'loss_weights': loss_weights
    }
], indirect=True)
def test_validate_prediction_labels3(multimodal_model, dataset):
    """ Test if an error is raised when multiple loss functions were compiled but the length of the target and\
    the length of the predictions do not match. """
    y_pred = dataset['output']['multimodal']['y_pred']
    y_true = dataset['output']['multimodal']['y_true'][:1]

    with pytest.raises(ValueError) as err:
        multimodal_model.validate_prediction_labels(y_pred=y_pred, y_true=y_true)

    assert 'Number of predictions differs from the number of labels.' in str(err.value)


@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optimizer_class,
        'loss': loss_fn,
    }
], indirect=True)
def test_validate_prediction_labels4(unimodal_model, dataset):
    """ Test if an error is raised when a single loss function is complied but the prediction
    is a list of tensors. """
    y_pred = dataset['output']['multimodal']['y_pred']
    y_true = dataset['output']['unimodal']['y_true']

    with pytest.raises(ValueError) as err:
        unimodal_model.validate_prediction_labels(y_pred=y_pred, y_true=y_true)

    assert 'The prediction is a list and should be a tensor since only one loss has been passed' in str(err.value)


@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optimizer_class,
        'loss': loss_fn,
    }
], indirect=True)
def test_compute_loss_unimodal(unimodal_model, dataset):
    """ Test if the loss computation for a single loss function matches the expectation. """
    y_true = dataset['output']['unimodal']['y_true']
    y_pred = dataset['output']['unimodal']['y_pred']

    loss_val, _ = unimodal_model.compute_loss(y_pred=y_pred, y_true=y_true)
    expected_loss_val = loss_fn(input=y_pred, target=y_true).item()
    assert np.allclose(loss_val, expected_loss_val)


@pytest.mark.parametrize('multimodal_model', [
    {
        'input_dim': input_dim,
        'output_dims': output_dims,
        'compile': True,
        'optimizer': optimizer_class,
        'loss': loss_fns,
        'loss_weights': loss_weights,
    }
], indirect=True)
def test_compute_loss_multimodal(multimodal_model, dataset):
    """ Test if the loss computation for multiple loss functions matches the expectation. """
    y_pred = dataset['output']['multimodal']['y_pred']
    y_true = dataset['output']['multimodal']['y_true']

    loss_val, _ = multimodal_model.compute_loss(y_pred=y_pred, y_true=y_true)
    expected_loss0_val = loss_fns[0](input=y_pred[0], target=y_true[0]).item()
    expected_loss1_val = loss_fns[1](input=y_pred[1], target=y_true[1]).item()
    expected_loss_val = loss_weights[0] * expected_loss0_val + loss_weights[1] * expected_loss1_val
    assert np.allclose(loss_val, expected_loss_val)


@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optimizer_class,
        'loss': loss_fn,
        'metrics': metric
    }
], indirect=True)
def test_compute_metrics_unimodal(unimodal_model, dataset):
    """ Test if the metric computation for a unimodal model matches the expectation. """
    y_pred = dataset['output']['unimodal']['y_pred']
    y_true = dataset['output']['unimodal']['y_true']

    expected_acc = torch.mean((y_true == torch.argmax(y_pred, dim=-1)).float()).item()
    result = unimodal_model.compute_metrics(y_pred=y_pred, y_true=y_true)
    assert np.allclose(expected_acc, result['accuracy'])


@pytest.mark.parametrize('multimodal_model', [
    {
        'input_dim': input_dim,
        'output_dims': output_dims,
        'compile': True,
        'optimizer': optimizer_class,
        'loss': loss_fns,
        'loss_weights': loss_weights,
        'metrics': metrics
    }
], indirect=True)
def test_compute_metrics_multimodal(multimodal_model, dataset):
    """ Test if the metrics computation for a multimodal model matches the expectation. """
    y_pred = dataset['output']['multimodal']['y_pred']
    y_true = dataset['output']['multimodal']['y_true']

    expected_acc1 = torch.mean((y_true[0] == torch.argmax(y_pred[0], dim=-1)).float()).item()
    expected_acc2 = torch.mean((y_true[1] == torch.argmax(y_pred[1], dim=-1)).float()).item()
    results = multimodal_model.compute_metrics(y_pred=y_pred, y_true=y_true)
    assert np.isclose(expected_acc1, results['output_1_accuracy'])
    assert np.isclose(expected_acc2, results['output_2_accuracy'])


@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optimizer_class,
        'loss': loss_fn,
        'metrics': metric
    }
], indirect=True)
def test_train_step(unimodal_model, dataset, mocker):
    """ Test if the train step return the appropriate statistics. """
    x = dataset['input']
    y_true = dataset['output']['unimodal']['y_true']

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


@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optimizer_class,
        'loss': loss_fn,
        'metrics': metric,
    }
], indirect=True)
def test_test_step(unimodal_model, dataset, mocker):
    """ Test if the test step return the expected statistics. """
    x = dataset['input']
    y_true = dataset['output']['unimodal']['y_true']

    mocker.patch('alibi.models.pytorch.model.nn.Module.eval')
    mocker.patch('alibi.models.pytorch.model.nn.Module.forward')
    mocker.patch('alibi.models.pytorch.model.Model.validate_prediction_labels')
    mocker.patch('alibi.models.pytorch.model.Model.compute_loss', return_value=[torch.tensor(0.), {'loss': 0.}])
    mocker.patch('alibi.models.pytorch.model.Model.compute_metrics', return_value={'accuracy': 1.})

    results = unimodal_model.test_step(x, y_true)
    assert np.allclose(results['loss'], 0)
    assert np.allclose(results['accuracy'], 1)


@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optimizer_class,
        'loss': loss_fn,
        'metrics': metric,
    }
], indirect=True)
def test_fit(unimodal_model, dataset, mocker):
    """ Test if the fit function returns the expected statistics. """
    x = dataset['input']
    y = dataset['output']['unimodal']['y_true']
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=len(x))

    mocker.patch('alibi.models.pytorch.model.Model.train_step', return_value={'loss': 0., 'accuracy': 1.})
    metrics_val = unimodal_model.fit(trainloader=dataloader, epochs=1)
    assert np.allclose(metrics_val['loss'], 0.)
    assert np.allclose(metrics_val['accuracy'], 1.)


@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optimizer_class,
        'loss': loss_fn,
        'metrics': metric,
    }
], indirect=True)
def test_evaluate(unimodal_model, dataset, mocker):
    """ Test if the fit function returns the expected statistics. """
    x = dataset['input']
    y = dataset['output']['unimodal']['y_true']
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=len(x))

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


@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optimizer_class,
        'loss': loss_fn,
    }
], indirect=True)
def test_reset_loss_unimodal(unimodal_model, dataset):
    """ Test if the model is resetting the loss function in the unimodal case."""
    y_pred = dataset['output']['unimodal']['y_pred']
    y_true = dataset['output']['unimodal']['y_true']

    unimodal_model.loss(y_pred, y_true)
    unimodal_model._reset_loss()
    assert np.isclose(unimodal_model.loss.total, 0)
    assert np.isclose(unimodal_model.loss.count, 0)


@pytest.mark.parametrize('multimodal_model', [
    {
        'input_dim': input_dim,
        'output_dims': output_dims,
        'compile': True,
        'optimizer': optimizer_class,
        'loss': loss_fns,
        'loss_weights': loss_weights,
    }
], indirect=True)
def test_reset_loss_multimodal(multimodal_model, dataset):
    """ Test if the model is resetting the loss function in the multimodal case. """
    y_pred = dataset['output']['multimodal']['y_pred']
    y_true = dataset['output']['multimodal']['y_true']

    for i in range(2):
        multimodal_model.loss[i](y_true=y_true[i], y_pred=y_pred[i])

    multimodal_model._reset_loss()

    for i in range(2):
        assert np.isclose(multimodal_model.loss[i].total, 0)
        assert np.isclose(multimodal_model.loss[i].count, 0)


@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optimizer_class,
        'loss': loss_fn,
        'metrics': metric,
    }
], indirect=True)
def test_reset_metrics_unimodal(unimodal_model, dataset):
    """ Test if the model is resetting the metrics in the unimodal case. """
    y_pred = dataset['output']['unimodal']['y_pred']
    y_true = dataset['output']['unimodal']['y_true']

    unimodal_model.metrics[0].compute_metric(y_true=y_true, y_pred=y_pred)
    unimodal_model._reset_metrics()
    assert np.isclose(unimodal_model.metrics[0].total, 0)
    assert np.isclose(unimodal_model.metrics[0].count, 0)


@pytest.mark.parametrize('multimodal_model', [
    {
        'input_dim': input_dim,
        'output_dims': output_dims,
        'compile': True,
        'optimizer': optimizer_class,
        'loss': loss_fn,
        'loss_weights': loss_weights,
        'metrics': metrics,
    }
], indirect=True)
def test_reset_metrics_multimodal(multimodal_model, dataset):
    """ Test if the model is resetting the metrics in the multimodal case. """
    y_pred = dataset['output']['multimodal']['y_pred']
    y_true = dataset['output']['multimodal']['y_true']

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