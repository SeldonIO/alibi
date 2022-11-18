from typing import List

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from alibi.models.pytorch.metrics import AccuracyMetric, LossContainer
from alibi.models.pytorch.model import Model


class UnimodalModel(Model):
    """ Simple uni-modal output model. """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim, bias=False)
        self.to(self.device)

    def forward(self, x: torch.Tensor):
        return self.fc1(x)


class MultimodalModel(Model):
    """ Simple multi-modal output model. """

    def __init__(self, input_dim: int, output_dims: List[int]):
        super().__init__()
        self.fcs = nn.ModuleList([nn.Linear(input_dim, dim, bias=False) for dim in output_dims])
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
lr = 1e-3


@pytest.fixture(scope='module')
def dataset(request):
    task = request.param
    return {
        'input': torch.randn(n_instances, input_dim),
        'output': {
            'unimodal': {
                'y_pred': torch.randn(n_instances, output_dim),
                'y_true': torch.randint(low=0, high=output_dim, size=(n_instances, )) if task == 'classification'
                else torch.randn(n_instances, output_dim)
            },
            'multimodal': {
                'y_pred': [
                    torch.randn(n_instances, output_dims[0]),
                    torch.randn(n_instances, output_dims[1])
                ],
                'y_true': [
                    torch.randint(low=0, high=output_dims[0], size=(n_instances, )) if task == 'classification'
                    else torch.randn(n_instances, output_dim),
                    torch.randint(low=0, high=output_dims[1], size=(n_instances, )) if task == 'classification'
                    else torch.randn(n_instances, output_dim)
                ]
            }
        }
    }


@pytest.fixture(scope='function')
def unimodal_model(request):
    """ Creates and optionally compiles a uni-modal output model. """
    model = UnimodalModel(input_dim=request.param['input_dim'],
                          output_dim=request.param['output_dim'])

    if request.param.get('compile', False):
        model.compile(optimizer=request.param['optimizer'](model.parameters(), lr=lr),
                      loss=request.param['loss'],
                      metrics=request.param.get('metrics', None))

    return model


@pytest.fixture(scope='function')
def multimodal_model(request):
    """ Creates and optionally compiles a multi-modal output model. """
    model = MultimodalModel(input_dim=request.param['input_dim'],
                            output_dims=request.param['output_dims'])

    if request.param.get('compile', False):
        model.compile(optimizer=request.param['optimizer'](model.parameters(), lr=lr),
                      loss=request.param['loss'],
                      loss_weights=request.param['loss_weights'],
                      metrics=request.param.get('metrics', None))

    return model


@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optim.SGD,
        'loss': loss_fn,
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
        'optimizer': optim.SGD,
        'loss': loss_fns,
        'loss_weights': loss_weights,
    }
], indirect=True)
def test_compile_multimodal(multimodal_model):
    """ Test compile function when multiple loss functions are passed. """
    assert isinstance(multimodal_model.loss, list)
    assert all([isinstance(x, LossContainer) for x in multimodal_model.loss])


@pytest.mark.parametrize('multimodal_model', [
    {
        'input_dim': input_dim,
        'output_dims': output_dims,
    }
], indirect=True)
def test_compile_multimodal_mismatch(multimodal_model):
    """ Test compile function raises an error when multiple loss functions are passed but the number \
    of loss weights does not match the number of loss functions. """
    with pytest.raises(ValueError) as err:
        multimodal_model.compile(optimizer=optim.SGD(multimodal_model.parameters(), lr=lr),
                                 loss=loss_fns,
                                 loss_weights=loss_weights[:1])

    assert 'The number of loss weights differs from the number of losses' in str(err.value)


@pytest.mark.parametrize('dataset', ['classification'], indirect=True)
@pytest.mark.parametrize('multimodal_model', [
    {
        'input_dim': input_dim,
        'output_dims': output_dims,
        'compile': True,
        'optimizer': optim.SGD,
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


@pytest.mark.parametrize('dataset', ['classification'], indirect=True)
@pytest.mark.parametrize('multimodal_model', [
    {
        'input_dim': input_dim,
        'output_dims': output_dims,
        'compile': True,
        'optimizer': optim.SGD,
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


@pytest.mark.parametrize('dataset', ['classification'], indirect=True)
@pytest.mark.parametrize('multimodal_model', [
    {
        'input_dim': input_dim,
        'output_dims': output_dims,
        'compile': True,
        'optimizer': optim.SGD,
        'loss': loss_fns,
        'loss_weights': loss_weights
    }
], indirect=True)
def test_validate_prediction_labels3(multimodal_model, dataset):
    """ Test if an error is raised when multiple loss functions were compiled but the length of the target and
    the length of the predictions do not match. """
    y_pred = dataset['output']['multimodal']['y_pred']
    y_true = dataset['output']['multimodal']['y_true'][:1]

    with pytest.raises(ValueError) as err:
        multimodal_model.validate_prediction_labels(y_pred=y_pred, y_true=y_true)

    assert 'Number of predictions differs from the number of labels.' in str(err.value)


@pytest.mark.parametrize('dataset', ['classification'], indirect=True)
@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optim.SGD,
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


@pytest.mark.parametrize('dataset', ['classification'], indirect=True)
@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optim.SGD,
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


@pytest.mark.parametrize('dataset', ['classification'], indirect=True)
@pytest.mark.parametrize('multimodal_model', [
    {
        'input_dim': input_dim,
        'output_dims': output_dims,
        'compile': True,
        'optimizer': optim.SGD,
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


@pytest.mark.parametrize('dataset', ['classification'], indirect=True)
@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optim.SGD,
        'loss': loss_fn,
        'metrics': [AccuracyMetric()]
    }
], indirect=True)
def test_compute_metrics_unimodal(unimodal_model, dataset):
    """ Test if the metric computation for an unimodal model matches the expectation. """
    y_pred = dataset['output']['unimodal']['y_pred']
    y_true = dataset['output']['unimodal']['y_true']

    result = unimodal_model.compute_metrics(y_pred=y_pred, y_true=y_true)
    expected_acc = torch.mean((y_true == torch.argmax(y_pred, dim=-1)).float()).item()
    assert np.allclose(expected_acc, result['accuracy'])


@pytest.mark.parametrize('dataset', ['classification'], indirect=True)
@pytest.mark.parametrize('multimodal_model', [
    {
        'input_dim': input_dim,
        'output_dims': output_dims,
        'compile': True,
        'optimizer': optim.SGD,
        'loss': loss_fns,
        'loss_weights': loss_weights,
        'metrics': {
            'output_1': AccuracyMetric(),
            'output_2': AccuracyMetric(),
        }
    }
], indirect=True)
def test_compute_metrics_multimodal(multimodal_model, dataset):
    """ Test if the metrics computation for a multimodal model matches the expectation. """
    y_pred = dataset['output']['multimodal']['y_pred']
    y_true = dataset['output']['multimodal']['y_true']

    results = multimodal_model.compute_metrics(y_pred=y_pred, y_true=y_true)
    expected_acc1 = torch.mean((y_true[0] == torch.argmax(y_pred[0], dim=-1)).float()).item()
    expected_acc2 = torch.mean((y_true[1] == torch.argmax(y_pred[1], dim=-1)).float()).item()
    assert np.isclose(expected_acc1, results['output_1_accuracy'])
    assert np.isclose(expected_acc2, results['output_2_accuracy'])


@pytest.mark.parametrize('dataset', ['regression'], indirect=True)
@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optim.SGD,
        'loss': nn.MSELoss(reduction='mean'),
    }
], indirect=True)
def test_train_step(unimodal_model, dataset):
    """ Test if the train step return the appropriate statistics. """
    x = dataset['input']
    y_true = dataset['output']['unimodal']['y_true']

    # compute the loss manually
    w = unimodal_model.fc1.weight.clone()
    diff = y_true - x @ w.T
    expected_loss = (torch.trace(diff @ diff.T) / (n_instances * output_dim)).item()

    # compute the gradients manually
    grad = -2 * (diff.T @ x) / (n_instances * output_dim)
    new_w = w - lr * grad

    # perform training step
    results = unimodal_model.train_step(x, y_true)

    # check if the loss and the updated weights match the manually computed ones
    assert np.isclose(results['loss'], expected_loss)
    assert torch.allclose(unimodal_model.fc1.weight, new_w)


@pytest.mark.parametrize('dataset', ['classification'], indirect=True)
@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optim.SGD,
        'loss': loss_fn,
        'metrics': [AccuracyMetric()],
    }
], indirect=True)
def test_test_step(unimodal_model, dataset):
    """ Test if the test step return the expected statistics. """
    x = dataset['input']
    y_true = dataset['output']['unimodal']['y_true']

    # reset gradients
    unimodal_model.optimizer.zero_grad()

    # perform test step
    results = unimodal_model.test_step(x, y_true)

    # check if gradients are not computed
    for param in unimodal_model.parameters():
        assert (param.grad is None) or (torch.allclose(param, torch.zeros_like(param)))

    # compute prediction
    unimodal_model.eval()
    with torch.no_grad():
        y_pred = unimodal_model(x)

    expected_loss_val = loss_fn(y_pred, y_true).item()
    expected_accuracy_val = torch.mean((torch.argmax(y_pred, dim=-1) == y_true).float()).item()
    assert np.allclose(results['loss'], expected_loss_val)
    assert np.allclose(results['accuracy'], expected_accuracy_val)


@pytest.mark.parametrize('epochs', [1, 2, 4, 10])
@pytest.mark.parametrize('batch_size', [1, 2, 4, 8, n_instances])
@pytest.mark.parametrize('dataset', ['classification'], indirect=True)
@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optim.SGD,
        'loss': loss_fn,
        'metrics': [AccuracyMetric()],
    }
], indirect=True)
def test_fit(unimodal_model, dataset, batch_size, epochs, mocker):
    """ Test if the fit function returns the expected statistics. """
    x = dataset['input']
    y = dataset['output']['unimodal']['y_true']
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    m = mocker.patch('alibi.models.pytorch.model.Model.train_step', return_value={'loss': 0., 'accuracy': 1.})
    # no need to reset the loss and the metrics since the `train_step` is mocked
    metrics_val = unimodal_model.fit(trainloader=dataloader, epochs=epochs)

    num_batches = (n_instances // batch_size) + ((n_instances % batch_size) > 0)
    num_calls = num_batches * epochs

    assert m.call_count == num_calls
    assert np.allclose(metrics_val['loss'], 0.)
    assert np.allclose(metrics_val['accuracy'], 1.)


@pytest.mark.parametrize('batch_size', [1, 2, 4, 8, n_instances])
@pytest.mark.parametrize('dataset', ['classification'], indirect=True)
@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optim.SGD,
        'loss': loss_fn,
        'metrics': [AccuracyMetric()],
    }
], indirect=True)
def test_evaluate(unimodal_model, dataset, batch_size, mocker):
    """ Test if the fit function returns the expected statistics. """
    x = dataset['input']
    y = dataset['output']['unimodal']['y_true']
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    m = mocker.patch('alibi.models.pytorch.model.Model.test_step', return_value={'loss': 0, 'accuracy': 1.})
    # no need to reset the loss and the metrics since the `test_step` is mocked
    metrics_val = unimodal_model.evaluate(testloader=dataloader)
    num_calls = (n_instances // batch_size) + ((n_instances % batch_size) > 0)

    assert m.call_count == num_calls
    assert np.allclose(metrics_val['loss'], 0)
    assert np.allclose(metrics_val['accuracy'], 1)


def test_metrics_to_str():
    """ Test the string representation of the metrics dictionary. """
    metrics = {'metric1': 0.9134, 'metric2': 0.3213}
    expected_metrics_str = "metric1: 0.9134\tmetric2: 0.3213\t"
    metric_str = Model._metrics_to_str(metrics)
    assert expected_metrics_str == metric_str


@pytest.mark.parametrize('dataset', ['classification'], indirect=True)
@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optim.SGD,
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


@pytest.mark.parametrize('dataset', ['classification'], indirect=True)
@pytest.mark.parametrize('multimodal_model', [
    {
        'input_dim': input_dim,
        'output_dims': output_dims,
        'compile': True,
        'optimizer': optim.SGD,
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


@pytest.mark.parametrize('dataset', ['classification'], indirect=True)
@pytest.mark.parametrize('unimodal_model', [
    {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'compile': True,
        'optimizer': optim.SGD,
        'loss': loss_fn,
        'metrics': [AccuracyMetric()],
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


@pytest.mark.parametrize('dataset', ['classification'], indirect=True)
@pytest.mark.parametrize('multimodal_model', [
    {
        'input_dim': input_dim,
        'output_dims': output_dims,
        'compile': True,
        'optimizer': optim.SGD,
        'loss': loss_fn,
        'loss_weights': loss_weights,
        'metrics': {
            'output_1': AccuracyMetric(),
            'output_2': AccuracyMetric(),
        }
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


def test_saving(tmp_path):
    """ Test saving functionality. """
    model1 = UnimodalModel(input_dim=input_dim, output_dim=output_dim)
    model2 = UnimodalModel(input_dim=input_dim, output_dim=output_dim)

    path = tmp_path / 'weights.pt'
    model1.save_weights(path)
    model2.load_weights(path)

    for params1, params2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(params1, params2)
