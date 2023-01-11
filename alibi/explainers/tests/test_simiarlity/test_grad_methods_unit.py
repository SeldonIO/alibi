import pytest

from alibi.explainers.tests.test_simiarlity.conftest import get_flattened_model_parameters
from alibi.explainers.similarity.grad import GradientSimilarity, Task
from alibi.utils import _get_options_string


@pytest.mark.parametrize('random_cls_dataset', [({'shape': 10, 'size': 100})], indirect=True)
@pytest.mark.parametrize('linear_cls_model',
                         [
                             ({'framework': 'pytorch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True, ids=['torch-model', 'tf-model'])
def test_method_explanations(linear_cls_model, random_cls_dataset):
    """
    Test explanations run and give correct shapes for each backend.
    """
    backend, model, loss_fn, target_fn = linear_cls_model
    params = get_flattened_model_parameters(model)
    (X_train, Y_train), (_, _) = random_cls_dataset

    explainer = GradientSimilarity(
        predictor=model,
        loss_fn=loss_fn,
        sim_fn='grad_dot',
        precompute_grads=True,
        backend=backend
    )
    # test stored gradients
    explainer.fit(X_train=X_train, Y_train=Y_train)
    assert explainer.grad_X_train.shape == (len(X_train), *params.shape)
    result = explainer.explain(X_train[0])
    assert result.data['scores'].shape == (1, 100, )
    assert result.data['ordered_indices'].shape == (1, 100,)

    result = explainer.explain(X_train[0:4])
    assert result.data['scores'].shape == (4, 100, )
    assert result.data['ordered_indices'].shape == (4, 100,)


@pytest.mark.parametrize('random_cls_dataset', [({'shape': 10, 'size': 100})], indirect=True)
@pytest.mark.parametrize('linear_cls_model',
                         [
                             ({'framework': 'pytorch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True,
                         ids=['torch-model', 'tf-model']
                         )
def test_explainer_method_preprocessing(linear_cls_model, random_cls_dataset):
    """
    Preprocessing method returns correct data format for correct inputs.
    """
    backend, model, loss_fn, target_fn = linear_cls_model
    (X_train, Y_train), (_, _) = random_cls_dataset

    explainer = GradientSimilarity(
        predictor=model,
        loss_fn=loss_fn,
        backend=backend,
    )

    # test stored gradients
    explainer.fit(X_train=X_train, Y_train=Y_train)
    X, Y = explainer._preprocess_args(X_train[0:3])
    assert X.shape == (3, 10)
    assert Y.shape == (3,)

    X, Y = explainer._preprocess_args(X_train[0])
    assert X.shape == (1, 10)

    X, Y = explainer._preprocess_args(X_train[0])
    assert X.shape == (1, 10)

    grad_X_test = explainer.backend.get_grads(model, X, Y, loss_fn)
    assert grad_X_test.shape == (110, )


@pytest.mark.parametrize('linear_cls_model',
                         [
                             ({'framework': 'pytorch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True, ids=['torch-model', 'tf-model'])
def test_method_sim_fn_error_messaging(linear_cls_model):
    """
    `sim_fn` must be one of ``'grad_dot'``, ``'grad_cos'`` or ``'grad_asym_dot'``.
    """
    backend, model, loss_fn, target_fn = linear_cls_model

    # sim_fn is one of ['grad_dot', 'grad_cos']
    with pytest.raises(ValueError) as err:
        GradientSimilarity(
            predictor=model,
            loss_fn=loss_fn,
            sim_fn='not_grad_dot',
            precompute_grads=False,
            backend=backend
        )

    assert "Unknown method not_grad_dot. Consider using: 'grad_dot' | 'grad_cos' | 'grad_asym_dot'." in str(err.value)

    for sim_fn in ['grad_dot', 'grad_cos', 'grad_asym_dot']:
        GradientSimilarity(
            predictor=model,
            loss_fn=loss_fn,
            sim_fn=sim_fn,
            precompute_grads=False,
            backend=backend
        )


@pytest.mark.parametrize('linear_cls_model',
                         [
                             ({'framework': 'pytorch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True, ids=['torch-model', 'tf-model'])
def test_method_task_error_messaging(linear_cls_model):
    """
    `task` must be one of ``'classification'`` or ``'regression'``.
    """
    backend, model, loss_fn, target_fn = linear_cls_model

    # sim_fn is one of ['grad_dot', 'grad_cos']
    with pytest.raises(ValueError) as err:
        GradientSimilarity(
            predictor=model,
            loss_fn=loss_fn,
            precompute_grads=False,
            backend=backend,
            task='not_classification'
        )

    assert f"Unknown task not_classification. Consider using: {_get_options_string(Task)}." in str(err.value)

    for task in ['classification', 'regression']:
        GradientSimilarity(
            predictor=model,
            loss_fn=loss_fn,
            precompute_grads=False,
            backend=backend,
            task=task
        )


@pytest.mark.parametrize('random_cls_dataset', [({'shape':  10, 'size': 100})], indirect=True)
@pytest.mark.parametrize('linear_cls_model',
                         [
                             ({'framework': 'pytorch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True, ids=['torch-model', 'tf-model'])
def test_task_classification_input(random_cls_dataset, linear_cls_model):
    """
    Classification task explainer works when `Y_train` is `None` or a `np.ndarray`.
    """
    backend, model, loss_fn, target_fn = linear_cls_model
    (X_train, Y_train), (_, _) = random_cls_dataset

    classification_explainer = GradientSimilarity(
        predictor=model,
        loss_fn=loss_fn,
        sim_fn='grad_dot',
        task='classification',
        precompute_grads=False,
        backend=backend
    )

    # classification similarity method y value can be none or a ndarray
    classification_explainer.fit(X_train=X_train, Y_train=Y_train)
    classification_explainer.explain(X_train[0:1])
    classification_explainer.explain(X_train[0:1], Y_train[0:1])


@pytest.mark.parametrize('random_reg_dataset', [({'shape': 10, 'size': 100})], indirect=True)
@pytest.mark.parametrize('linear_reg_model',
                         [
                             ({'framework': 'pytorch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True,
                         ids=['torch-model', 'tf-model']
                         )
def test_regression_task_input(linear_reg_model, random_reg_dataset):
    """
    Regression task explainer works when `Y_train` is a ``np.ndarray``. Doesn't work when `Y_train`
    is ``None``.
    """

    backend, model, loss_fn, target_fn = linear_reg_model
    (X_train, Y_train), (_, _) = random_reg_dataset
    regression_explainer = GradientSimilarity(
        predictor=model,
        loss_fn=loss_fn,
        sim_fn='grad_dot',
        task='regression',
        precompute_grads=False,
        backend=backend
    )

    # classification similarity method y value cannot be none
    regression_explainer.fit(X_train=X_train, Y_train=Y_train)
    with pytest.raises(ValueError) as err:
        regression_explainer.explain(X_train[0])
    assert 'Regression task requires a target value.' in str(err.value)

    regression_explainer.explain(X_train[0], Y_train)


@pytest.mark.parametrize('linear_reg_model',
                         [
                             ({'framework': 'pytorch', 'input_shape': (10,), 'output_shape': 10}),
                             ({'framework': 'tensorflow', 'input_shape': (10,), 'output_shape': 10})
                         ],
                         indirect=True,
                         ids=['torch-model', 'tf-model']
                         )
def test_device_error_msgs(linear_reg_model):
    """
    Test that incorrect device type raises an error for each backend.
    """

    backend, model, loss_fn, target_fn = linear_reg_model
    with pytest.raises(TypeError) as err:
        GradientSimilarity(
            predictor=model,
            loss_fn=loss_fn,
            sim_fn='grad_dot',
            precompute_grads=False,
            backend=backend,
            device=[0]
        )
    if backend == 'pytorch':
        assert ("`device` must be a ``None``, `string`, `integer` or `torch.device` object."
               " Got <class 'list'> instead.") in str(err.value)
    elif backend == 'tensorflow':
        assert "`device` must be a `string` or ``None``. Got <class 'list'> instead." in str(err.value)
