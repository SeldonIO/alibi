# `alibi.models.tensorflow.autoencoder`

This module contains a Tensorflow general implementation of an autoencoder, by combining the encoder and the decoder
module. In addition it provides an implementation of a heterogeneous autoencoder which includes a type checking of the
output.

## `AE`

_Inherits from:_ `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

Autoencoder. Standard autoencoder architecture. The model is composed from two submodules, the encoder and

the decoder. The forward pass consists of passing the input to the encoder, obtain the input embedding and
pass the embedding through the decoder. The abstraction can be used for multiple data modalities.

### Constructor

```python
AE(self, encoder: keras.src.models.model.Model, decoder: keras.src.models.model.Model, **kwargs) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `encoder` | `keras.src.models.model.Model` |  |  |
| `decoder` | `keras.src.models.model.Model` |  |  |

### Methods

#### `call`

```python
call(x: tensorflow.python.framework.tensor.Tensor, kwargs) -> Union[tensorflow.python.framework.tensor.Tensor, List[tensorflow.python.framework.tensor.Tensor]]
```

Forward pass.

Parameters
----------
x
    Input tensor.
**kwargs
    Other arguments passed to encoder/decoder `call` method.

Returns
-------
x_hat
    Reconstruction of the input tensor.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  |  |

**Returns**
- Type: `Union[tensorflow.python.framework.tensor.Tensor, List[tensorflow.python.framework.tensor.Tensor]]`

## `HeAE`

_Inherits from:_ `AE`, `Model`, `TensorFlowTrainer`, `Trainer`, `Layer`, `TFLayer`, `KerasAutoTrackable`, `AutoTrackable`, `Trackable`, `Operation`, `KerasSaveable`

Heterogeneous autoencoder. The model follows the standard autoencoder architecture and includes and additional

type check to ensure that the output of the model is a list of tensors. For more details, see
:py:class:`alibi.models.pytorch.autoencoder.AE`.

### Constructor

```python
HeAE(self, encoder: keras.src.models.model.Model, decoder: keras.src.models.model.Model, **kwargs) -> None
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `encoder` | `keras.src.models.model.Model` |  |  |
| `decoder` | `keras.src.models.model.Model` |  |  |

### Methods

#### `build`

```python
build(input_shape: Tuple[int, .Ellipsis]) -> None
```

Build method.

Parameters
----------
input_shape
    Tensor's input shape.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `input_shape` | `Tuple[int, .Ellipsis]` |  |  |

**Returns**
- Type: `None`

#### `call`

```python
call(x: tensorflow.python.framework.tensor.Tensor, kwargs) -> List[tensorflow.python.framework.tensor.Tensor]
```

Forward pass.

Parameters
----------
x
    Input tensor.
**kwargs
    Other arguments passed to the encoder/decoder.

Returns
--------
List of reconstruction of the input tensor. First element corresponds to the reconstruction of all the         numerical features if they exist, and the rest of the elements correspond to each categorical feature.

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `tensorflow.python.framework.tensor.Tensor` |  |  |

**Returns**
- Type: `List[tensorflow.python.framework.tensor.Tensor]`
