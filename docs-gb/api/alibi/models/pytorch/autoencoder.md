# `alibi.models.pytorch.autoencoder`

This module contains a Pytorch general implementation of an autoencoder, by combining the encoder and the decoder
module. In addition it provides an implementation of a heterogeneous autoencoder which includes a type checking of the
output.

## `AE`

_Inherits from:_ `Model`, `Module`

Autoencoder. Standard autoencoder architecture. The model is composed from two submodules, the encoder and
the decoder. The forward pass consist of passing the input to the encoder, obtain the input embedding and
pass the embedding through the decoder. The abstraction can be used for multiple data modalities.

### Constructor

```python
AE(self, encoder: torch.nn.modules.module.Module, decoder: torch.nn.modules.module.Module, **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `encoder` | `torch.nn.modules.module.Module` |  | Encoder network. |
| `decoder` | `torch.nn.modules.module.Module` |  | Decoder network. |

### Methods

#### `forward`

```python
forward(x: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor]]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | Input tensor. |

**Returns**
- Type: `Union[torch.Tensor, List[torch.Tensor]]`

## `HeAE`

_Inherits from:_ `AE`, `Model`, `Module`

Heterogeneous autoencoder. The model follows the standard autoencoder architecture and includes and additional
type check to ensure that the output of the model is a list of tensors. For more details, see
:py:class:`alibi.models.pytorch.autoencoder.AE`.

### Constructor

```python
HeAE(self, encoder: torch.nn.modules.module.Module, decoder: torch.nn.modules.module.Module, **kwargs)
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `encoder` | `torch.nn.modules.module.Module` |  | Encoder network. |
| `decoder` | `torch.nn.modules.module.Module` |  | Decoder network. |

### Methods

#### `forward`

```python
forward(x: torch.Tensor) -> List[torch.Tensor]
```

| Name | Type | Default | Description |
| ---- | ---- | ------- | ----------- |
| `x` | `torch.Tensor` |  | Input tensor. |

**Returns**
- Type: `List[torch.Tensor]`
