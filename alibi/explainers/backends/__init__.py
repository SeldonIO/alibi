from alibi.utils.missing_optional_dependency import import_optional

pytorch_base_backend = import_optional('alibi.explainers.backends.pytorch.cfrl_base')
pytorch_tabular_backend = import_optional('alibi.explainers.backends.pytorch.cfrl_tabular')
tensorflow_base_backend = import_optional('alibi.explainers.backends.tensorflow.cfrl_base')
tensorflow_tabular_backend = import_optional('alibi.explainers.backends.tensorflow.cfrl_tabular')
