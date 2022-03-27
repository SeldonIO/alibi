from alibi.utils.missing_optional_dependency import import_optional
from alibi.explainers.backends.cfrl_tabular_shared import get_he_preprocessor, get_statistics, \
    get_conditional_vector, apply_category_mapping

pytorch_base_backend = import_optional('alibi.explainers.backends.pytorch.cfrl_base')
pytorch_tabular_backend = import_optional('alibi.explainers.backends.pytorch.cfrl_tabular')
tensorflow_base_backend = import_optional('alibi.explainers.backends.tensorflow.cfrl_base')
tensorflow_tabular_backend = import_optional('alibi.explainers.backends.tensorflow.cfrl_tabular')


__all__ = [
    'get_he_preprocessor',
    'get_statistics',
    'get_conditional_vector',
    'apply_category_mapping',
    'pytorch_base_backend',
    'pytorch_tabular_backend',
    'tensorflow_base_backend',
    'tensorflow_tabular_backend',
]
