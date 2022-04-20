from .missing_optional_dependency import import_optional
from .download import spacy_model
from .data import gen_category_map
from .mapping import ohe_to_ord, ord_to_ohe
from .visualization import visualize_image_attr


DistributedExplainer = import_optional('alibi.utils.distributed', names=['DistributedExplainer'])
LanguageModel, DistilbertBaseUncased, BertBaseUncased, RobertaBase = import_optional(
    'alibi.utils.lang_model', names=['LanguageModel', 'DistilbertBaseUncased', 'BertBaseUncased', 'RobertaBase'])


__all__ = [
    'spacy_model',
    'gen_category_map',
    'ohe_to_ord',
    'ord_to_ohe',
    'visualize_image_attr',
    'DistributedExplainer',
    'LanguageModel',
    'DistilbertBaseUncased',
    'BertBaseUncased',
    'RobertaBase'
]
