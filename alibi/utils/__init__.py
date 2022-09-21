from enum import Enum
from typing import Type

from .data import gen_category_map
from .download import spacy_model
from .mapping import ohe_to_ord, ord_to_ohe
from .missing_optional_dependency import import_optional
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


# Miscellanious private utilities used internally
def _get_options_string(enum: Type[Enum]) -> str:
    """Get the enums options seperated by pipe as a string.
    Note: this only works on enums inheriting from `str`, i.e. class MyEnum(str, Enum).
    Note: Python 3.11 will introduce enum.StrEnum which will be the preferred type for string enumerations.
    If we want finer control over typing we could define a new type."""
    return f"""'{"' | '".join(enum)}'"""
