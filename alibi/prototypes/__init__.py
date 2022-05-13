"""
The 'alibi.prototypes' modules includes prototypes and criticism selection methods.
"""

from .protoselect import ProtoSelect
from .protoselect import visualize_image_prototypes

__all__ = ['ProtoSelect',
           'visualize_image_prototypes']
