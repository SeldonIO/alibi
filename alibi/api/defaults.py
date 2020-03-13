"""
This module defines the default metadata and data dictionaries for each explanation method.
"""
# Anchors
DEFAULT_META_ANCHOR = {"type": ["blackbox"],
                       "explanations": ["local"],
                       "params": {}}

DEFAULT_DATA_ANCHOR = {"anchor": [],
                       "precision": None,
                       "coverage": None,
                       "raw": None}  # type: dict

DEFAULT_DATA_ANCHOR_IMG = {"anchor": [],
                           "segments": None,
                           "precision": None,
                           "coverage": None,
                           "raw": None}  # type: dict

# CEM
DEFAULT_META_CEM = {"type": ["blackbox", "tensorflow", "keras"],
                    "explanations": ["local"],
                    "params": {}}

DEFAULT_DATA_CEM = {"PN": None,
                    "PP": None,
                    "PN_pred": None,
                    "PP_pred": None,
                    "grads_graph": None,
                    "grads_num": None,
                    "X": None,
                    "X_pred": None
                    }

# Counterfactuals
DEFAULT_META_CF = {"type": ["blackbox", "tensorflow", "keras"],
                   "explanations": ["local"],
                   "params": {}}

DEFAULT_DATA_CF = {"cf": None,
                   "all": [],
                   "orig_class": None,
                   "orig_proba": None,
                   "success": None}  # type: dict

# CFProto
DEFAULT_META_CFP = {"type": ["blackbox", "tensorflow", "keras"],
                    "explanations": ["local"],
                    "params": {}}

DEFAULT_DATA_CFP = {"cf": None,
                    "all": [],
                    "orig_class": None,
                    "orig_proba": None,
                    "id_proto": None
                    }  # type: dict
