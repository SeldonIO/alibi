"""
This module defines the default metadata and data dictionaries for each explanation method.
Note that the "name" field is automatically populated upon initialization of the corresponding
Explainer class.
"""
# Anchors
DEFAULT_META_ANCHOR = {"name": None,
                       "type": ["blackbox"],
                       "explanations": ["local"],
                       "params": {}}
"""
Default anchor metadata.
"""

DEFAULT_DATA_ANCHOR = {"anchor": [],
                       "precision": None,
                       "coverage": None,
                       "raw": None}  # type: dict
"""
Default anchor data.
"""

DEFAULT_DATA_ANCHOR_IMG = {"anchor": [],
                           "segments": None,
                           "precision": None,
                           "coverage": None,
                           "raw": None}  # type: dict
"""
Default anchor image data.
"""

# CEM
DEFAULT_META_CEM = {"name": None,
                    "type": ["blackbox", "tensorflow", "keras"],
                    "explanations": ["local"],
                    "params": {}}
"""
Default CEM metadata.
"""

DEFAULT_DATA_CEM = {"PN": None,
                    "PP": None,
                    "PN_pred": None,
                    "PP_pred": None,
                    "grads_graph": None,
                    "grads_num": None,
                    "X": None,
                    "X_pred": None
                    }
"""
Default CEM data.
"""

# Counterfactuals
DEFAULT_META_CF = {"name": None,
                   "type": ["blackbox", "tensorflow", "keras"],
                   "explanations": ["local"],
                   "params": {}}
"""
Default counterfactual metadata.
"""

DEFAULT_DATA_CF = {"cf": None,
                   "all": [],
                   "orig_class": None,
                   "orig_proba": None,
                   "success": None}  # type: dict
"""
Default counterfactual data.
"""

# CFProto
DEFAULT_META_CFP = {"name": None,
                    "type": ["blackbox", "tensorflow", "keras"],
                    "explanations": ["local"],
                    "params": {}}
"""
Default counterfactual prototype metadata.
"""

DEFAULT_DATA_CFP = {"cf": None,
                    "all": [],
                    "orig_class": None,
                    "orig_proba": None,
                    "id_proto": None
                    }  # type: dict
"""
Default counterfactual prototype metadata.
"""

# KernelSHAP
DEFAULT_META_SHAP = {
    "name": None,
    "type": ["blackbox"],
    "explanations": ["local", "global"],
    "params": {}
}  # type: dict
"""
Default KernelSHAP metadata.
"""

DEFAULT_DATA_SHAP = {
    "shap_values": [],
    "expected_value": [],
    "link": 'identity',
    "categorical_names": None,
    "feature_names": None,
    "raw": {
        "raw_prediction": None,
        "prediction": None,
        "instances": None,
        "importances": {},
    }
}  # type: dict
"""
Default KernelSHAP data.
"""
