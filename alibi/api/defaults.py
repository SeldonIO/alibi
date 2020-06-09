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
DEFAULT_META_KERNEL_SHAP = {
    "name": None,
    "type": ["blackbox"],
    "task": None,
    "explanations": ["local", "global"],
    "params": {}
}  # type: dict
"""
Default KernelSHAP metadata.
"""

DEFAULT_DATA_KERNEL_SHAP = {
    "shap_values": [],
    "expected_value": [],
    "link": 'identity',
    "categorical_names": {},
    "feature_names": [],
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

# ALE
DEFAULT_META_ALE = {
    "name": None,
    "type": ["blackbox"],
    "explanations": ["global"],
    "params": {}
}  # type: dict
"""
Default ALE metadata.
"""

DEFAULT_DATA_ALE = {
    "ale_values": [],
    "constant_value": None,
    "ale0": [],
    "feature_values": [],
    "feature_names": None,
    "target_names": None,
    "feature_deciles": None
}  # type: dict
"""
Default ALE data.
"""

# TreeShap
DEFAULT_META_TREE_SHAP = {
    "name": None,
    "type": ["whitebox"],
    "task": None,  # updates with 'classification' or 'regression'
    "explanations": ["local", "global"],
    "params": {}
}  # type: dict
"""
Default TreeSHAP metadata.
"""

DEFAULT_DATA_TREE_SHAP = {
    "shap_values": [],
    "shap_interaction_values": [],
    "expected_value": [],
    "model_output": None,  # see TreeShap __init__ for possible values
    "categorical_names": {},
    "feature_names": [],
    "raw": {
        "raw_prediction": None,
        "loss": None,
        "prediction": None,
        "instances": None,
        "labels": None,
        "importances": {},
    }
}  # type: dict

"""
Default TreeSHAP data.
"""

# Integrated gradients
DEFAULT_META_INTGRAD = {
    "name": None,
    "type": ["whitebox"],
    "explanations": ["local"],
    "params": {}
}  # type: dict
"""
Default IntegratedGradients metadata.
"""

DEFAULT_DATA_INTGRAD = {
    "attributions": None,
    "X": None,
    "baselines": None,
    "predictions": None,
    "deltas": None
}  # type: dict
"""
Default IntegratedGradients data.
"""
