"""
This module defines the default metadata and data dictionaries for each explanation method.
Note that the "name" field is automatically populated upon initialization of the corresponding
Explainer class.
"""
# Anchors
DEFAULT_META_ANCHOR: dict = {
    "name": None,
    "type": ["blackbox"],
    "explanations": ["local"],
    "params": {},
    "version": None
}
"""
Default anchor metadata.
"""

DEFAULT_DATA_ANCHOR: dict = {
    "anchor": [],
    "precision": None,
    "coverage": None,
    "raw": None
}
"""
Default anchor data.
"""

DEFAULT_DATA_ANCHOR_IMG: dict = {
    "anchor": [],
    "segments": None,
    "precision": None,
    "coverage": None,
    "raw": None
}
"""
Default anchor image data.
"""

# CEM
DEFAULT_META_CEM: dict = {
    "name": None,
    "type": ["blackbox", "tensorflow", "keras"],
    "explanations": ["local"],
    "params": {},
    "version": None
}
"""
Default CEM metadata.
"""
DEFAULT_DATA_CEM: dict = {
    "PN": None,
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
DEFAULT_META_CF: dict = {
    "name": None,
    "type": ["blackbox", "tensorflow", "keras"],
    "explanations": ["local"],
    "params": {},
    "version": None
}
"""
Default counterfactual metadata.
"""

DEFAULT_DATA_CF: dict = {
    "cf": None,
    "all": [],
    "orig_class": None,
    "orig_proba": None,
    "success": None
}
"""
Default counterfactual data.
"""

# CFProto
DEFAULT_META_CFP: dict = {
    "name": None,
    "type": ["blackbox", "tensorflow", "keras"],
    "explanations": ["local"],
    "params": {},
    "version": None
}
"""
Default counterfactual prototype metadata.
"""

DEFAULT_DATA_CFP: dict = {
    "cf": None,
    "all": [],
    "orig_class": None,
    "orig_proba": None,
    "id_proto": None
}
"""
Default counterfactual prototype metadata.
"""

# KernelSHAP
KERNEL_SHAP_PARAMS = [
    'link',
    'group_names',
    'grouped',
    'groups',
    'weights',
    'summarise_background',
    'summarise_result',
    'transpose',
    'kwargs',
]
"""
KernelShap parameters updated and returned in ``metadata['params']``.
See :py:class:`alibi.explainers.shap_wrappers.KernelShap`.
"""

DEFAULT_META_KERNEL_SHAP: dict = {
    "name": None,
    "type": ["blackbox"],
    "task": None,
    "explanations": ["local", "global"],
    "params": dict.fromkeys(KERNEL_SHAP_PARAMS),
    "version": None
}
"""
Default KernelShap metadata.
"""

DEFAULT_DATA_KERNEL_SHAP: dict = {
    "shap_values": [],
    "expected_value": [],
    "categorical_names": {},
    "feature_names": [],
    "raw": {
        "raw_prediction": None,
        "prediction": None,
        "instances": None,
        "importances": {},
    }
}
"""
Default KernelShap data.
"""

# ALE
DEFAULT_META_ALE: dict = {
    "name": None,
    "type": ["blackbox"],
    "explanations": ["global"],
    "params": {},
    "version": None
}
"""
Default ALE metadata.
"""

DEFAULT_DATA_ALE: dict = {
    "ale_values": [],
    "constant_value": None,
    "ale0": [],
    "feature_values": [],
    "feature_names": None,
    "target_names": None,
    "feature_deciles": None
}
"""
Default ALE data.
"""

# TreeShap
TREE_SHAP_PARAMS = [
    'model_output',
    'summarise_background',
    'summarise_result',
    'approximate',
    'interactions',
    'explain_loss',
    'algorithm',
    'kwargs'
]
"""
TreeShap parameters updated and returned in ``metadata['params']``.
See :py:class:`alibi.explainers.shap_wrappers.TreeShap`.
"""

DEFAULT_META_TREE_SHAP: dict = {
    "name": None,
    "type": ["whitebox"],
    "task": None,  # updates with 'classification' or 'regression'
    "explanations": ["local", "global"],
    "params": dict.fromkeys(TREE_SHAP_PARAMS),
    "version": None
}
"""
Default TreeShap metadata.
"""

DEFAULT_DATA_TREE_SHAP: dict = {
    "shap_values": [],
    "shap_interaction_values": [],
    "expected_value": [],
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
}

"""
Default TreeShap data.
"""

# Integrated gradients
DEFAULT_META_INTGRAD: dict = {
    "name": None,
    "type": ["whitebox"],
    "explanations": ["local"],
    "params": {},
    "version": None
}
"""
Default IntegratedGradients metadata.
"""

DEFAULT_DATA_INTGRAD: dict = {
    "attributions": None,
    "X": None,
    "forward_kwargs": None,
    "baselines": None,
    "predictions": None,
    "deltas": None
}
"""
Default IntegratedGradients data.
"""

DEFAULT_META_CFRL: dict = {
    "name": None,
    "type": ["blackbox"],
    "explanations": ["local"],
    "params": {},
    "version": None
}
"""
Default CounterfactualRL metadata.
"""

DEFAULT_DATA_CFRL: dict = {
    "orig": None,
    "cf": None,
    "target": None,
    "condition": None
}
"""
Default CounterfactualRL data.
"""

# Similarity methods
DEFAULT_META_SIM: dict = {
    "name": None,
    "type": ["whitebox"],
    "explanations": ["local"],
    "params": {},
    "version": None
}
"""
Default SimilarityExplainer metadata.
"""

DEFAULT_DATA_SIM: dict = {
    "scores": None,
    "ordered_indices": None,
    "most_similar": None,
    "least_similar": None
}
"""
Default SimilarityExplainer data.
"""

DEFAULT_META_PROTOSELECT: dict = {
    "name": None,
    "type": ["data"],
    "explanation": ["global"],
    "params": {},
    "version": None
}
"""
Default ProtoSelect metadata.
"""

DEFAULT_DATA_PROTOSELECT: dict = {
    "prototypes": None,
    "prototype_indices": None,
    "prototype_labels": None
}
"""
Default ProtoSelect data.
"""

# PartialDependence
DEFAULT_META_PD: dict = {
    "name": None,
    "type": ["blackbox"],
    "explanations": ["global"],
    "params": {},
    "version": None
}
"""
Default PartialDependence metadata.
"""

DEFAULT_DATA_PD: dict = {
    "feature_deciles": None,
    "pd_values": None,
    "ice_values": None,
    "feature_values": None,
    "feature_names": None,
}
"""
Default PartialDependence data.
"""

# PartialDependenceVariance
DEFAULT_META_PDVARIANCE: dict = {
    "name": None,
    "type": ["blackbox"],
    "explanations": ["global"],
    "params": {},
    "version": None
}
"""
Default PartialDependenceVariance metadata.
"""

DEFAULT_DATA_PDVARIANCE: dict = {
    "feature_deciles": None,
    "pd_values": None,
    "feature_values": None,
    "feature_names": None,
    "feature_importance": None,
    "feature_interaction": None,
    "conditional_importance": None,
    "conditional_importance_values": None
}
"""
Default PartialDependenceVariance data.
"""

# PermutationImportance
DEFAULT_META_PERMUTATION_IMPORTANCE: dict = {
    "name": None,
    "type": ["blackbox"],
    "explanations": ["global"],
    "params": {},
    "version": None
}
"""
Default PermutationImportance metadata.
"""

DEFAULT_DATA_PERMUTATION_IMPORTANCE: dict = {
    "feature_names": None,
    "metric_names": None,
    "feature_importance": None,
}
"""
Default PermutationImportance data.
"""
