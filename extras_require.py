"""

Notes:
    shap is separated due to build issues, see https://github.com/slundberg/shap/pull/1802
    tensorflow issue: # https://github.com/SeldonIO/alibi-detect/issues/375
    # versioning: https://github.com/SeldonIO/alibi/issues/333
    we avoic numba 0.54 due to: https://github.com/SeldonIO/alibi/issues/466
"""

extra_pkgs = {
    'ray': [
        ('ray', '>=0.8.7, <2.0.0')
    ],
    'tensorflow': [
        ('tensorflow', '>=2.0.0, !=2.6.0, !=2.6.1, <2.8.0')
    ],
    'torch': [
        ('torch', '>=1.9.0, <2.0.0')
    ],
    'shap': [
        ('shap', '>=0.40.0, <0.41.0'),
        ('numba', '>=0.50.0, !=0.54.0, <0.56.0'),
    ],
}


def extras_require():
    """
    Returns a dictionary of extra packages required for each package.
    """
    return {key: [''.join(dep) for dep in val] for key, val in extra_pkgs.items()}


