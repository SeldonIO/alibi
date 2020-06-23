import warnings

tf_required = "tensorflow<2.0.0"
tf_upgrade = "tensorflow>2.0.0"

try:
    import tensorflow as tf

    has_tensorflow = True
    tf_version = tf.__version__

except ImportError:
    has_tensorflow = False


def assert_tensorflow_installed() -> None:
    """
    Raise an ImportError if TensorFlow is not installed.
    If TensorFlow>=2.0.0 is installed, issue a warning that some functionality may not work.
    If TensorFlow<2.0.0 is installed, issue a warning that in the future some functionality will require an upgrade.
    """

    template = "Module requires {pkg}: pip install alibi[tensorflow]"
    if not has_tensorflow:
        raise ImportError(template.format(pkg=tf_required))
    if int(tf_version[0]) > 1:
        template = "Detected tensorflow={pkg1} in the environment. Some functionality requires {pkg2}."
        warnings.warn(template.format(pkg1=tf_version, pkg2=tf_required))
    if int(tf_version[0]) < 2:
        template = "Detected tensorflow={pkg1} in the environment." \
                   "In the near future some functionality will require {pkg2}"
        warnings.warn(template.format(pkg1=tf_version, pkg2=tf_upgrade))
