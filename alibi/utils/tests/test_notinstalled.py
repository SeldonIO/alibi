import pytest


def test_installed_package_works():
    try:
        import requests
    except ImportError as err:
        from alibi.utils.missing_optional_dependency import MissingOptionalDependency
        requests = MissingOptionalDependency(err, 'requests', install_option='requests')
    assert requests.__version__


def test_uninstalled_package_raises():
    try:
        import thispackagedoesnotexist as package
    except ImportError as err:
        from alibi.utils.missing_optional_dependency import MissingOptionalDependency
        package = MissingOptionalDependency(err, 'test', install_option='test')

    with pytest.raises(ImportError):
        package.__version__  # noqa
