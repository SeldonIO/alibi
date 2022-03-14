import pytest


def test_installed_package_works():
    try:
        import requests
    except ImportError as err:
        from alibi.utils.notinstalled import NotInstalledPackage
        requests = NotInstalledPackage(err, 'requests', install_option='requests')
    assert requests.__version__


def test_uninstalled_package_raises():
    try:
        import thispackagedoesnotexist as package
    except ImportError as err:
        from alibi.utils.notinstalled import NotInstalledPackage
        package = NotInstalledPackage(err, 'test', install_option='test')

    with pytest.raises(ImportError) as e:
        package.__version__  # noqa
