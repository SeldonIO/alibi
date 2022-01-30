import pytest


def test_installed_package_works():
    package_name = "requests"
    version = ">=2.21.0, <3.0.0"

    try:
        import requests
    except ImportError:
        from alibi.utils.notinstalled import NotInstalledPackage
        requests = NotInstalledPackage(package_name, version=version)

    assert requests.__version__


def test_uninstalled_package_raises():
    package_name = "this_package_does_not_exist"
    version = "==1.2.3"

    try:
        import thispackagedoesnotexist as package
    except ImportError:
        from alibi.utils.notinstalled import NotInstalledPackage
        package = NotInstalledPackage(package_name, version=version)

    with pytest.raises(ImportError) as e:
        package.__version__  # noqa
        assert package_name in str(e)
        assert version in str(e)
