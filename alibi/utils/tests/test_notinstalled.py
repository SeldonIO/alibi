import pytest


def test_installed_package_works():
    package_name = "pandas"
    version = ">=0.23.4"

    try:
        import pandas as pd
    except ImportError:
        from alibi.utils.notinstalled import NotInstalledPackage
        pd = NotInstalledPackage(package_name, version=version)

    assert pd.__version__


def test_uninstsalled_package_raises():
    package_name = "thispackagedoesnotexist"
    version = "==1.2.3"

    try:
        import thispackagedoesnotexist as package
    except ImportError:
        from alibi.utils.notinstalled import NotInstalledPackage
        package = NotInstalledPackage(package_name, version=version)

    with pytest.raises(ImportError) as e:
        package.__version__
        assert package_name in str(e)
        assert version in str(e)
