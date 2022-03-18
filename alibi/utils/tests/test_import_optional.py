from alibi.utils.missing_optional_dependency import import_optional
import pytest


def test_import_optional_module():
    requests = import_optional('requests')
    assert requests.__version__


def test_import_optional_names():
    from requests import get, post
    get2, post2 = import_optional('requests', names=['get', 'post'])
    assert get2 == get
    assert get2.__name__ == 'get'
    assert post2 == post
    assert post2.__name__ == 'post'


def test_import_optional_module_missing():
    package = import_optional('thispackagedoesnotexist')
    with pytest.raises(ImportError):
        package.__version__  # noqa


def test_import_optional_name_missing():
    nonexistent_function = import_optional('thispackagedoesnotexist', names=['nonexistent_function'])
    with pytest.raises(ImportError):
        nonexistent_function.__version__  # noqa


def test_import_optional_names_missing():
    nonexistent_function_1, nonexistent_function_2 = import_optional(
        'thispackagedoesnotexist',
        names=['nonexistent_function_1', 'nonexistent_function_2'])
    with pytest.raises(ImportError):
        nonexistent_function_1.__version__  # noqa

    with pytest.raises(ImportError):
        nonexistent_function_2.__version__  # noqa
