import pytest
from alibi.utils.missing_optional_dependency import import_optional, ERROR_TYPES, MissingDependency


class TestImportOptional:
    def setup_method(self):
        # mock missing dependency error for non existent module
        ERROR_TYPES['thispackagedoesnotexist'] = MissingDependency

    def teardown_method(self):
        # remove mock missing dependency error for other tests
        del ERROR_TYPES['thispackagedoesnotexist']

    def test_import_optional_module(self):
        requests = import_optional('requests')
        assert requests.__version__

    def test_import_optional_names(self):
        from requests import get, post
        get2, post2 = import_optional('requests', names=['get', 'post'])
        assert get2 == get
        assert get2.__name__ == 'get'
        assert post2 == post
        assert post2.__name__ == 'post'

    def test_import_optional_module_missing(self):
        package = import_optional('thispackagedoesnotexist')
        with pytest.raises(ImportError):
            package.__version__  # noqa

    def test_import_optional_name_missing(self):
        nonexistent_function = import_optional('thispackagedoesnotexist', names=['nonexistent_function'])
        with pytest.raises(ImportError):
            nonexistent_function.__version__  # noqa

    def test_import_optional_names_missing(self):
        nonexistent_function_1, nonexistent_function_2 = import_optional(
            'thispackagedoesnotexist',
            names=['nonexistent_function_1', 'nonexistent_function_2'])
        with pytest.raises(ImportError):
            nonexistent_function_1.__version__  # noqa

        with pytest.raises(ImportError):
            nonexistent_function_2.__version__  # noqa
