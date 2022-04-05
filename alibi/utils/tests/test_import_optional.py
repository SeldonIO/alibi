import pytest

from alibi.utils.missing_optional_dependency import import_optional, ERROR_TYPES, MissingDependency


class TestImportOptional:
    """Test the import_optional function."""

    def setup_method(self):
        # mock missing dependency error for non existent module
        ERROR_TYPES['thispackagedoesnotexist'] = MissingDependency

    def teardown_method(self):
        # remove mock missing dependency error for other tests
        del ERROR_TYPES['thispackagedoesnotexist']

    def test_import_optional_module(self):
        """Test import_optional correctly imports installed module."""
        requests = import_optional('requests')
        assert requests.__version__

    def test_import_optional_names(self):
        """Test import_optional correctly imports names from installed module."""
        from requests import get, post
        get2, post2 = import_optional('requests', names=['get', 'post'])
        assert get2 == get
        assert get2.__name__ == 'get'
        assert post2 == post
        assert post2.__name__ == 'post'

    def test_import_optional_module_missing(self):
        """Test import_optional correctly replaces module that doesn't exist with MissingDependency."""
        package = import_optional('thispackagedoesnotexist')
        assert isinstance(package, MissingDependency)
        with pytest.raises(ImportError) as err:
            package.__version__  # noqa
        assert 'thispackagedoesnotexist' in str(err.value)
        assert 'pip install alibi[all]' in str(err.value)

        with pytest.raises(ImportError) as err:
            package(0, 'test')  # noqa
        assert 'thispackagedoesnotexist' in str(err.value)
        assert 'pip install alibi[all]' in str(err.value)

    def test_import_optional_names_missing(self):
        """Test import_optional correctly replaces names from module that doesn't exist with MissingDependencies."""
        nonexistent_function_1, nonexistent_function_2 = import_optional(
            'thispackagedoesnotexist',
            names=['nonexistent_function_1', 'nonexistent_function_2'])
        assert isinstance(nonexistent_function_1, MissingDependency)
        with pytest.raises(ImportError) as err:
            nonexistent_function_1.__version__  # noqa
        assert 'nonexistent_function_1' in str(err.value)
        assert 'pip install alibi[all]' in str(err.value)

        assert isinstance(nonexistent_function_2, MissingDependency)
        with pytest.raises(ImportError) as err:
            nonexistent_function_2.__version__  # noqa
        assert 'nonexistent_function_2' in str(err.value)
        assert 'pip install alibi[all]' in str(err.value)
