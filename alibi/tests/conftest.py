import pytest


def pytest_addoption(parser):
    parser.addoption("--opt-dep", action="store")


@pytest.fixture(scope='session')
def opt_dep(request):
    """Optional dependency fixture.

    Tests that use this fixture must be run with the --opt-dep option via terminal. If not they will skip. This fixture
    is used in CI to indicate the optional dependencies installed in the tox environments the tests are run in. See
    setup.cfg and .github/workflows/ci.yml for more details.
    """
    opt_dep_value = request.config.option.opt_dep
    if opt_dep_value is None:
        pytest.skip()
    return opt_dep_value
