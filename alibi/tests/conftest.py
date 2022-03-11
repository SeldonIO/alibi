import pytest


def pytest_addoption(parser):
    parser.addoption("--opt-dep", action="store")


@pytest.fixture(scope='session')
def opt_dep(request):
    opt_dep_value = request.config.option.opt_dep
    if opt_dep_value is None:
        pytest.skip()
    return opt_dep_value