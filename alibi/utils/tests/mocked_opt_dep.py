import non_existent_module  # noqa: F401


class MockedClassWithoutRequiredDeps:
    def __init__(self):
        self.opt_dep = "opt_dep"


def mocked_function_without_required_deps():
    pass
