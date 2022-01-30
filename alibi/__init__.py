import sys


class ManageOptionalDependencies:
    @classmethod
    def find_spec(cls, name, path, target=None):
        print(f"Importing {name!r}")
        # wrap not found modules here
        return None


sys.meta_path.insert(0, ManageOptionalDependencies)


from . import confidence, datasets, explainers, utils
from .version import __version__  # noqa F401

__all__ = ['confidence', 'datasets', 'explainers', 'utils']
