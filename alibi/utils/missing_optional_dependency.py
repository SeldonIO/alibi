class MissingOptionalDependency:
    """
    Class to gracefully catch ImportErrors for modules and packages that are not installed
    Usage:
        >>> try:
        ...     from alibi.explainers import AnchorText
        >>> except ImportError as err:
        ...     from alibi.utils.missing_optional_dependency import MissingOptionalDependency
        ...     explainer = MissingOptionalDependency(err, 'AnchorText', install_option='transformers')
    """

    def __init__(self, error: ImportError, name: str, install_option: str = 'all'):
        """

        Parameters
        ----------
        name:
            Name of the object or function that cannot be imported due to uninstalled optional dependencies.
        install_option:
            Missing optional dependency required for the imported functionality.
        """
        self.__name__ = name
        self.error = error
        self.install_option = install_option
        install_opts_msg = f"`python -m pip install alibi[{self.install_option}].`"

        self.pip_message = (
            (
                    f"Attempted to use {self.__name__} without the correct optional dependencies installed. To install "
                    + f"the correct optional dependencies, \nrun {install_opts_msg} from the command line. For more "
                    + "information, check the 'Dependency installs' section of the \ninstallation docs at "
                    + "https://docs.seldon.io/projects/alibi/en/latest/overview/getting_started.html."
            )
        )

    def __getattr__(self, name):
        raise ImportError(self.pip_message) from self.error
