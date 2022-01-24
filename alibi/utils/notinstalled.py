KNOWN_PACKAGES = {"cvxpy": {"version": ">=1.0.24", "extra_name": "cvxpy"}}


class NotInstalledPackage:
    """
    Class to gracefully catch ImportErrors for modules and packages that are not installed
    :param package_name (str): Name of the package you want to load
    :param version (str, Optional): Version of the package
    Usage:
        >>> try:
        ...     import thispackagedoesnotexist as package
        >>> except ImportError:
        ...     from alibi.utils.notinstalled import NotInstalledPackage
        ...     package = NotInstalledPackage("thispackagedoesnotexist")
    """

    def __init__(self, package_name: str, version: str = None, modality: str = None):
        self.package_name = package_name
        package_info = KNOWN_PACKAGES.get(package_name, {})
        self.version = version if version else package_info.get("version", "")
        self.modality = modality

        extra_name = package_info.get("extra_name", None)
        if modality:
            install_opts_msg = (
                    f"`python -m pip install alibi[{extra_name}]`, or "
                    + f"`python -m pip install alibi[{modality}].`"
            )
        else:
            install_opts_msg = f"`python -m pip install alibi[{extra_name}].`"

        self.pip_message = (
            (
                f"Install extra requirement {package_name} using "
                + install_opts_msg
                + "For more information, check the 'Dependency installs' section of the installation docs at "
                + "https://docs.seldon.io/projects/alibi/en/latest/overview/getting_started.html"
            )
            if extra_name
            else ""
        )

    def __getattr__(self, name):
        raise ImportError(
            f"The package {self.package_name}{self.version} is not installed. "
            + self.pip_message
        )
