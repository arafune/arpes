"""Runtime level configuration for PyARPES.

For the most part, this contains state related to:

1. UI settings
2. TeX and plotting
3. Data loading
4. Workspaces

This module also provides functions for loading configuration
in via external files, to allow better modularity between
different projects.
"""
from __future__ import annotations

import json
import logging
import os.path
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib as mpl
import pint

if TYPE_CHECKING:
    from arpes._typing import CONFIGTYPE, ConfigSettings
# pylint: disable=global-statement


ureg = pint.UnitRegistry()

DATA_PATH = None
SOURCE_ROOT = str(Path(__file__).parent)

SETTINGS: ConfigSettings = {
    "interactive": {
        "main_width": 350,
        "marginal_width": 150,
        "palette": "magma",
    },
    "xarray_repr_mod": False,
    "use_tex": False,
}

# these are all set by ``update_configuration``
DOCS_BUILD: bool = False
HAS_LOADED: bool = False
FIGURE_PATH: str | Path | None = None
DATASET_PATH: str | Path | None = None


def warn(msg: str) -> None:
    """Conditionally render a warning using `warnings.warn`."""
    if DOCS_BUILD:
        return
    warnings.warn(msg, stacklevel=2)


def update_configuration(user_path: Path | str = "") -> None:
    """Performs an update of PyARPES configuration from a module.

    This is kind of Django/flask style but is somewhat gross. Probably
    I should refactor this at some point to use a settings management library.
    The best thing might be to look at what other projects use.

    Args:
        user_path: The path to the user configuration module. Defaults to None.
                   If None is provided then this is a noop.
    """
    global HAS_LOADED  # noqa: PLW0603
    global FIGURE_PATH  # noqa: PLW0603
    global DATASET_PATH  # noqa: PLW0603
    if HAS_LOADED and not user_path:
        return
    HAS_LOADED = True
    try:
        FIGURE_PATH = Path(user_path) / "figures"
        DATASET_PATH = Path(user_path) / "datasets"
    except TypeError:
        pass


CONFIG: CONFIGTYPE = {
    "WORKSPACE": {},
    "CURRENT_CONTEXT": None,
    "ENABLE_LOGGING": True,
    "LOGGING_STARTED": False,
    "LOGGING_FILE": None,
}


class WorkspaceManager:
    """A context manager for swapping workspaces temporarily.

    This is extremely useful for loading data between separate projects
    or saving figures to different projects.

    Example:
        You can use this to load data from another named workspace:

        >>> with WorkspaceManager("another_project"):      # doctest: +SKIP
        ...    file_5_from_another_project = load_data(5)  # doctest: +SKIP
    """

    def __init__(self, workspace: str | None = None) -> None:
        """Context manager for changing workspaces temporarily. Do not instantiate directly.

        Args:
            workspace: The name of the workspace to enter temporarily. Defaults to None.
        """
        self._cached_workspace: CONFIGTYPE = {}
        self._workspace = workspace

    def __enter__(self) -> None:
        """Caches the current workspace and enters a new one.

        Raises:
            ValueError: If a workspace cannot be identified with the requested name.
        """
        global CONFIG  # noqa: PLW0602
        self._cached_workspace = CONFIG["WORKSPACE"]
        if not self._workspace:
            return
        if not CONFIG["WORKSPACE"]:
            attempt_determine_workspace()
        workspace_path = Path(CONFIG["WORKSPACE"]["path"]).parent / self._workspace
        if workspace_path.exists():
            CONFIG["WORKSPACE"] = dict(CONFIG["WORKSPACE"])
            CONFIG["WORKSPACE"]["name"] = self._workspace
            CONFIG["WORKSPACE"]["path"] = str(workspace_path)
        else:
            msg = f"Could not find workspace: {self._workspace}"
            raise ValueError(msg)

    def __exit__(self, *args: object) -> None:
        """Clean up by resetting the PyARPES workspace."""
        global CONFIG  # noqa: PLW0602
        CONFIG["WORKSPACE"] = self._cached_workspace


def workspace_matches(path: str | Path) -> bool:
    """Determines whether a given path should be treated as a workspace.

    In the past, we used to define a workspace by several conditions together, including
    the presence of an anaysis spreadsheet containing extra metadata.

    This is much simpler now: a workspace just has a data folder in it.

    Args:
        path: The path we are chekcking.

    Returns:
        True if the path is a workspace and False otherwise.
    """
    contents = os.listdir(path)
    return any(sentinel in contents for sentinel in ["data", "Data"])


def attempt_determine_workspace(current_path: str | Path = "") -> None:
    """Determines the current workspace, if working inside a workspace.

    Looks rootwards (upwards in the folder tree) for a workspace. When one is found,
    this sets the relevant configuration variables so that they are available.

    The logic for determining whether the current directory is a workspace
    has been simplified: see `workspace_matches` for more details.

    Args:
        current_path: Override for "os.getcwd". Defaults to None.
    """
    pdataset = Path.cwd() if DATASET_PATH is None else DATASET_PATH

    try:
        current_path = Path.cwd()
        for _ in range(3):
            if workspace_matches(current_path):
                CONFIG["WORKSPACE"] = {"path": current_path, "name": Path(current_path).name}
                return
            current_path = Path(current_path).parent
    except Exception:
        logging.exception("Exception occurs")
    CONFIG["WORKSPACE"] = {
        "path": pdataset,
        "name": Path(pdataset).stem,
    }


def load_json_configuration(filename: str) -> None:
    """Updates PyARPES configuration from a JSON file.

    Beware, this function performs a shallow update of the configuration.
    This can be adjusted if it turns out that there is a use case for
    nested configuration.

    Args:
        filename: A filename or path containing the settings.
    """
    with Path(filename).open() as config_file:
        CONFIG.update(json.load(config_file))


try:
    from local_config import *  # pylint: disable=wildcard-import
except ImportError:
    warn(
        "Could not find local configuration file. If you don't "
        "have one, you can safely ignore this message.",
    )


def override_settings(new_settings):
    """Deep updates/overrides PyARPES settings."""
    from arpes.utilities.collections import deep_update

    global SETTINGS  # noqa: PLW0602
    deep_update(SETTINGS, new_settings)


def load_plugins() -> None:
    """Registers plugins/data-sources in the endstations.plugin module.

    Finds all classes which represents data loading plugins in the endstations.plugin
    module and registers them.

    If you need to register a custom plugin you should just call
    `arpes.endstations.add_endstation` directly.
    """
    import importlib

    from arpes.endstations import add_endstation, plugin

    skip_modules = {"__pycache__", "__init__"}
    plugins_dir = str(Path(plugin.__file__).parent)
    modules = os.listdir(plugins_dir)
    modules = [
        m if os.path.isdir(os.path.join(plugins_dir, m)) else os.path.splitext(m)[0]
        for m in modules
        if m not in skip_modules
    ]
    for module in modules:
        try:
            loaded_module = importlib.import_module(f"arpes.endstations.plugin.{module}")
            for item in loaded_module.__all__:
                add_endstation(getattr(loaded_module, item))
        except (AttributeError, ImportError):
            pass


def is_using_tex() -> bool:
    """Whether we are configured to use LaTeX currently or not.

    Uses the values in rcParams for safety.

    Returns:
        True if matplotlib will use LaTeX for plotting and False otherwise.
    """
    return mpl.rcParams["text.usetex"]


@dataclass
class UseTex:
    """A context manager to control whether LaTeX is used when plotting.

    Internally, this uses `use_tex`, but it caches a few settings so that
    it can restore them afterwards.

    Attributes:
        use_tex: Whether to temporarily enable (use_tex=True) or disable
          (use_tex=False) TeX support in plotting.
    """

    use_tex: bool = False
    saved_context: dict[str, Any] = field(default_factory=dict)

    def __enter__(self) -> None:
        """Save old settings so we can restore them later."""
        self.saved_context["text.usetex"] = mpl.rcParams["text.usetex"]
        self.saved_context["SETTINGS.use_tex"] = SETTINGS.get("use_tex", False)
        # temporarily set the TeX configuration to the requested one
        use_tex(rc_text_should_use=self.use_tex)

    def __exit__(self, *args: object) -> None:
        """Reset configuration back to the cached settings."""
        SETTINGS["use_tex"] = self.saved_context["use_tex"]
        mpl.rcParams["text.usetex"] = self.saved_context["text.usetex"]


def use_tex(*, rc_text_should_use: bool = False) -> None:
    """Configures Matplotlib to use TeX.

    Does not attempt to perform any detection of an existing LaTeX
    installation and as a result, using this inappropriately can cause
    matplotlib to generate errors when you try to run standard plots.

    Args:
        rc_text_should_use: Whether to enable TeX. Defaults to False.
    """
    # in matplotlib v3 we do not need to change other settings unless
    # the preamble needs customization
    mpl.rcParams["text.usetex"] = rc_text_should_use
    SETTINGS["use_tex"] = rc_text_should_use


def setup_logging() -> None:
    """Configures IPython to log commands to a local folder.

    This is handled by default so that there is reproducibiity
    even in the case where the analyst has very poor Jupyter hygiene.

    This is by no means perfect. In particular, it could be improved
    substantively if anaysis products better referred to the logged record
    and not merely where they came from in the notebooks.
    """
    global CONFIG
    if HAS_LOADED:
        return
    try:
        import IPython

        ipython = IPython.get_ipython()
    except ImportError:
        return
    try:
        if ipython.logfile:
            CONFIG["LOGGING_STARTED"] = True
            CONFIG["LOGGING_FILE"] = ipython.logfile
    except AttributeError:
        return
    try:
        if CONFIG["ENABLE_LOGGING"] and not CONFIG["LOGGING_STARTED"]:
            CONFIG["LOGGING_STARTED"] = True
            from arpes.utilities.jupyter import generate_logfile_path

            log_path = generate_logfile_path()
            log_path.parent.mkdir(exist_ok=True)
            ipython.magic(f"logstart {log_path}")
            CONFIG["LOGGING_FILE"] = log_path
    except Exception:
        logging.exception("Exception occurs")


setup_logging()
update_configuration()
load_plugins()
