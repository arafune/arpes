"""Centralized runtime configuration manager for PyARPES."""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from dataclasses import dataclass, field
from logging import DEBUG, INFO
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib as mpl
from IPython.core.getipython import get_ipython
from IPython.core.interactiveshell import InteractiveShell

from arpes.debug import setup_logger
from arpes.helper.collections import deep_update
from arpes.helper.jupyter import generate_logfile_path
from arpes.plugin_loader import load_plugins

if TYPE_CHECKING:
    from importlib.abc import Loader

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__)


@dataclass
class ConfigManager:
    """Singleton-like managing runtime configuration, settings, and workspace for PyARPES."""

    config: dict = field(
        default_factory=lambda: {
            "WORKSPACE": {"path": "", "name": ""},
            "CURRENT_CONTEXT": None,
            "ENABLE_LOGGING": True,
            "LOGGING_STARTED": False,
            "LOGGING_FILE": None,
        },
    )

    settings: dict = field(
        default_factory=lambda: {
            "interactive": {
                "main_width": 350,
                "marginal_width": 150,
                "palette": "magma",
            },
            "use_tex": False,
        },
    )

    dataset_path: Path | None = None
    data_path: str | Path | None = None
    figure_path: Path | None = None

    def initialize(self, *, force: bool = False) -> None:
        """Initializes PyARPES configuration, logging, and plugin system.

        This should be called at the beginning of an interactive or scripting session.
        In Jupyter or marimo, it is automatically invoked during import.

        Args:
            force (bool): If True, forces re-initialization even if already initialized.

        Notes:
            - This sets up logging (e.g., IPython command logging).
            - It loads any available plugins via the plugin loader.
            - It attempts to load a `local_config` module if present.
            - Safe to call multiple times; subsequent calls do nothing unless `force=True`.
        """
        if self.config.get("_initialized", False) and not force:
            logger.debug("ConfigManager already initialized; skipping re-initialization.")
            return
        self.config["_initialized"] = True
        self.setup_logging()
        self.load_plugins()
        self.load_local_config()
        importlib.import_module("arpes.xarray_extensions")

    def load_local_config(self, module_name: str = "local_config") -> None:
        """Optionally loads a local configuration module."""
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            msg = f"Could not find {module_name}. "
            msg += "If you don't have one, you can safely ignore this message."
            warnings.warn(
                msg,
                stacklevel=2,
            )
            return
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        loader: Loader = spec.loader
        loader.exec_module(mod)

        if hasattr(mod, "CONFIG_OVERRIDES"):
            deep_update(self.config, mod.CONFIG_OVERRIDES)

        if hasattr(mod, "SETTINGS_OVERRIDES"):
            deep_update(self.settings, mod.SETTINGS_OVERRIDES)

        if hasattr(mod, "custom_init") and callable(mod.custom_init):
            mod.custom_init(self)

    def setup_logging(self) -> None:
        """Sets up logging for the current IPython session if enabled in config."""
        ipython = get_ipython()
        if not ipython or not self.config.get("ENABLE_LOGGING", False):
            return

        if isinstance(ipython, InteractiveShell) and ipython.logfile:
            self.config["LOGGING_STARTED"] = True
            self.config["LOGGING_FILE"] = ipython.logfile

        if not self.config["LOGGING_STARTED"]:
            log_path = generate_logfile_path()
            log_path.parent.mkdir(exist_ok=True)
            if isinstance(ipython, InteractiveShell):
                ipython.run_line_magic("logstart", str(log_path))
            self.config["LOGGING_FILE"] = log_path

    def load_plugins(self) -> None:
        """Loads plugins using the plugin loader."""
        load_plugins()

    def use_tex(self, *, enable: bool = False) -> None:
        """Enables or disables TeX rendering in matplotlib and updates settings.

        Does not attempt to perform any detection of an existing LaTeX installation and as a result,
        using this inappropriately can cause matplotlib to generate errors when you try to run
        standard plots.

        Args:
            enable: Whether to enable TeX.Defaults to False.
        """
        """Enables or disables TeX rendering in matplotlib and updates settings."""
        mpl.rcParams["text.usetex"] = enable
        self.settings["use_tex"] = enable

    def is_using_tex(self) -> bool:
        """Returns True if TeX rendering is enabled in matplotlib."""
        return mpl.rcParams["text.usetex"]

    def update_config_from_json(self, filename: str) -> None:
        """Updates the config dictionary from a JSON file."""
        with Path(filename).open(encoding="utf-8") as f:
            self.config.update(json.load(f))

    # --- Workspace management ---
    def enter_workspace(self, name: str) -> None:
        """Switches to a named workspace if it exists."""
        assert isinstance(self.config["WORKSPACE"], dict)
        current: dict[str, str | Path] = self.config["WORKSPACE"]
        if not current.get("path"):
            self.detect_workspace()
        base_path = Path(current["path"]).parent / name
        if not base_path.exists():
            msg = f"Could not find workspace: {name}"
            raise ValueError(msg)

        self.config["WORKSPACE"] = {"name": name, "path": str(base_path)}

    def exit_workspace(self) -> None:
        """Resets workspace to the last known config or default."""
        self.detect_workspace()

    def detect_workspace(self, path: str | Path = "") -> None:
        """Detects the workspace directory, searching up to three parent directories."""
        root = Path.cwd() if not self.dataset_path else Path(self.dataset_path)
        current = Path(path) if path else root

        for _ in range(3):
            if self._is_workspace(current):
                self.config["WORKSPACE"] = {
                    "path": str(current),
                    "name": current.name,
                }
                return
            current = current.parent

        # fallback
        self.config["WORKSPACE"] = {
            "path": str(root),
            "name": root.name,
        }

    def _is_workspace(self, path: Path) -> bool:
        """Checks if a given path is a workspace by looking for data directories."""
        return any((path / d).exists() for d in ["data", "Data"])

    @property
    def workspace_path(self) -> Path:
        """Returns the current workspace path as a Path object."""
        assert isinstance(self.config["WORKSPACE"], dict)
        return Path(self.config["WORKSPACE"]["path"])

    @property
    def workspace_name(self) -> str | Path:
        """Returns the current workspace name."""
        assert isinstance(self.config["WORKSPACE"], dict)
        return self.config["WORKSPACE"]["name"]

    def set_workspace(self, base_path: str | Path) -> None:
        """Explicitly sets the workspace path and related paths.

        Args:
            base_path (str | Path): The root path of the workspace directory.
        """
        base_path = Path(base_path).resolve()
        self.config["WORKSPACE"] = {
            "path": str(base_path),
            "name": base_path.name,
        }
        self.data_path = base_path / "data"
        self.dataset_path = base_path / "datasets"
        self.figure_path = base_path / "figures"


def is_jupyter() -> bool:
    """Returns True if the current environment is a Jupyter Notebook.

    This checks whether the IPython shell is running as ZMQInteractiveShell,
    which is the backend used by Jupyter notebooks and JupyterLab.
    """
    ip = get_ipython()
    return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"


def is_marimo() -> bool:
    """Returns True if the current Python session is running inside a marimo notebook.

    This is detected by checking whether 'marimo' is present in sys.modules.
    """
    return "marimo" in sys.modules


def should_initialize_automatically() -> bool:
    """Determines whether PyARPES should automatically initialize its configuration.

    Returns True if the current environment is Jupyter or marimo, which
    are interactive environments where users generally expect PyARPES
    to be ready to use without manual initialization.
    """
    return is_jupyter() or is_marimo()


config_manager = ConfigManager()
