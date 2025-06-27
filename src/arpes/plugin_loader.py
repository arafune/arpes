"""Module for dynamic plugin loading in the arpes package.

This module defines functions related to discovering and initializing
external or internal plugins that extend the functionality of the arpes system.
It is separated from `config.py` to avoid circular import issues.

Typical usage:
    from arpes.plugin_loader import load_plugins

Functions:
    - load_plugins(): Discover and register available plugins.
"""

import importlib
from logging import DEBUG, INFO
from pathlib import Path

from .debug import setup_logger

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, level=LOGLEVEL)


def load_plugins() -> None:
    """Registers plugins/data-sources in the endstations.plugin module.

    Finds all classes which represents data loading plugins in the endstations.plugin
    module and registers them.

    If you need to register a custom plugin you should just call
    `arpes.endstations.add_endstation` directly.
    """
    from .endstations import add_endstation, plugin  # noqa: PLC0415

    skip_modules = {"__pycache__", "__init__"}
    plugins_dir = Path(plugin.__file__).parent
    modules = [
        str(m) if Path(plugins_dir / m).is_dir() else str(Path(m).stem)
        for m in Path(plugins_dir).iterdir()
        if m.stem not in skip_modules
    ]
    logger.debug(f"modules are {modules}")
    for module in modules:
        try:
            loaded_module = importlib.import_module(f"arpes.endstations.plugin.{module}")
            for item in loaded_module.__all__:
                add_endstation(getattr(loaded_module, item))
        except (AttributeError, ImportError):
            pass
