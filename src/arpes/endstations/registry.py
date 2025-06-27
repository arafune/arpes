"""Registry and lookup system for ARPES endstation plugins.

This module manages the registration and resolution of endstation plugins.
Each plugin is associated with a principal name and possibly multiple aliases,
which are used to identify the appropriate class for loading and interpreting data.

It provides the following core functionality:
- Registering new endstation classes (`add_endstation`)
- Looking up endstation classes from aliases (`endstation_from_alias`)
- Resolving the correct endstation plugin for a scan (`resolve_endstation`)
- Extracting the canonical plugin name (`endstation_name_from_alias`)

This registry is typically populated automatically via plugin loading,
but plugins can also be added manually (e.g. in notebooks or extensions).
"""

from __future__ import annotations

import warnings
from logging import DEBUG, INFO
from typing import TYPE_CHECKING

from arpes.debug import setup_logger
from arpes.plugin_loader import load_plugins

if TYPE_CHECKING:
    from _typeshed import Incomplete

    from .base import EndstationBase

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


_ENDSTATION_ALIASES: dict[str, type[EndstationBase]] = {}


def endstation_from_alias(alias: str) -> type[EndstationBase]:
    """Lookup the data loading class from an alias."""
    return _ENDSTATION_ALIASES[alias]


def endstation_name_from_alias(alias: str) -> str:
    """Lookup the data loading principal location from an alias."""
    return endstation_from_alias(alias).PRINCIPAL_NAME


def add_endstation(endstation_cls: type[EndstationBase]) -> None:
    """Registers a data loading plugin (Endstation class) together with its aliases.

    You can use this to add a plugin after the original search if it is defined in another
    module or in a notebook.
    """
    assert endstation_cls.PRINCIPAL_NAME is not None
    for alias in endstation_cls.ALIASES:
        if alias in _ENDSTATION_ALIASES:
            continue

        _ENDSTATION_ALIASES[alias] = endstation_cls

    _ENDSTATION_ALIASES[endstation_cls.PRINCIPAL_NAME] = endstation_cls


def resolve_endstation(*, retry: bool = True, **kwargs: Incomplete) -> type[EndstationBase]:
    """Tries to determine which plugin to use for loading a piece of data.

    Args:
        retry (bool): Whether to attempt to reload plugins and try again after failure.
          This is used as an import guard basiscally in case the user imported things
          very strangely to ensure plugins are loaded.
        kwargs: Contains the actual information required to identify the scan.

    Returns:
        The loading plugin that should be used for the data.
    """
    endstation_name = kwargs.get("location", kwargs.get("endstation"))

    # check if the user actually provided a plugin
    if isinstance(endstation_name, type):
        return endstation_name

    if endstation_name is None:
        warnings.warn("Endstation not provided. Using `fallback` plugin.", stacklevel=2)
        endstation_name = "fallback"
    logger.debug(f"_ENDSTATION_ALIASES is : {_ENDSTATION_ALIASES}")
    try:
        return endstation_from_alias(endstation_name)
    except KeyError as key_error:
        if retry:
            logger.debug("retry with `arpes.config.load_plugins()`")

            load_plugins()
            return resolve_endstation(retry=False, **kwargs)
        msg = "Could not identify endstation. Did you set the endstation or location?"
        msg += "Find a description of the available options in the endstations module."
        raise ValueError(msg) from key_error
