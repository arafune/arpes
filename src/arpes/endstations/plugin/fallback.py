"""Implements dynamic plugin selection when users do not specify the location for their data."""

from __future__ import annotations

import warnings
from logging import DEBUG, INFO
from typing import TYPE_CHECKING, ClassVar

from arpes.config import load_plugins
from arpes.debug import setup_logger
from arpes.endstations import EndstationBase, resolve_endstation

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr
    from _typeshed import Incomplete

    from arpes.endstations import ScanDesc
__all__ = ("FallbackEndstation",)

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


AUTOLOAD_WARNING = (
    "PyARPES has chosen {} for your data since `location` was not specified. "
    "If this is not correct, ensure the `location` key is specified. Read the plugin documentation "
    "for more details."
)


class FallbackEndstation(EndstationBase):
    """Sequentially tries different loading plugins.

    Different from the rest of the data loaders. This one is used when there is no location
    specified and attempts sequentially to call a variety of standard plugins until one is found
    that works.
    """

    PRINCIPAL_NAME = "fallback"
    ALIASES: ClassVar[list[str]] = []

    ATTEMPT_ORDER: ClassVar[list[str]] = [
        "ANTARES",
        "MBS",
        "ALS-BL7",
        "ALS-BL403",
        "Igor",
        "Kaindl",
        "ALG-Main",
        "ALG-SToF",
        "SPD",
    ]

    @classmethod
    def determine_associated_loader(
        cls: type[FallbackEndstation],
        file: str | Path,
    ) -> type[EndstationBase]:
        """Determines which loading plugin to use for a given piece of data.

        This is done by looping through loaders in a predetermined priority order,
        and asking each whether it is capable of loading the file.
        """
        load_plugins()

        for location in cls.ATTEMPT_ORDER:
            logger.debug(f"{cls.__name__} is trying {location}")

            try:
                endstation_cls = resolve_endstation(retry=False, location=location)
                if endstation_cls.is_file_accepted(file):
                    return endstation_cls
            except BaseException:
                logger.exception("Exception occurs.")

        msg = f"PyARPES failed to find a plugin acceptable for {file}."
        raise ValueError(msg)

    def load(
        self,
        scan_desc: ScanDesc | None = None,
        file: str | Path = "",
        **kwargs: Incomplete,
    ) -> xr.Dataset:
        """Delegates to a dynamically chosen plugin for loading."""
        if scan_desc is None:
            scan_desc = {}
        if not file:
            assert scan_desc is not None
            assert "file" in scan_desc
            file = scan_desc["file"]
        assert isinstance(file, str)
        associated_loader = FallbackEndstation.determine_associated_loader(file)
        try:
            file_number = int(file)
            file = associated_loader.find_first_file(file_number)
            scan_desc["file"] = file
        except ValueError:
            pass

        warnings.warn(AUTOLOAD_WARNING.format(associated_loader), stacklevel=2)
        return associated_loader().load(scan_desc, **kwargs)

    @classmethod
    def find_first_file(cls: type[FallbackEndstation], file_number: int) -> Path:
        """Finds any file associated to this scan.

        Instead actually using the superclass code here, we first try to determine
        which loading plugin should be used. Then, we delegate to that class to
        find the first associated file.
        """
        associated_loader = cls.determine_associated_loader(str(file_number))
        warnings.warn(AUTOLOAD_WARNING.format(associated_loader), stacklevel=2)
        return associated_loader.find_first_file(file_number)
