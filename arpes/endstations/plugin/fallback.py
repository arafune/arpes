"""Implements dynamic plugin selection when users do not specify the location for their data."""
from __future__ import annotations

import warnings
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, ClassVar

from arpes.endstations import EndstationBase, resolve_endstation
from arpes.trace import Trace, traceable

if TYPE_CHECKING:
    from pathlib import Path

    from _typeshed import Incomplete

    from arpes.endstations import SCANDESC
__all__ = ("FallbackEndstation",)

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


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
    ]

    @classmethod
    @traceable
    def determine_associated_loader(
        cls: type[FallbackEndstation],
        file: str | Path,
        *,
        trace: Trace | None = None,
    ) -> type[EndstationBase]:
        """Determines which loading plugin to use for a given piece of data.

        This is done by looping through loaders in a predetermined priority order,
        and asking each whether it is capable of loading the file.
        """
        import arpes.config  # pylint: disable=redefined-outer-name

        arpes.config.load_plugins()

        for location in cls.ATTEMPT_ORDER:
            trace(f"{cls.__name__} is trying {location}")

            try:
                endstation_cls = resolve_endstation(retry=False, location=location)
                if endstation_cls.is_file_accepted(file):
                    return endstation_cls
            except Exception as err:
                logger.info(f"Exception occurs. {err=}, {type(err)=}")

        msg = f"PyARPES failed to find a plugin acceptable for {file}."
        raise ValueError(msg)

    def load(
        self,
        scan_desc: SCANDESC | None = None,
        file: str | Path = "",
        **kwargs: Incomplete,
    ):
        """Delegates to a dynamically chosen plugin for loading."""
        if scan_desc is None:
            scan_desc = {}
        if not file:
            assert scan_desc is not None
            file = scan_desc["file"]
        assert isinstance(file, str)
        associated_loader = FallbackEndstation.determine_associated_loader(
            file,
            scan_desc,
            trace=self.trace,
        )
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
        which loading pluging should be used. Then, we delegate to that class to
        find the first associated file.
        """
        associated_loader = cls.determine_associated_loader(file_number)
        warnings.warn(AUTOLOAD_WARNING.format(associated_loader), stacklevel=2)
        return associated_loader.find_first_file(file_number)
