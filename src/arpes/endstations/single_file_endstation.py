"""Module providing the SingleFileEndstation and SESEndstation classes.

These classes implement ARPES endstation plugins designed for data formats
where all relevant scan data is contained within a single file or tightly
associated file sets.

- SingleFileEndstation: Abstract base for endstations loading from a single data file.
- SESEndstation: Specialized loader for Scienta SESWrapper format and associated data.

The module handles file resolution, data loading, and coordinate normalization
specific to these data formats.
"""

from __future__ import annotations

from logging import DEBUG, INFO
from pathlib import Path
from typing import TYPE_CHECKING

from arpes.configuration.interface import get_data_path
from arpes.debug import setup_logger

from .base import EndstationBase

if TYPE_CHECKING:
    from arpes._typing import ScanDesc

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


class SingleFileEndstation(EndstationBase):
    """Abstract endstation which loads data from a single file.

    This just specializes the routine used to determine the location of files on disk.

    Unlike general endstations, if your data comes in a single file you can trust that the
    file given to you in the spreadsheet or direct load calls is all there is.
    """

    def resolve_frame_locations(self, scan_desc: ScanDesc | None = None) -> list[Path]:
        """Single file endstations just use the referenced file from the scan description."""
        if scan_desc is None:
            msg = "Must pass dictionary as file scan_desc to all endstation loading code."
            raise ValueError(
                msg,
            )

        original_data_loc = scan_desc.get("path", scan_desc.get("file"))
        assert original_data_loc
        if not Path(original_data_loc).exists():
            data_path = get_data_path()
            if data_path is not None:
                original_data_loc = Path(data_path) / original_data_loc
            else:
                msg = "File not found"
                raise RuntimeError(msg)
        return [Path(original_data_loc)]
