"""Plugin facility to read and normalize information from different sources to a common format."""

from __future__ import annotations

from .base import EndstationBase, HemisphericalEndstation, SynchrotronEndstation
from .fits_endstation import FITSEndstation, find_clean_coords
from .registry import (
    add_endstation,
    endstation_from_alias,
    endstation_name_from_alias,
    resolve_endstation,
)
from .ses_endstation import SESEndstation
from .single_file_endstation import SingleFileEndstation

__all__ = [
    "EndstationBase",
    "FITSEndstation",
    "HemisphericalEndstation",
    "SESEndstation",
    "SingleFileEndstation",
    "SynchrotronEndstation",
    "add_endstation",
    "endstation_from_alias",
    "endstation_name_from_alias",
    "find_clean_coords",
    "resolve_endstation",
]
