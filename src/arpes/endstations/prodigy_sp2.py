"""IO support for Prodigy-exported SP2 files.

This module provides tools to parse, load, and export ARPES spectrum data
saved in Igor Pro's ITX format (as exported by Prodigy) and SP2 format.
It enables interoperability between Prodigy exports and pyARPES by providing
data structures and functions to convert files to and from `xarray.DataArray`.

Main components:
- `ProdigySP2`: Parser and converter class for Prodigy SP2 files.
- `load_sp2`: Function to load data from older `.sp2` format files.
- Internal utilities for header parsing, unit correction, and metadata integration.

The output format is compatible with pyARPES, using physical units (e.g. radians),
and attaches metadata as `attrs` in `xarray` structures.

Typical usage:
    arr = load_sp2("example.sp2")
"""

from __future__ import annotations

import re
from logging import DEBUG, INFO
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from arpes.debug import setup_logger
from arpes.endstations._helper.prodigy import angle_unit_to_rad, correct_angle_region

if TYPE_CHECKING:
    from numpy.typing import NDArray


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


def load_sp2(
    path_to_file: Path | str,
    *,
    keep_degree: bool = False,
    **kwargs: str | float,
) -> xr.DataArray:
    """Load and parse sp2 file.

    Args:
        path_to_file(Path | str): Path to sp2 file
        keep_degree (bool): If True, keep angle unit in degree. Default to False.
        kwargs(str | int | float): Treated as attrs

    Returns:
        xr.DataArray: pyARPES compatible
    """
    params: dict[str, str | float] = {}
    data: list[float] = []
    pixels: tuple[int, int] = (0, 0)
    coords: dict[str, NDArray[np.float64]] = {}
    with Path(path_to_file).open(encoding="Windows-1252") as sp2file:
        for line in sp2file:
            if line.startswith("#"):
                params = _parse_sp2_comment(line, params)
            elif line.startswith("P"):
                pass
            elif pixels != (0, 0):
                data.append(float(line))
            else:
                pixels = (
                    int(line.split()[1]),
                    int(line.split()[0]),
                )
    if pixels != (0, 0):
        if isinstance(params["X Range"], str):
            e_range = [float(i) for i in re.findall(r"-?[0-9]+\.?[0-9]*", params["X Range"])]
            coords["eV"] = np.linspace(e_range[0], e_range[1], pixels[1], dtype=np.float64)
        if isinstance(params["Y Range"], str):
            a_range = [float(i) for i in re.findall(r"-?[0-9]+\.?[0-9]*", params["Y Range"])]
            corrected_angles = correct_angle_region(a_range[0], a_range[1], pixels[0])
            if keep_degree:
                coords["phi"] = np.linspace(
                    corrected_angles[0],
                    corrected_angles[1],
                    pixels[0],
                )
            else:
                coords["phi"] = np.deg2rad(
                    np.linspace(corrected_angles[0], corrected_angles[1], pixels[0]),
                )
    params["spectrum_type"] = "cut"
    params = angle_unit_to_rad(params)
    data_array: xr.DataArray = xr.DataArray(
        np.array(data).reshape(pixels),
        coords=coords,
        dims=["phi", "eV"],
        attrs=params,
    )
    data_array.coords["phi"].attrs["units"] = "Radians"
    for k, v in kwargs.items():
        data_array.attrs[k] = v
    return data_array


def _parse_sp2_comment(
    line: str,
    params: dict[str, str | float],
) -> dict[str, str | float | int]:
    try:
        params[line[2:].split("=", maxsplit=1)[0].strip()] = int(
            line[2:].split("=", maxsplit=1)[1].strip(),
        )
    except ValueError:
        try:
            params[line[2:].split("=", maxsplit=1)[0].strip()] = float(
                line[2:].split("=", maxsplit=1)[1].strip(),
            )
        except ValueError:
            params[line[2:].split("=", maxsplit=1)[0].strip()] = (
                line[2:].split("=", maxsplit=1)[1].strip()
            )
    except IndexError:
        pass
    return params
