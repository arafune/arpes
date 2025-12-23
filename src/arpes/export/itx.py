"""Module for export pyarpes xarray data to itx file format."""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from logging import DEBUG, INFO
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

from arpes.debug import setup_logger

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)

HEADER_TEMPLATE = """IGOR
X //Created Date (UTC): {}
X //Created by: R. Arafune
X //Acquisition Parameters:
X //Scan Mode         = {}
X //User Comment      = {}
X //Analysis Mode     = UPS
X //Lens Mode         = {}
X //Lens Voltage      = {}
X //Spectrum ID       = {}
X //Analyzer Slits    = {}
X //Number of Scans   = {}
X //Number of Samples = {}
X //Scan Step         = {}
X //DwellTime         = {}
X //Excitation Energy = {}
X //Kinetic Energy    = {}
X //Pass Energy       = {}
X //Bias Voltage      = {}
X //Detector Voltage  = {}
X //WorkFunction      = {}
"""

DIGIT_ID = 3

Measure_type = Literal["FAT", "SFAT"]


def export_itx(
    file_name: str | Path,
    arr: xr.DataArray,
    *,
    add_notes: bool = False,
) -> None:
    """Export pyarpes spectrum data to itx file.

    Args:
        file_name(str | Path): file name for export
        arr(xr.DataArray): pyarpes DataArray
        add_notes(bool): if True, add some info to notes in wave (default: False)

    Warnings:
        This function will be deprecated in future, because xarray can be exported to HDF format.
    """
    warnings.warn(
        "This method will be deprecated.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    with Path(file_name).open(mode="w", encoding="UTF-8") as itx_file:
        itx_file.write(convert_itx_format(arr, add_notes=add_notes))


def convert_itx_format(
    arr: xr.DataArray,
    *,
    add_notes: bool = False,
) -> str:
    """Export pyarpes spectrum data to itx file.

    Note: that the wave name is changed based on the ID number.

    Args:
        arr(xr.DataArray):  DataArray to export
        keep_degree(bool): if True, keep the unit of angle axis(degree).
        add_notes(bool): if True, add some info to notes in wave.(default: False)

    Returns:
        str: itx formatted ARPES data

    Warnings:
        This function will be deprecated in future, because xarray can be exported to HDF format,
        which is loaded by igor directly.
    """
    warnings.warn(
        "This method will be deprecated.",
        category=DeprecationWarning,
        stacklevel=2,
    )

    assert isinstance(arr, xr.DataArray)
    if "User Comment" in arr.attrs:
        arr.attrs["User Comment"] += ";" + _user_comment_from_attrs(arr)
    else:
        arr.attrs["User Comment"] = _user_comment_from_attrs(arr)
    start_energy: float = arr.indexes["eV"][0]
    step_energy: float = arr.indexes["eV"][1] - arr.indexes["eV"][0]
    end_energy: float = arr.indexes["eV"][-1]
    parameters = arr.attrs
    parameters["StartEnergy"] = start_energy
    parameters["StepWidth"] = step_energy
    itx_str: str = _build_itx_header(
        arr.attrs,
        comment=arr.attrs.get("User Comment", ""),
        measure_mode=arr.attrs.get("scan_mode", "Fixed Analyzer Transmission"),
    )
    phi_pixel = len(arr.coords["phi"])
    energy_pixel = len(arr.coords["eV"])
    id_number = parameters.get("id", parameters.get("Spectrum ID"))
    wavename = "ID_" + str(id_number).zfill(DIGIT_ID)
    itx_str += f"WAVES/S/N=({phi_pixel},{energy_pixel}) '{wavename}'\nBEGIN\n"
    try:
        intensities_list = arr.to_dict()["data_vars"]["spectrum"]["data"]
    except KeyError:
        intensities_list = arr.to_dict()["data"]
    for a_intensities in intensities_list:
        itx_str += " ".join(map(str, a_intensities)) + "\n"
    itx_str += "END\n"
    start_phi_deg: float = arr.indexes["phi"][0]
    end_phi_deg: float = arr.indexes["phi"][-1]
    itx_str += (
        f"""X SetScale/I x, {start_phi_deg}, {end_phi_deg}, "deg (theta_y)", '{wavename}'\n"""
    )
    itx_str += f"""X SetScale/I y, {start_energy}, {end_energy}, "eV", '{wavename}'\n"""

    itx_str += """X SetScale/I d, 0, 0, "{}", '{}'\n""".format(
        arr.attrs.get("count_unit", "cps"),
        wavename,
    )
    if add_notes:
        itx_str += """X Note /NOCR '{}' "{}"\r\n""".format(
            wavename,
            arr.attrs["User Comment"],
        )
        excitation_energy = arr.attrs.get("hv", parameters.get("Excitation Energy"))
        itx_str += f"""X Note /NOCR '{wavename}', "Excitation Energy:{excitation_energy}"\r\n"""
        # parameter should be recorded.
        # x, y, z (if defined)
        #
    return itx_str


def _user_comment_from_attrs(
    dataarray: xr.DataArray,
) -> str:
    key_pos: set[str] = {"x", "y", "z"}
    key_angle: set[str] = {"beta", "chi", "psi"}
    user_comment = ""
    for key, value in dataarray.attrs.items():
        if key in key_pos and not np.isnan(value):
            logger.debug(f"key: {key}, value: {type(value)} ")
            user_comment += str(key) + ":" + f"{value}" + ";"
        if key in key_angle:
            user_comment += str(key) + ":" + f"{value}"
    return user_comment


def _build_itx_header(
    param: dict[str, str | float],
    comment: str = "",
    measure_mode: Measure_type = "FAT",
) -> str:
    """Make itx file header.

    Parameters
    ----------
    param: dict[str, str | float]
        Spectrum parameter
    spectrum_id: int
        Unique id for spectrum
    num_scan: int
        Number of scan.
    comment: str
        Comment string.  Used in "//User Comment"
    measure_mode : Measure_type
        Measurement mode (FAT/SFAT)

    Returns:
    -------
    str
        Header part of itx
    """
    mode = "Fixed Analyzer Transmission" if measure_mode == "FAT" else "Snapshot"
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")
    if param["User Comment"]:
        comment += ";" + str(param["User Comment"])
    return HEADER_TEMPLATE.format(
        now,
        mode,
        param["User Comment"],
        param.get("lens_mode", param.get("Lens Mode")),
        param.get("Lens Voltage", param.get("lens voltage")),
        param.get("id", param.get("Spectrum ID")),
        param.get("Analyzer Slits", param.get("analyzer_slits")),
        param.get("Number of Scans", param.get("number_of_scans")),
        param.get("Number of Samples", param.get("number_of_samples")),
        param["StepWidth"],
        param.get("DwellTime", param.get("dwell_time")),
        param.get("hv", param.get("Excitation Energy")),
        param["StartEnergy"],
        param.get("pass_energy", param.get("Pass Energy", 5)),
        param.get("Bias Voltage", param.get("bias_voltage")),
        param.get("mcp_voltage", param.get("Detector Voltage")),
        param.get("workfunction", param.get("WorkFunction", 4.401)),
    )
