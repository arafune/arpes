"""Implements data loading for the URANOS beamline @ Solaris."""

from __future__ import annotations

import io

import warnings
from configparser import ConfigParser
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar
from zipfile import ZipFile

import numpy as np
import xarray as xr

from arpes.load_pxt import read_single_pxt
from arpes.endstations import (
    HemisphericalEndstation,
    SingleFileEndstation,
    SynchrotronEndstation,
    add_endstation,
)

if TYPE_CHECKING:
    from arpes._typing import Spectrometer
    from arpes.endstations import ScanDesc

__all__ = ["Uranos"]


class Uranos(HemisphericalEndstation, SingleFileEndstation, SynchrotronEndstation):

    PRINCIPAL_NAME = "Uranos"
    ALIASES: ClassVar[list[str]] = ["Uranos", "Uranos_JU", "Uranos_Solaris"]

    _SEARCH_DIRECTORIES = ("zip", "pxt")
    _TOLERATED_EXTENSIONS: ClassVar[set[str]] = {".zip", ".pxt", ".txt"}

    RENAME_KEYS: ClassVar[dict[str, str]] = {
        "sample": "sample_name",
        "spectrum_name": "spectrum_type",
        "low_energy": "sweep_low_energy",
        "center_energy": "sweep_center_energy",
        "high_energy": "sweep_high_energy",
        "step_time": "n_sweeps",
        "energy_step": "sweep_step",
        "instrument": "analyzer",
        "region_name": "spectrum_type",
    }

    MERGE_ATTRS: ClassVar[Spectrometer] = {
        "analyzer_name": "DA30-L",
        "analyzer_type": "hemispherical",
        "perpendicular_deflectors": True,
        "parallel_deflectors": True,
        "alpha": np.deg2rad(90),
    }

    def load_single_frame(
            self,
            frame_path: str | Path = "",
            scan_desc: ScanDesc | None = None,
            **kwargs: Incomplete,
    ) -> xr.Dataset:
        if kwargs:
            warnings.warn("Any kwargs is not supported in this function.", stacklevel=2)
        if scan_desc is None:
            scan_desc = {}
        file = Path(frame_path)

        if file.suffix == ".pxt":
            frame = read_single_pxt(frame_path).rename(W="eV", X="phi")
            frame = frame.assign_coords(phi=np.deg2rad(frame.phi))

            return xr.Dataset(
                {"spectrum": frame},
                attrs=scan_desc,
            )

        if file.suffix == ".zip":
            zf = ZipFile(frame_path)
            viewer_ini_ziped = zf.open("viewer.ini", "r")
            viewer_ini_io = io.TextIOWrapper(viewer_ini_ziped)
            viewer_ini: ConfigParser = ConfigParser(strict=False)
            viewer_ini.read_file(viewer_ini_io)

            # Usually, ['width', 'height', 'depth'] -> ['eV', 'phi', 'psi']
            # For safety, get label name and sort them
            raw_coords = {}
            for label in ["width", "height", "depth"]:
                num, coord, name = determine_dim(viewer_ini, label)
                raw_coords[name] = [num, coord]
            raw_coords_name = list(raw_coords.keys())
            raw_coords_name.sort()

            # After sorting, labels must be ['Energy [eV]', 'Thetax [deg]',
            # 'Thetay [deg]'], which means ['eV', 'phi', 'psi'].
            built_coords = {
                "psi": raw_coords[raw_coords_name[2]][1],
                "phi": raw_coords[raw_coords_name[1]][1],
                "eV": raw_coords[raw_coords_name[0]][1],
            }
            (psi_num, phi_num, eV_num) = (
                raw_coords[raw_coords_name[2]][0],
                raw_coords[raw_coords_name[1]][0],
                raw_coords[raw_coords_name[0]][0],
            )

            data_path = viewer_ini.get(viewer_ini.sections()[-1], "path")
            raw_data = zf.read(data_path)
            loaded_data = np.frombuffer(raw_data, dtype="float32")
            loaded_data.shape = (psi_num, phi_num, eV_num)

            attr_path = viewer_ini.get(viewer_ini.sections()[0], "ini_path")

            attr_ziped = zf.open(attr_path, "r")
            attr_io = io.TextIOWrapper(attr_ziped)
            attr_conf = ConfigParser(strict=False)
            attr_conf.read_file(attr_io)

            attrs = {
                v.replace(" ", "_"): k
                for section in attr_conf.sections()
                for v, k in attr_conf.items(section)
            }
            data = xr.DataArray(
                loaded_data,
                dims=["psi", "phi", "eV"],
                coords=built_coords,
                attrs=attrs,
            )
            data = data.assign_coords(
                phi=np.deg2rad(data.phi),
                psi=np.deg2rad(data.psi),
            )

            return xr.Dataset(
                {"spectrum": data},
                attrs={**scan_desc},
            )
        err_msg = "Check your file is it really suitable?"
        raise RuntimeError(err_msg)


def determine_dim(viewer_ini: ConfigParser, dim_name: str) -> tuple[int, NDArray[np.float64], str]:
    """Determine dimension values from from the ini file.

    Args:
        viewer_ini (ConfigParser): Parser of "viewer.ini"
        dim_name (str): dimension name

    Returns:
        dimension info (num of dim, coord, dim name)
    """
    spectrum_info = viewer_ini.sections()[-1]

    num = viewer_ini.getint(spectrum_info, dim_name)

    offset = viewer_ini.getfloat(spectrum_info, dim_name + "_offset")
    delta = viewer_ini.getfloat(spectrum_info, dim_name + "_delta")
    end = offset + num * delta
    coord = np.linspace(offset, end, num=num, endpoint=False)

    name = viewer_ini.get(spectrum_info, dim_name + "_label")

    return num, coord, name

add_endstation(Uranos)
