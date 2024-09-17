"""Implements data loading for the URANOS beamline @ Solaris.

Plugin based on the SSRF_NSRL.
"""

from __future__ import annotations

import io
import warnings
from configparser import ConfigParser
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar
from zipfile import ZipFile

import numpy as np
import xarray as xr

from arpes.endstations import (
    HemisphericalEndstation,
    SingleFileEndstation,
    SynchrotronEndstation,
    add_endstation,
)
from arpes.load_pxt import read_single_pxt
from arpes.utilities import clean_keys

if TYPE_CHECKING:
    from numpy._typing import NDArray

    from arpes._typing import Spectrometer
    from arpes.endstations import ScanDesc

__all__ = ["Uranos"]


class Uranos(HemisphericalEndstation, SingleFileEndstation, SynchrotronEndstation):
    """Class for Uranos beamline at Solaris Krakow, PL."""
    PRINCIPAL_NAME = "Uranos"
    ALIASES: ClassVar[list[str]] = ["Uranos", "Uranos_JU", "Uranos_Solaris"]

    _SEARCH_DIRECTORIES = ("zip", "pxt")
    _TOLERATED_EXTENSIONS: ClassVar[set[str]] = {".zip", ".pxt"}

    RENAME_KEYS: ClassVar[dict[str, str]] = {
        "sample": "sample_name",
        "spectrum_name": "spectrum_type",
        "low_energy": "sweep_low_energy",
        "center_energy": "sweep_center_energy",
        "high_energy": "sweep_high_energy",
        "step_time": "n_sweeps",
        "energy_step": "sweep_step",
        "instrument": "analyzer",
        "region_name": "id",
        "excitation_energy": "hv",
        "X": "x",
        "Y": "y",
        "z": "z",
        "r1": "theta",
        "r3": "beta",
    }

    NORMAL_EMISSION: ClassVar[dict[str, float]] = {
        "theta": 178.0,
        "beta": -88.0,
    }

    ANALYZER_WORK_FUNCTION = 4.38

    MERGE_ATTRS: ClassVar[Spectrometer] = {
        "analyzer_name": "DA30L",
        "analyzer_type": "hemispherical",
        "perpendicular_deflectors": True,
        "parallel_deflectors": True,
        "work_function": ANALYZER_WORK_FUNCTION,
        "alpha": np.deg2rad(90),
    }

    def load_single_frame(
            self,
            frame_path: str | Path = "",
            scan_desc: ScanDesc | None = None,
            **kwargs: str | float,
    ) -> xr.Dataset:
        """Load arpes data: cut / cuts (pxt file) or map (zip file)."""
        if kwargs:
            warnings.warn("Any kwargs is not supported in this function.", stacklevel=2)
        if scan_desc is None:
            scan_desc = {}

        file = Path(frame_path)

        if file.suffix == ".pxt":
            datas = read_single_pxt(frame_path,
                                    byte_order="<",
                                    allow_multiple=True,
                                    ).rename(W="eV", X="phi")
            data_var_name = next(iter(datas.data_vars.keys()))
            data = datas[data_var_name]
            return xr.Dataset({"spectrum": data}, attrs=data.attrs)

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
                v: _formatted_value(k)
                for section in attr_conf.sections()
                for v, k in attr_conf.items(section)
            }
            attrs = clean_keys(attrs)

            data = xr.DataArray(
                loaded_data,
                dims=["psi", "phi", "eV"],
                coords=built_coords,
                attrs=attrs,
            )
            return xr.Dataset(
                {"spectrum": data},
                attrs=data.attrs,
            )
        msg = "Not supported file extension"
        raise RuntimeError(msg)

    def postprocess_final(
            self,
            data: xr.Dataset,
            scan_desc: ScanDesc | None = None,
    ) -> xr.Dataset:
        """Perform final processing on the ARPES data.

        - Add missing parameters.

        Args:
            data(xr.Dataset): ARPES data
            scan_desc(SCANDESC | None): scan_description. Not used currently

        Returns:
            xr.Dataset: pyARPES compatible.
        """
        """Add missing parameters."""
        if scan_desc is None:
            scan_desc = {}
        defaults = {
            "beta": Uranos.NORMAL_EMISSION["beta"],
            "chi": 0.0,
            "psi": 0.0,
            "theta": Uranos.NORMAL_EMISSION["theta"],
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "alpha": np.deg2rad(90),
            "energy_notation": "Binding",
        }
        for k, v in defaults.items():
            data.attrs[k] = data.attrs.get(k, v)
            for s in [dv for dv in data.data_vars.values() if "eV" in dv.dims]:
                s.attrs[k] = s.attrs.get(k, v)

        binding_energies = (data.coords["eV"].values
                            - data.attrs["hv"]
                            + Uranos.ANALYZER_WORK_FUNCTION)
        data = data.assign_coords({"eV": binding_energies})

        data = data.rename({k: v for k, v in self.RENAME_KEYS.items() if k in data.coords})

        for coord in ["psi", "phi"]:
            if coord in data.coords:
                data = data.assign_coords({coord: np.deg2rad(data[coord])})

        for coord in ["theta", "beta"]:
            if coord in data.attrs:
                data.attrs[coord] = np.deg2rad(data.attrs[coord] - Uranos.NORMAL_EMISSION[coord])

        return super().postprocess_final(data, scan_desc)


def determine_dim(viewer_ini: ConfigParser, dim_name: str) -> tuple[int, NDArray[np.float64], str]:
    """Determine dimension values from the ini file.

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


def _formatted_value(value: str) -> float | str:
    """Convert string value to float if possible."""
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        return value


add_endstation(Uranos)
