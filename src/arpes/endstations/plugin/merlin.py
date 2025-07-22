"""The MERLIN ARPES Endstation at the Advanced Light Source."""

from __future__ import annotations

import re
from logging import DEBUG, INFO
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import xarray as xr

from arpes.debug import setup_logger
from arpes.endstations import (
    HemisphericalEndstation,
    SESEndstation,
    SynchrotronEndstation,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from _typeshed import Incomplete

    from arpes._typing.attrs_property import ScanDesc, Spectrometer

__all__ = ("BL403ARPESEndstation",)

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


class BL403ARPESEndstation(SynchrotronEndstation, HemisphericalEndstation, SESEndstation):
    """The MERLIN ARPES Endstation at the Advanced Light Source."""

    PRINCIPAL_NAME = "ALS-BL403"
    ALIASES: ClassVar[list[str]] = [
        "BL403",
        "BL4",
        "BL4.0.3",
        "ALS-BL403",
        "ALS-BL4",
    ]

    _TOLERATED_EXTENSIONS: ClassVar[set[str]] = {
        ".pxt",
    }
    _SEARCH_PATTERNS = (
        r"[\-a-zA-Z0-9_\w+]+_{}_S[0-9][0-9][0-9]$",
        r"[\-a-zA-Z0-9_\w+]+_{}_R[0-9][0-9][0-9]$",
        r"[\-a-zA-Z0-9_\w+]+_[0]+{}_S[0-9][0-9][0-9]$",
        r"[\-a-zA-Z0-9_\w+]+_[0]+{}_R[0-9][0-9][0-9]$",
        # more generic
        r"[\-a-zA-Z0-9_\w]+_[0]+{}$",
        r"[\-a-zA-Z0-9_\w]+_{}$",
        r"[\-a-zA-Z0-9_\w]+{}$",
        r"[\-a-zA-Z0-9_\w]+[0]{}$",
    )

    RENAME_KEYS: ClassVar[dict[str, str]] = {
        "Polar": "theta",
        "Polar Compens": "theta",  # these are caps-ed because they are dimensions in some cases!
        "BL Energy": "hv",
        "tilt": "beta",
        "polar": "theta",
        "azimuth": "chi",
        "temperature_sensor_a": "temperature_cryotip",
        "temperature_sensor_b": "temperature",
        "cryostat_temp_a": "temp_cryotip",
        "cryostat_temp_b": "temp",
        "bl_energy": "hv",
        "polar_compens": "theta",
        "K2200 V": "volts",
        "pwr_supply_v": "volts",
        "mcp": "mcp_voltage",
        "slit_plate": "slit_number",
        "user": "experimenter",
        "sample": "sample_name",
        "mesh_current": "photon_flux",
        "ring_energy": "beam_energy",
        "epu_pol": "undulator_polarization",
        "epu_gap": "undulator_gap",
        "epu_z": "undulator_z",
        "center_energy": "daq_center_energy",
        "low_energy": "sweep_low_energy",
        "high_energy": "sweep_high_energy",
        "energy_step": "sweep_step",
        "number_of_sweeps": "n_sweeps",
    }

    MERGE_ATTRS: ClassVar[Spectrometer] = {
        "analyzer": "R8000",
        "analyzer_name": "Scienta R8000",
        "parallel_deflectors": False,
        "perpendicular_deflectors": False,
        "analyzer_radius": np.nan,
        "analyzer_type": "hemispherical",
        "repetition_rate": 5e8,
        "undulator_harmonic": 2,  # TODO:
        "undulator_type": "elliptically_polarized_undulator",
    }

    ATTR_TRANSFORMS: ClassVar[dict[str, Callable[..., dict[str, float | list[str] | str]]]] = {
        "acquisition_mode": lambda _: _.lower(),
        "lens_mode": lambda _: {
            "lens_mode": None,
            "lens_mode_name": _,
        },
        "undulator_polarization": int,
        "region_name": lambda _: {
            "daq_region_name": _,
            "daq_region": _,
        },
    }

    def concatenate_frames(
        self,
        frames: list[xr.Dataset],
        scan_desc: ScanDesc | None = None,
    ) -> xr.Dataset:
        """Concatenates frames from different files into a single scan.

        Above standard process here, we need to look for a Motor_Pos.txt
        file which contains the coordinates of the scanned axis so that we can
        stitch the different elements together.
        """
        if len(frames) < 2:  # noqa: PLR2004
            return super().concatenate_frames(frames)
        if scan_desc is None:
            scan_desc = {}
        # determine which axis to stitch them together along, and then do this
        original_filename = scan_desc.get("file", scan_desc.get("path"))
        assert original_filename is not None

        internal_match = re.match(
            r"([a-zA-Z0-9\w+_]+)_[S][0-9][0-9][0-9]\.pxt",
            Path(original_filename).name,
        )
        if internal_match is not None:
            if internal_match.groups():
                motors_path = str(
                    Path(original_filename).parent / f"{internal_match.groups()[0]}_Motor_Pos.txt",
                )
                try:
                    with Path(motors_path).open() as f:
                        lines = f.readlines()

                    axis_name = lines[0].strip()
                    axis_name = self.RENAME_KEYS.get(axis_name, axis_name)
                    values = [float(_.strip()) for _ in lines[1 : len(frames) + 1]]

                    for v, frame in zip(values, frames, strict=True):
                        frame.coords[axis_name] = v

                    frames.sort(key=lambda x: x.coords[axis_name])

                    for frame in frames:
                        # promote x, y, z to coords so they get concatted
                        for _ in [
                            frame,
                            *[dv for dv in frame.data_vars.values() if "eV" in dv.dims],
                        ]:
                            for c in ["x", "y", "z"]:
                                if c not in _.coords:
                                    _.coords[c] = _.attrs[c]

                    return xr.concat(frames, axis_name, coords="different")
                except Exception:
                    logger.exception("Exception occurs.")

        else:
            internal_match = re.match(
                r"([a-zA-Z0-9\w+_]+)_[R][0-9][0-9][0-9]\.pxt",
                Path(original_filename).name,
            )
            if internal_match is not None and internal_match.groups():
                return xr.merge(frames)

        return super().concatenate_frames(frames)

    def load_single_frame(
        self,
        frame_path: str | Path = "",
        scan_desc: ScanDesc | None = None,
        **kwargs: Incomplete,
    ) -> xr.Dataset:
        """Loads all regions for a single .pxt frame, and perform per-frame normalization."""
        from arpes.load_pxt import find_ses_files_associated, read_single_pxt

        if scan_desc is None:
            scan_desc = {}
        ext = Path(frame_path).suffix
        if "nc" in ext:
            # was converted to hdf5/NetCDF format with Conrad's Igor scripts
            scan_desc["path"] = frame_path
            return self.load_SES_nc(scan_desc=scan_desc, **kwargs)

        p = Path(scan_desc.get("path", scan_desc.get("file", "")))

        # find files with same name stem, indexed in format R###
        regions = find_ses_files_associated(p, separator="R")

        if len(regions) == 1:
            pxt_data = read_single_pxt(frame_path).assign_coords(
                {"eV": -read_single_pxt(frame_path).eV.values},
            )  # negate energy
            return xr.Dataset({"spectrum": pxt_data}, attrs=pxt_data.attrs)
        # need to merge several different detector 'regions' in the same scan
        region_files = [self.load_single_region(region_path) for region_path in regions]

        # can they share their energy axes?
        all_same_energy = True
        for reg in region_files[1:]:
            dim = "eV" + reg.attrs["Rnum"]
            all_same_energy = all_same_energy and np.array_equal(
                region_files[0].coords["eV000"],
                reg.coords[dim],
            )

        if all_same_energy:
            for i, reg in enumerate(region_files):
                dim = "eV" + reg.attrs["Rnum"]
                region_files[i] = reg.rename({dim: "eV"})
        else:
            pass

        return self.concatenate_frames(region_files, scan_desc=scan_desc)

    def load_single_region(
        self,
        region_path: str | Path = "",
        scan_desc: ScanDesc | None = None,
        **kwargs: Incomplete,
    ) -> xr.Dataset:
        """Loads a single region for multi-region scans."""
        if scan_desc:
            logger.debug("BL403ARPESEndstation: scan_desc is not used in this class.")
        if kwargs:
            for k, v in kwargs.items():
                logger.debug(f"BL403ARPESEndstation: key {k}: value{v} is not used in this class.")
        from arpes.load_pxt import read_single_pxt

        name = Path(region_path).stem
        num = name[-3:]

        pxt_data = read_single_pxt(region_path).assign_coords(
            {"eV": -read_single_pxt(region_path).eV.values},
        )  # negate energy
        pxt_data = pxt_data.rename({"eV": "eV" + num})
        pxt_data.attrs["Rnum"] = num
        pxt_data.attrs["alpha"] = np.pi / 2
        return xr.Dataset(
            {"spectrum" + num: pxt_data},
            attrs=pxt_data.attrs,
        )  # separate spectra for possibly unrelated data

    def postprocess_final(
        self,
        data: xr.Dataset,
        scan_desc: ScanDesc | None = None,
    ) -> xr.Dataset:
        """Performs final data normalization for MERLIN data.

        Additional steps we perform here are:

        1. We attach the slit information for the R8000 used on MERLIN.
        2. We normalize the undulator polarization from the sentinel values
          recorded by the beamline.
        3. We convert angle units to radians.

        Args:
            data: The input data
            scan_desc: Originating load parameters

        Returns:
            Processed copy of the data
        """
        ls = [data, *[dv for dv in data.data_vars.values() if "eV" in dv.dims]]

        for dat in ls:
            if "slit_number" in dat.attrs:
                slit_lookup = {
                    1: ("straight", 0.1),
                    7: ("curved", 0.5),
                }
                shape, width = slit_lookup.get(dat.attrs["slit_number"], (None, None))
                dat.attrs["slit_shape"] = shape
                dat.attrs["slit_width"] = width
            if "undulator_polarization" in dat.attrs:
                phase_angle_lookup = {0: (0, 0), 2: (np.pi / 2, 0)}  # LH  # LV
                polarization_theta, polarization_alpha = phase_angle_lookup[
                    int(dat.attrs["undulator_polarization"])
                ]
                dat.attrs["probe_polarization_theta"] = polarization_theta
                dat.attrs["probe_polarization_alpha"] = polarization_alpha
            for angle_attr in ("alpha", "beta", "chi", "psi", "theta"):
                if angle_attr in dat.attrs:
                    dat.attrs[angle_attr] = np.deg2rad(float(dat.attrs[angle_attr]))
            for cname in ("theta", "beta", "chi", "phi"):
                if cname not in dat.attrs and cname not in dat.coords and cname in dat.attrs:
                    dat.attrs[cname] = data.attrs[cname]
            dat.attrs["grating"] = "HEG"
            dat.attrs["alpha"] = np.pi / 2
            dat.attrs["psi"] = 0

        for c in ("beta", "chi", "psi", "phi", "theta"):
            if c in data.dims:
                data.coords[c] = np.deg2rad(data.coords[c])

        return super().postprocess_final(data, scan_desc)
