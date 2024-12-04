"""Implements support for the Lanzara/Kaindl HHG lab."""

from __future__ import annotations

import re
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import xarray as xr

from arpes import DATA_PATH
from arpes.constants import TWO_DIMENSION
from arpes.endstations import HemisphericalEndstation, SESEndstation

if TYPE_CHECKING:
    from arpes.endstations import ScanDesc

__all__ = ("KaindlEndstation",)

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


def find_kaindl_files_associated(reference_path: Path) -> list[Path]:
    name_match = re.match(
        r"([\w+]*_?scan_[0-9][0-9][0-9]_)[0-9][0-9][0-9]\.pxt",
        reference_path.name,
    )

    if name_match is None:
        return [reference_path]

    # otherwise need to collect all of the components
    fragment = name_match.groups()[0]
    components = list(reference_path.parent.glob(f"{fragment}*.pxt"))
    components.sort()

    return components


def read_ai_file(path: Path) -> pd.DataFrame:
    """Reads metadata from the Kaindl _AI.txt files.

    Kayla and Conrad discovered that Scienta does not record these files in a standardized format,
    but instead puts an arbitrarily long header at the top of the file and sometimes omits the
    column names.

    By manual inspection, we determined that despite this, the columns appear consistent
    across files recorded in these two formats. The columns are:

    ["Elapsed Time (s)", "Main Chamber", "Garage", "Integrated Photo AI",
     "Photo AI", "Photocurrent", "Heater Power", "Temperature A",
     "Temperature B"]

    depending on whether the header is there or not we need to skip a variable number of lines.
    The way we are detecting this is to look for the presence of the header and if it is in the file
    use it as the previous line before the start of the data. Ultimately we defer loading to pandas.

    Otherwise, if the header is absent we look for a tab as the first line of data.
    """
    with Path(path).open() as f:
        lines = f.readlines()

    first_line_no = None
    for i, line in enumerate(lines):
        if "\t" in line:
            first_line_no = i
            break

    # update with above
    column_names = [
        "Elapsed Time (s)",
        "Main Chamber",
        "Garage",
        "Integrated Photo AI",
        "Photo AI",
        "Photocurrent",
        "Heater Power",
        "Temperature A",
        "Temperature B",
    ]

    return pd.read_csv(str(path), sep="\t", skiprows=first_line_no, names=column_names)


class KaindlEndstation(HemisphericalEndstation, SESEndstation):
    """The Kaindl Tr-ARPES high harmonic generation setup."""

    PRINCIPAL_NAME = "Kaindl"
    ALIASES: ClassVar[list] = []

    _TOLERATED_EXTENSIONS: ClassVar[set[str]] = {
        ".pxt",
    }
    _SEARCH_PATTERNS = (
        r"[\-a-zA-Z0-9_\w+]+scan_[0]*{}_[0-9][0-9][0-9]",
        r"[\-a-zA-Z0-9_\w+]+scan_[0]*{}",
    )

    RENAME_KEYS: ClassVar[dict[str, str]] = {
        "Delay Stage": "delay",
    }

    def resolve_frame_locations(self, scan_desc: ScanDesc | None = None) -> list[Path]:
        """Fines .pxt files associated to a potentially multi-cut scan.

        This is very similar to what happens on BL4 at the ALS. You can look
        at the code for MERLIN to see more about how this works, or in
        `find_kaindl_files_associated`.
        """
        if scan_desc is None:
            msg = "Must pass dictionary as file scan_desc to all endstation loading code."
            raise ValueError(
                msg,
            )

        original_data_loc = scan_desc.get("path", scan_desc.get("file"))
        assert original_data_loc is not None
        assert original_data_loc != ""
        p = Path(original_data_loc)
        if not p.exists():
            if DATA_PATH is not None:
                original_data_loc = Path(DATA_PATH) / original_data_loc
            else:
                msg = "File not found"
                raise RuntimeError(msg)

        return find_kaindl_files_associated(Path(original_data_loc))

    def concatenate_frames(
        self,
        frames: list[xr.Dataset],
        scan_desc: ScanDesc | None = None,
    ) -> xr.Dataset | None:
        """Concenates frames from individual .pxt files on the Kaindl setup.

        The unique challenge here is to look for and parse the motor positions file (if
        they exist) and add this as a coordinate. As in Beamline 4 at the ALS, these Motor_Pos
        file gives the scan coordinate which we need to concatenate along.
        """
        if len(frames) < TWO_DIMENSION:
            return super().concatenate_frames(frames)

        # determine which axis to stitch them together along, and then do this
        assert scan_desc
        original_filename = scan_desc.get("path", scan_desc.get("file"))
        assert original_filename is not None

        internal_match = re.match(
            r"([a-zA-Z0-9\w+_]+)_[0-9][0-9][0-9]\.pxt",
            Path(original_filename).name,
        )
        assert internal_match is not None
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

                for v, f in zip(values, frames, strict=True):
                    f.coords[axis_name] = v

                frames.sort(key=lambda x: x.coords[axis_name])
                return xr.concat(frames, axis_name)
            except Exception:
                logger.exception("Exception occurs.")
        return None

    def postprocess_final(
        self,
        data: xr.Dataset,
        scan_desc: ScanDesc | None = None,
    ) -> xr.Dataset:
        """Performs final data preprocessing for the Kaindl lab Tr-ARPES setup.

        This is very similar to what happens at BL4/MERLIN because the code was adopted
        from an old version of the DAQ on that beamline.

        Args:
            data (xr.DataSet): [TODO:description]
            scan_desc (ScanDesc): [TODO:description]
        """
        assert scan_desc
        original_filename = scan_desc.get("path", scan_desc.get("file"))
        assert original_filename
        internal_match = re.match(
            r"([a-zA-Z0-9\w+_]+_[0-9][0-9][0-9])\.pxt",
            Path(original_filename).name,
        )
        assert internal_match is not None
        all_filenames: list[Path] = find_kaindl_files_associated(Path(original_filename))
        all_filenames = [f.parent / f"{f.stem}_AI.txt" for f in all_filenames]

        def load_attr_for_frame(filename: Path, attr_name: str):
            # this is rereading which is not ideal but can be adjusted later
            """[TODO:summary].

            Args:
                filename (str): [TODO:description]
                attr_name (str): [TODO:description]
            """
            df = read_ai_file(filename)
            return np.mean(df[attr_name])

        def attach_attr(data: xr.Dataset, attr_name: str, as_name: str) -> xr.Dataset:
            """[TODO:summary].

            Args:
                data (xr.Dataset): [TODO:description]
                attr_name (str): [TODO:description]
                as_name (str): [TODO:description]
            """
            attributes = np.array([load_attr_for_frame(f, attr_name) for f in all_filenames])

            if len(attributes) == 1:
                data[as_name] = attributes[0]
            else:
                non_spectrometer_dims = [d for d in data.spectrum.dims if d not in {"eV", "phi"}]
                non_spectrometer_coords = {
                    c: v for c, v in data.spectrum.coords.items() if c in non_spectrometer_dims
                }

                new_shape = [len(data.coords[d]) for d in non_spectrometer_dims]
                attributes_arr = xr.DataArray(
                    attributes.reshape(new_shape),
                    coords=non_spectrometer_coords,
                    dims=non_spectrometer_dims,
                )

                data = xr.merge([data, xr.Dataset({as_name: attributes_arr})])

            return data

        try:
            data = attach_attr(data, "Photocurrent", "photocurrent")
            data = attach_attr(data, "Temperature B", "temp")
            data = attach_attr(data, "Temperature A", "cryotip_temp")
        except FileNotFoundError as err:
            logger.info(f"Exception occurs: {err}")

        if internal_match.groups():
            attrs_path = str(
                Path(original_filename).parent / f"{internal_match.groups()[0]}_AI.txt",
            )

            try:
                extra = pd.read_csv(attrs_path, sep="\t", skiprows=6)
                data = data.assign_attrs(extra=extra.to_json())
            except Exception:
                logger.exception("Exception occurs")

        deg_to_rad_coords = {"theta", "beta", "phi"}

        for c in deg_to_rad_coords:
            if c in data.dims:
                data.coords[c] = np.deg2rad(data.coords[c])

        deg_to_rad_attrs = {"theta", "beta", "alpha", "chi"}
        for angle_attr in deg_to_rad_attrs:
            if angle_attr in data.attrs:
                data.attrs[angle_attr] = np.deg2rad(float(data.attrs[angle_attr]))

        ls = [data, *[dv for dv in data.data_vars.values() if "eV" in dv.dims]]
        for _ in ls:
            _.coords["x"] = np.nan
            _.coords["y"] = np.nan
            _.coords["z"] = np.nan

        return super().postprocess_final(data, scan_desc)
