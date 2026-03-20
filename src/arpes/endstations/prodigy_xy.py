"""Input support for SpecsLab Prodigy exported .xy files.

Supported data dimensionality:
- 1D: energy only
- 2D: energy vs nonenergy (angle or spatial coordinate)
- 3D: energy vs nonenergy vs parameter

Data model:
- The first column is always energy (eV).
- The second column is intensity.
- Additional dimensions are reconstructed from header metadata:
    * NonEnergyOrdinate → second dimension (if present)
    * Parameter → third dimension (if present)

Notes:
- Energy axis is reconstructed using linspace for numerical stability.
- Dimension names are preserved from the file and normalized later by plugins.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from arpes.helper import clean_keys

if TYPE_CHECKING:
    from numpy.typing import NDArray

from dataclasses import dataclass


@dataclass
class Axis:
    name: str
    values: NDArray[np.float64]


SECOND_DIM_NAME = "nonenergy"
THIRD_DIM_NAME = "parameter"

# XY Prodigy file header section
HEADER_SECTION_START = "# Group"
HEADER_SECTION_END = "# Cycle"

HEADER_KV_RE = re.compile(r"# (?P<key>[^:]+):\s*(?P<value>.*\S)?")

HEADER_VALUES_TYPES = {
    "curves_scan": int,
    "values_curve": int,
    "dwell_time": float,
    "excitation_energy": float,
    "kinetic_energy": float,
    "pass_energy": float,
    "bias_voltage": float,
    "detector_voltage": float,
    "eff_workfunction": float,
}

# XY Prodigy dimension name and values patterns
NUMBER_RE = r"(?P<value>[+-]?\d+(?:\.\d+)?)"

PARAMETER_RE = re.compile(r'# Parameter: "(?P<name>[^"\[]+)(?: \[[^\]]+\])?" = ' + NUMBER_RE)
NONENERGY_RE = re.compile(r"# NonEnergyOrdinate:\s+" + NUMBER_RE)

__all__ = ["load_xy"]


def load_xy(
    path_to_file: Path | str,
    **kwargs: str | float,
) -> xr.DataArray:
    """Load and parse the Prodigy exported xy files.

    Args:
        path_to_file (Path | str): Path to xy file.
        kwargs (str | int | float): Treated as attrs and passed to DataArray

    Returns:
        xr.DataArray: pyARPES compatible
    """
    # Read the file as lines
    with Path(path_to_file).open("r") as f:
        file_as_lines = f.readlines()

    # Separate metadata
    metadata_lines = [line for line in file_as_lines if line.startswith("#")]

    data = np.loadtxt(file_as_lines, comments="#", dtype=np.float64)

    # energy + flat intensity
    energies = data[:, 0]
    flat_intensity = data[:, 1]

    # parse metadata
    params = _parse_xy_head(metadata_lines)
    xy_dims = _parse_xy_dims(metadata_lines)

    if params.get("scan_mode") == "SnapshotFAT":
        # for SnapshotFAT mode the values_curve param is typically 1
        # therefore the number of energies must be calculated manually

        # first calculate the product of all non-energy dimensions (fallback to 1 for 1D case)
        n_other = np.prod([len(v) for v in xy_dims.values()], dtype=np.int64, initial=1).item()
        # then calculate the number of energies
        n_energy = len(flat_intensity) // n_other
    else:
        n_energy = int(params["values_curve"])

    energy_axis = np.linspace(energies[0], energies[n_energy - 1], n_energy)

    axes: list[Axis] = [Axis("eV", energy_axis)]

    # enforce order: nonenergy first, then others
    if SECOND_DIM_NAME in xy_dims:
        axes.append(Axis(SECOND_DIM_NAME, xy_dims[SECOND_DIM_NAME]))

    for name, values in xy_dims.items():
        if name != SECOND_DIM_NAME:
            axes.append(Axis(name, values))

    sizes = [ax.values.size for ax in axes]

    expected_size = int(np.prod(sizes))
    if flat_intensity.size != expected_size:
        msg = f"Data size mismatch: got {flat_intensity.size}, expected {expected_size}"
        raise ValueError(msg)

    intensity = _reshape_intensity(flat_intensity, axes)

    coords: dict[str, NDArray[np.float64]] = {ax.name: ax.values for ax in axes}
    dims = [ax.name for ax in axes]

    data_array = xr.DataArray(
        intensity,
        coords=coords,
        dims=dims,
        attrs=params.copy(),
    )

    data_array.attrs.update(kwargs)

    return data_array


def _parse_xy_head(header_lines: list[str]) -> dict[str, str | int | float]:
    """Parse header section into a typed parameter dictionary."""
    in_header_section = False
    params = {}

    for line in header_lines:
        # Search for header beginning
        if not in_header_section:
            if line.startswith(HEADER_SECTION_START):
                in_header_section = True
            continue

        # Search for header end
        if line.startswith(HEADER_SECTION_END):
            break

        # Read parameters
        m = HEADER_KV_RE.match(line)
        if m:
            key = m.group("key").strip()
            val = (m.group("value") or "").strip()
            params[key] = val

    params = clean_keys(params)

    # Cast the values
    for key, cast in HEADER_VALUES_TYPES.items():
        if key in params:
            params[key] = cast(params[key])

    return params


def _parse_xy_dims(header_lines: list[str]) -> dict[str, NDArray[np.float64]]:
    """Parse non-energy dimensions from header.

    Returns only dimensions present in the file:
    - no NonEnergyOrdinate → 1D
    - NonEnergyOrdinate only → 2D
    - NonEnergyOrdinate + Parameter → 3D
    """
    second_dim: list[float] = []
    second_name: str = SECOND_DIM_NAME

    third_dim: list[float] = []
    third_name: str | None = None

    second_dim_done: bool = False
    for line in header_lines:
        # --- third dimension (Parameter) ---
        m = PARAMETER_RE.match(line)
        if m:
            if third_name is None:
                third_name = m.group("name")
            third_dim.append(float(m.group("value")))
            continue

        # --- second dimension (NonEnergyOrdinate) ---
        if not second_dim_done:
            m = NONENERGY_RE.match(line)
            if m:
                value = float(m.group("value"))
                second_dim.append(value)
                if len(second_dim) > 1 and np.allclose(value, second_dim[0]):
                    second_dim.pop()
                    second_dim_done = True

    xy_dims: dict[str, NDArray[np.float64]] = {}

    if second_dim:
        xy_dims[second_name] = np.array(second_dim)

    if third_dim:
        xy_dims[third_name or THIRD_DIM_NAME] = np.array(third_dim)

    return clean_keys(xy_dims)


def _reshape_intensity(flat: NDArray[np.float64], axes: list[Axis]) -> NDArray[np.float64]:
    """Reshape flat intensity array into N-dimensional form.

    Data in Prodigy files is stored with fastest-changing axis last.
    This function reshapes and reorders it to match axes order:
    (energy, nonenergy, parameter, ...).
    """
    sizes = [len(ax.values) for ax in axes]
    shape = list(reversed(sizes))
    arr = flat.reshape(shape)
    return np.transpose(arr, tuple(reversed(range(len(shape)))))

