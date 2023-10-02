"""Some general purpose analysis routines otherwise defying categorization."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

import arpes.constants
import arpes.models.band
import arpes.utilities
import arpes.utilities.math
from arpes.fits import GStepBModel, broadcast_model
from arpes.provenance import update_provenance
from arpes.utilities import normalize_to_spectrum
from arpes.utilities.math import fermi_distribution

from .filters import gaussian_filter_arr

if TYPE_CHECKING:
    from arpes._typing import DataType

__all__ = (
    "normalize_by_fermi_distribution",
    "symmetrize_axis",
    "condense",
    "rebin",
    "fit_fermi_edge",
)


@update_provenance("Fit Fermi Edge")
def fit_fermi_edge(data: DataType, energy_range: slice | None = None) -> xr.Dataset:
    """Fits a Fermi edge.

    Not much easier than doing it manually, but this can be
    useful sometimes inside procedures where you don't want to reimplement this logic.

    Args:
        data(DataType): ARPES data
        energy_range(slice | tuple[float, float]): energy range for fitting

    Returns:
        The Fermi edge location.
    """
    if energy_range is None:
        energy_range = slice(-0.1, 0.1)

    broadcast_directions = list(data.dims)
    broadcast_directions.remove("eV")
    assert len(broadcast_directions) == 1  # for now we don't support more

    return broadcast_model(GStepBModel, data.sel(eV=energy_range), broadcast_directions[0])


@update_provenance("Normalized by the 1/Fermi Dirac Distribution at sample temp")
def normalize_by_fermi_distribution(
    data: DataType,
    max_gain: float = 0,
    rigid_shift: float = 0,
    instrumental_broadening: float = 0,
    total_broadening: float = 0,
) -> xr.DataArray:
    """Normalizes a scan by 1/the fermi dirac distribution.

    You can control the maximum gain with ``clamp``, and whether
    the Fermi edge needs to be shifted (this is for those desperate situations where you want
    something that "just works") via ``rigid_shift``.

    Args:
        data: Input
        max_gain: Maximum value for the gain. By default the value used
            is the mean of the spectrum.
        rigid_shift: How much to shift the spectrum chemical potential.
        instrumental_broadening: Instrumental broadening to use for
            convolving the distribution
        total_broadening: the value for total broadning.

    Pass the nominal value for the chemical potential in the scan. I.e. if the chemical potential is
    at BE=0.1, pass rigid_shift=0.1.

    Returns:
        Normalized DataArray
    """
    data_array = normalize_to_spectrum(data)
    if not total_broadening:
        distrib = fermi_distribution(
            data_array.coords["eV"].values - rigid_shift,
            total_broadening / arpes.constants.K_BOLTZMANN_EV_KELVIN,
        )
    else:
        distrib = fermi_distribution(
            data_array.coords["eV"].values - rigid_shift,
            data_array.S.temp,
        )

    # don't boost by more than 90th percentile of input, by default
    if not max_gain:
        max_gain = min(np.mean(data_array.values), np.percentile(data_array.values, 10))

    distrib[distrib < 1 / max_gain] = 1 / max_gain
    distrib_arr = xr.DataArray(distrib, {"eV": data_array.coords["eV"].values}, ["eV"])

    if not instrumental_broadening:
        distrib_arr = gaussian_filter_arr(distrib_arr, sigma={"eV": instrumental_broadening})

    return data_array / distrib_arr


@update_provenance("Symmetrize about axis")
def symmetrize_axis(
    data: DataType,
    axis_name: str,
    flip_axes: list[str] | None = None,
) -> xr.DataArray:
    """Symmetrizes data across an axis.

    It would be better ultimately to be able
    to implement an arbitrary symmetry (such as a mirror or rotational symmetry
    about a line or point) and to symmetrize data by that method.

    Args:
        data: input data
        axis_name: name of axis to be symmbetrized.
        flip_axes (list[str]): lis of axis name to be flipped flipping.

    Returns:
        Data after symmetrization procedure.
    """
    data = data.copy(deep=True)  # slow but make sure we don't bork axis on original
    data.coords[axis_name].values = data.coords[axis_name].values - data.coords[axis_name].values[0]

    selector = {}
    selector[axis_name] = slice(None, None, -1)
    rev = data.sel(**selector).copy()

    rev.coords[axis_name].values = -rev.coords[axis_name].values

    if flip_axes is None:
        flip_axes = []

    for axis in flip_axes:
        selector = {}
        selector[axis] = slice(None, None, -1)
        rev = rev.sel(**selector)
        rev.coords[axis].values = -rev.coords[axis].values

    return rev.combine_first(data)


@update_provenance("Condensed array")
def condense(data: xr.DataArray) -> xr.DataArray:
    """Clips the data so that only regions where there is substantial weight are included.

    In practice this usually means selecting along the ``eV`` axis, although other selections
    might be made.

    Args:
        data: xarray.DataArray

    Returns:
        The clipped data.
    """
    if "eV" in data.dims:
        data = data.sel(eV=slice(None, 0.05))

    return data


@update_provenance("Rebinned array")
def rebin(
    data: DataType,
    shape: dict[str, int] | None = None,
    bin_width: dict[str, int] | None = None,
    method: Literal["sum", "mean"] = "sum",
    **kwargs: int,
) -> DataType:
    """Rebins the data onto a different (smaller) shape.

    (xarray groupby_bins is used internally)

    By default the behavior is to
    split the data into chunks that are integrated over.

    When both ``shape`` and ``bin_width`` are supplied, ``shape`` is used.

    Dimensions corresponding to missing entries in ``shape`` or ``reduction`` will not
    be changed.

    Args:
        data: ARPES data
        shape(dict[str, int]): Target shape
          (key is dimension (coords) name, the value is the size of the coords after rebinning.)
          The priority is higer than that of the reduction argument.
        bin_width(dict[str, int]): Factor to reduce each dimension by
          The dict key is dimension name and it's value is the binning width in pixel.
        method: sum or mean after groupby_bins  (default sum)
        **kwargs: Treated as bin_width. Like as eV=2, phi=3 to set.

    Returns:
        The rebinned data.
    """
    assert isinstance(data, xr.DataArray | xr.Dataset)
    if bin_width is None:
        bin_width = {}
    for k in kwargs:
        if k in data.dims:
            bin_width[k] = kwargs[k]
    if shape is None:
        shape = {}
        for k, v in bin_width.items():
            shape[k] = len(data.coords[k]) // v
    assert bool(shape), "Set shape/bin_width"
    for bin_axis, bins in shape.items():
        data = _bin(data, bin_axis, bins, method)
    return data


def _bin(data: DataType, bin_axis: str, bins: int, method: Literal["sum", "mean"]) -> DataType:
    original_left, original_right = (
        data.coords[bin_axis].values[0],
        data.coords[bin_axis].values[-1],
    )
    original_region = original_right - original_left
    if method == "sum":
        data = data.groupby_bins(bin_axis, bins).sum().rename({bin_axis + "_bins": bin_axis})
    elif method == "mean":
        data = data.groupby_bins(bin_axis, bins).mean().rename({bin_axis + "_bins": bin_axis})
    else:
        msg = "method must be sum or mean"
        raise TypeError(msg)
    left = data.coords[bin_axis].values[0].left
    right = data.coords[bin_axis].values[0].right
    left = left + original_region * 0.001
    medium_values = [
        (left + right) / 2,
        *[(b.left + b.right) / 2 for b in data.coords[bin_axis].values[1:]],
    ]
    data.coords[bin_axis] = np.array(medium_values)
    return data
