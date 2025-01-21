"""Contains utilities for performing statistical operations in spectra and DataArrays."""

from __future__ import annotations

from logging import DEBUG, INFO

import xarray as xr

from arpes.debug import setup_logger
from arpes.provenance import update_provenance
from arpes.utilities import lift_dataarray_to_generic

__all__ = ("mean_and_deviation",)

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


@update_provenance("Calculate mean and standard deviation for observation axis")
@lift_dataarray_to_generic
def mean_and_deviation(
    data: xr.DataArray,  # data.name is used.
    axis: str = "",
    name: str = "",
) -> xr.Dataset:
    """Calculates the mean and standard deviation of a DataArray along an axis.

    The reduced axis corresponds to individual observations of a tensor/array valued quantity.
    This axis can be passed or inferred from a set of standard observation-like axes.

    New data variables are created with names `{name}` and `{name}_std`.
    If a name is not attached to the DataArray, it should be provided.

    Args:
        data: The input data.
        axis: The name of the dimension which we should perform the reduction along.
        name: The name of the variable which should be reduced. By default, uses `data.name`.

    Returns:
        A dataset with variables corresponding to the mean and standard error of each
        relevant variable in the input DataArray.  (Dimension is reduced.)
    """
    preferred_axes = ["bootstrap", "cycle", "idx"]
    name = "" if not data.name else name

    if not axis:
        for pref_axis in preferred_axes:
            if pref_axis in data.dims:
                axis = pref_axis
                break

    assert axis in data.dims
    return xr.Dataset(
        data_vars={name: data.mean(axis), name + "_std": data.std(axis)},
        attrs=data.attrs,
    )
