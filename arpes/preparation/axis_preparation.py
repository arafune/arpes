"""Utilities related to treatment of coordinate axes."""

from __future__ import annotations

import copy
import functools
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from scipy.ndimage import geometric_transform

from arpes.provenance import PROVENANCE, provenance, update_provenance
from arpes.utilities import lift_dataarray_to_generic
from arpes.utilities.normalize import normalize_to_spectrum

if TYPE_CHECKING:
    from collections.abc import Callable

    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from arpes._typing import DataType

__all__ = (
    "flip_axis",
    "normalize_dim",
    "dim_normalizer",
    "transform_dataarray_axis",
    "normalize_total",
    "sort_axis",
    "vstack_data",
)


@update_provenance("Build new DataArray/Dataset with an additional dimension")
def vstack_data(arr_list: list[DataType], new_dim: str) -> DataType:
    """Build a new DataArray | Dataset with an additional dimension.

    Args:
        arr_list (list[xr.Dataset] | list[xr.DataArray]): Source data series
        new_dim (str): name of axis as a new dimension

    Returns:
        xr.DataArray | xr.Dataset  Dataaa with an additional dimension
    """
    if not all((new_dim in data.attrs) for data in arr_list):
        assert all([(new_dim in data.coords for data in arr_list)])
    else:
        arr_list = [data.assign_coords({new_dim: data.attrs[new_dim]}) for data in arr_list]
    return xr.concat(arr_list, dim=new_dim)


@update_provenance("Sort Axis")
def sort_axis(data: xr.DataArray, axis_name: str) -> xr.DataArray:
    """Sorts slices of `data` along `axis_name` so that they lie in order.

    Args:
        data(xr.DataArray): [TODO:description]
        axis_name(str): [TODO:description]

    Returns(xr.DataArray):
        [TODO:description]
    """
    assert isinstance(data, xr.DataArray)
    copied = data.copy(deep=True)
    coord = data.coords[axis_name].values
    order = np.argsort(coord)

    copied.values = np.take(copied.values, order, axis=list(data.dims).index(axis_name))
    copied.coords[axis_name] = np.sort(copied.coords[axis_name])
    return copied


@update_provenance("Flip data along axis")
def flip_axis(arr: xr.DataArray, axis_name: str, *, flip_data: bool = True) -> xr.DataArray:
    """Flips the coordinate values along an axis w/o changing the data as well.

    Args:
        arr (xr.DataArray): [TODO:description]
        axis_name(str): [TODO:description]
        flip_data(bool): [TODO:description]

    Returns(xr.DataArray):
        [TODO:description]
    """
    coords = copy.deepcopy(arr.coords)
    coords[axis_name] = coords[axis_name][::-1]

    return xr.DataArray(
        np.flip(arr.values, arr.dims.index(axis_name)) if flip_data else arr.values,
        coords,
        arr.dims,
        attrs=arr.attrs,
    )


@lift_dataarray_to_generic
def normalize_dim(
    arr: DataType,
    dim_or_dims: str | list[str],
    *,
    keep_id: bool = False,
) -> xr.DataArray:
    """Normalizes the intensity.

    all values along axes other than `dim_or_dims` have the same value.

    The function normalizes so that the average value of cells in the output is 1.

    Args:
        arr: Input data which should be normalized
        dim_or_dims: The set of dimensions which should be normalized
        keep_id: Whether to reset the data's id after normalization

    Returns:
        The normalized data.
    """
    dims: list[str]
    dims = [dim_or_dims] if isinstance(dim_or_dims, str) else dim_or_dims
    assert isinstance(dims, list)

    summed_arr = arr.fillna(arr.mean()).sum([d for d in arr.dims if d not in dims])
    normalized_arr = arr / (summed_arr / np.prod(summed_arr.shape))

    to_return = xr.DataArray(normalized_arr.values, arr.coords, tuple(arr.dims), attrs=arr.attrs)

    if not keep_id and "id" in to_return.attrs:
        del to_return.attrs["id"]
    provenance_context: PROVENANCE = {
        "what": "Normalize axis or axes",
        "by": "normalize_dim",
        "dims": dims,
    }

    provenance(to_return, arr, provenance_context)

    return to_return


@update_provenance("Normalize total spectrum intensity")
def normalize_total(data: DataType, *, total_intensity: float = 1000000) -> xr.DataArray:
    """Normalizes data so that the total intensity is 1000000 (a bit arbitrary).

    Args:
        data(DataType): Input ARPES data
        total_intensity: value for normalizaiton

    Returns:
        xr.DataArray
    """
    data_array = normalize_to_spectrum(data)
    assert isinstance(data_array, xr.DataArray)
    return data_array / (data_array.sum(data.dims) / total_intensity)


def dim_normalizer(
    dim_name: str,
) -> Callable[[xr.Dataset | xr.DataArray], xr.DataArray | xr.Dataset]:
    """Safe partial application of dimension normalization.

    Args:
        dim_name (str): [TODO:description]
    """

    def normalize(arr: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
        if dim_name not in arr.dims:
            return arr
        return normalize_dim(arr, dim_name)

    return normalize


def transform_dataarray_axis(  # noqa: PLR0913
    func: Callable[[Incomplete], Incomplete],
    old_and_new_axis_names: tuple[str, str],
    new_axis: NDArray[np.float_] | xr.DataArray,
    dataset: xr.Dataset,
    prep_name: Callable[[str], str],
    transform_spectra: dict[str, xr.DataArray] | None = None,
    *,
    remove_old: bool = True,
) -> xr.Dataset:
    """Applies a function onto a DataArray axis.

    Args:
        func ([TODO:type]): [TODO:description]
        old_and_new_axis_names (tuple[str, str]) : old and new axis names as the tuple form
        new_axis ([TODO:type]): [TODO:description]
        dataset(xr.Dataset): [TODO:description]
        prep_name ([TODO:type]): [TODO:description]
        transform_spectra ([TODO:type]): [TODO:description]
        remove_old ([TODO:type]): [TODO:description]
    """
    old_axis_name, new_axis_name = old_and_new_axis_names

    ds = dataset.copy()
    if transform_spectra is None:
        # transform *all* DataArrays in the dataset that have old_axis_name in their dimensions.
        # In the standard usage, k is "spectra", v is xr.DataArray
        transform_spectra = {k: v for k, v in ds.data_vars.items() if old_axis_name in v.dims}
    assert isinstance(transform_spectra, dict)

    ds.coords[new_axis_name] = new_axis

    new_dataarrays = []
    for name in transform_spectra:
        dr = ds[name]

        old_axis = dr.dims.index(old_axis_name)
        shape = list(dr.sizes.values())
        shape[old_axis] = len(new_axis)
        new_dims = list(dr.dims)
        new_dims[old_axis] = new_axis_name

        g = functools.partial(func, axis=old_axis)
        output = geometric_transform(dr.values, g, output_shape=shape, output="f", order=1)

        new_coords = dict(dr.coords)
        new_coords.pop(old_axis_name)

        new_dataarray = xr.DataArray(
            output,
            coords=new_coords,
            dims=new_dims,
            attrs=dr.attrs.copy(),
            name=prep_name(str(dr.name)),
        )
        new_dataarrays.append(new_dataarray)
        if "id" in new_dataarray.attrs:
            del new_dataarray.attrs["id"]

        if remove_old:
            del ds[name]
        else:
            assert prep_name(name) != name, "You must make sure names don't collide"

    new_ds = xr.merge([ds, *new_dataarrays])

    new_ds.attrs.update(ds.attrs.copy())

    if "id" in new_ds:
        del new_ds.attrs["id"]
    provenance_context: PROVENANCE = {
        "what": "Transformed a Dataset coordinate axis",
        "by": "transform_dataarray_axis",
        "old_axis": old_axis_name,
        "new_axis": new_axis_name,
        "transformed_vars": list(transform_spectra.keys()),
    }

    provenance(new_ds, dataset, provenance_context)

    return new_ds
