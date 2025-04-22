"""Provides the core IO facilities supported by PyARPES.

The most important here are the data loading functions (load_data, load_example_data).
and pickling utilities.

Heavy lifting is actually performed by the plugin definitions which know how to ingest
different data formats into the PyARPES data model.

TODO: An improvement could be made to the example data if served
over a network and someone was willing to host a few larger pieces
of data.
"""

import json
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from functools import singledispatch
from logging import DEBUG, INFO
from pathlib import Path
from typing import Any, Literal, TypedDict, Unpack

import numpy as np
import pandas as pd
import xarray as xr

from arpes._typing import XrTypes, DataType

from .debug import setup_logger
from .endstations import ScanDesc, load_scan
from .example_data.mock import build_mock_tarpes

__all__ = (
    "load_custom_netcdf",
    "load_data",
    "load_example_data",
    "save_custom_netcdf",
    "stitch",
)


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


def load_data(
    file: str | Path,
    location: str | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Loads a piece of data using available plugins. This the user facing API for data loading.

    Args:
        file (str | Path): An identifier for the file which should be loaded, i.e., the file path.
        location (str | type[EndstationBase]): The name of the endstation/plugin to use.
            You should try to provide one. If None is provided, the loader
            will try to find an appropriate one based on the file extension and brute force.
            This will be slower and can be error prone in certain circumstances.
        kwargs: pass to load_scan
            Optionally, you can pass a loading plugin (the class) through this kwarg and directly
            specify the class to be used.


    Returns:
        The loaded data. Ideally, data which is loaded through the plugin system should be highly
        compliant with the PyARPES data model and should work seamlessly with PyARPES analysis code.
    """
    try:
        file = int(str(file))  # type: ignore[assignment]  # pragma: no cover
        warnings.warn(
            "This functionality, the data specified by number,  will be removed.",
            DeprecationWarning,
            stacklevel=2,
        )
    except ValueError:
        assert isinstance(file, (str | Path))
        file = str(Path(file).absolute())

    desc: ScanDesc = {
        "file": file,  # type:ignore[typeddict-item]
        "location": location,  # type:ignore[typeddict-item]
    }

    if location is None:
        desc.pop("location")
        warnings.warn(
            (
                "You should provide a location indicating the endstation or "
                "instrument used directly en loading data without a dataset."
                "We are going to do our best but no guarantees."
            ),
            stacklevel=2,
        )
    logger.debug(f"contents of desc: {desc}")
    return load_scan(desc, **kwargs)


DATA_EXAMPLES: dict[str, tuple[str, str]] = {
    "cut": ("ALG-MC", "cut.fits"),
    "map": ("example_data", "fermi_surface.nc"),
    "photon_energy": ("example_data", "photon_energy.nc"),
    "nano_xps": ("example_data", "nano_xps.nc"),
    "temperature_dependence": ("example_data", "temperature_dependence.nc"),
    "cut2": ("SPD", "example_itx_data.itx"),
    "cut3": ("DSNP_UMCS", "BLGr_K_cut.xy"),
    "map2": ("DSNP_UMCS", "BLGr_GK_map.xy"),
}


def load_example_data(example_name: str = "cut") -> xr.Dataset:
    """Provides sample data for executable documentation.

    Args:
        example_name: (cut, cut2, cut3, map, map2, photon_energy, nano_xps, temperature_dependence)

    Returns:
        example DataSet
    """
    if example_name not in DATA_EXAMPLES:
        msg = f"Could not find requested example_name: {example_name}."
        msg += f"Please provide one of {list(DATA_EXAMPLES.keys())}"
        raise KeyError(msg)

    location, example = DATA_EXAMPLES[example_name]
    logger.debug(f"location:{location}")
    file = Path(__file__).parent / "example_data" / example
    return load_data(file=file, location=location)


@dataclass
class ExampleData:
    @property
    def cut(self) -> xr.Dataset:
        return load_example_data("cut")

    @property
    def map(self) -> xr.Dataset:
        return load_example_data("map")

    @property
    def photon_energy(self) -> xr.Dataset:
        return load_example_data("photon_energy")

    @property
    def nano_xps(self) -> xr.Dataset:
        return load_example_data("nano_xps")

    @property
    def temperature_dependence(self) -> xr.Dataset:
        return load_example_data("temperature_dependence")

    @property
    def cut2(self) -> xr.Dataset:
        return load_example_data("cut2")

    @property
    def cut3(self) -> xr.Dataset:
        return load_example_data("cut3")

    @property
    def map2(self) -> xr.Dataset:
        return load_example_data("map2")

    @property
    def t_arpes(self) -> list[xr.DataArray]:
        return build_mock_tarpes()


example_data = ExampleData()


def stitch(
    df_or_list: list[str] | pd.DataFrame,
    attr_or_axis: str | list[float] | tuple[float, ...],
    built_axis_name: str = "",
    *,
    sort: bool = True,
) -> XrTypes:
    """Stitches together a sequence of scans or a DataFrame.

    Args:
        df_or_list(list[str] | pd.DataFrame): The list of the files to load
        attr_or_axis(str|list[float]|tuple[float, ...]): Coordinate or attribute in order to
                      promote to an index. I.e. if 't_a' is specified, we will create a new axis
                      corresponding to the temperature and concatenate the data along this axis
        built_axis_name: The name of the concatenated output dimensions
        sort: Whether to sort inputs to the concatenation according to their `attr_or_axis` value.

    Returns:
        The concatenated data.
    """
    list_of_files = _df_or_list_to_files(df_or_list)
    if not built_axis_name:
        assert isinstance(attr_or_axis, str)
        built_axis_name = attr_or_axis
    if not list_of_files:
        msg = "Must supply at least one file to stitch"
        raise ValueError(msg)

    loaded: list[xr.Dataset] = []
    i = 0
    for f in list_of_files:
        data: xr.Dataset = load_data(f)
        value: xr.DataArray | float | None = None
        if isinstance(attr_or_axis, list | tuple):
            value = attr_or_axis[i]
        elif attr_or_axis in data.attrs:
            value = data.attrs[attr_or_axis]
        elif attr_or_axis in data.coords:
            value = data.coords[attr_or_axis]
        loaded.append(data.assign_coords({built_axis_name: value}))

    assert all(isinstance(data, xr.DataArray) for data in loaded) or all(
        isinstance(data, xr.Dataset) for data in loaded
    )

    if sort:
        loaded.sort(key=lambda x: np.min(x.coords[built_axis_name].values))
    assert isinstance(loaded, Iterable)
    concatenated = xr.concat(loaded, dim=built_axis_name)
    if "id" in concatenated.attrs:
        del concatenated.attrs["id"]
    from .provenance import provenance_multiple_parents

    provenance_multiple_parents(
        concatenated,
        loaded,
        {
            "what": "Stitched together separate datasets",
            "by": "stitch",
            "dim": built_axis_name,
        },
    )
    return concatenated


def _df_or_list_to_files(
    df_or_list: list[str] | pd.DataFrame,
) -> list[str]:
    """Helper function for stitch.

    Args:
        df_or_list(pd.DataFrame, list): input data file

    Returns: (list[str])
        list of files to stitch.
    """
    if isinstance(df_or_list, pd.DataFrame):
        return list(df_or_list.index)
    assert not isinstance(
        df_or_list,
        list | tuple,
    ), "Expected an iterable for a list of the scans to stitch together"
    return list(df_or_list)


class ToNetCDFParams(TypedDict, total=False):
    """Typed dictionary for the parameters of the to_netcdf function.

    There are many parameters are available, but here only two are defined.
    """

    mode: Literal["w", "a"]
    engine: Literal["netcdf4", "h5netcdf"]


@singledispatch
def save_custom_netcdf(
    obj: xr.DataArray | xr.Dataset | xr.DataTree,
    path: str | Path,
    **kwargs: Unpack[ToNetCDFParams],
) -> None:
    """Save an xarray object (DataArray, Dataset, or DataTree) to a NetCDF file.

    Args:
        obj (xr.DataArray | xr.Dataset | xr.DataTree): The xarray object to save.
        path (str | Path): The file path to write the NetCDF file to.
        **kwargs: Additional keyword arguments passed to `to_netcdf()`.

    Returns:
        None
    """
    del path, kwargs
    msg = f"Unsupported type: {type(obj)}"
    raise NotImplementedError(msg)


def _jsonify_attrs(obj: DataType) -> DataType:
    if hasattr(obj, "attrs"):
        obj.attrs = {"__json__": json.dumps(obj.attrs)}
    return obj


@save_custom_netcdf.register
def _(
    obj: xr.DataArray,
    path: str | Path,
    **kwargs: Unpack[ToNetCDFParams],
) -> None:
    """Save an xarray object (DataArray, Dataset, or DataTree) to a NetCDF file.

    encoding all attrs as JSON strings.

    Args:
        obj (xr.DataArray | xr.Dataset | xr.DataTree): The xarray object to save.
        path (str | Path): The file path to write the NetCDF file to.
        **kwargs: Additional keyword arguments passed to `to_netcdf()`.

    Returns:
        None
    """
    _jsonify_attrs(obj).to_netcdf(path, **kwargs)


@save_custom_netcdf.register
def _(
    obj: xr.Dataset,
    path: str | Path,
    **kwargs: Unpack[ToNetCDFParams],
) -> None:
    """Save an xarray object (DataArray, Dataset, or DataTree) to a NetCDF file.

    encoding all attrs as JSON strings.

    Args:
        obj (xr.DataArray | xr.Dataset | xr.DataTree): The xarray object to save.
        path (str | Path): The file path to write the NetCDF file to.
        **kwargs: Additional keyword arguments passed to `to_netcdf()`.

    Returns:
        None
    """
    ds_jsonified = xr.Dataset({name: _jsonify_attrs(var) for name, var in obj.data_vars.items()})
    _jsonify_attrs(ds_jsonified).to_netcdf(path, **kwargs)


@save_custom_netcdf.register
def _(
    obj: xr.DataTree,
    path: str | Path,
    **kwargs: Unpack[ToNetCDFParams],
) -> None:
    def jsonify_attrs_dataset(obj: xr.Dataset) -> xr.Dataset:
        ds_jsonified = xr.Dataset(
            {name: _jsonify_attrs(var) for name, var in obj.data_vars.items()},
        )
        assert isinstance(ds_jsonified, xr.Dataset)
        return _jsonify_attrs(ds_jsonified)

    new_tree = obj.map_over_datasets(jsonify_attrs_dataset)
    assert isinstance(new_tree, xr.DataTree)
    new_tree.to_netcdf(path, **kwargs)


def load_custom_netcdf(
    path: str | Path,
    **kwargs: Unpack[ToNetCDFParams],
) -> xr.DataArray | xr.Dataset | xr.DataTree:
    """Load an xarray object from a NetCDF file and decode all JSON-encoded attrs.

    The object type (DataArray, Dataset, or DataTree) is determined automatically


    from the saved metadata.

    Args:
        path (str | Path): The file path to read the NetCDF file from.
        **kwargs: Additional keyword arguments passed to the appropriate `open_*` function.

    Returns:
        xr.DataArray | xr.Dataset | xr.DataTree: The loaded and decoded xarray object.
    """
    obj = xr.open_dataset(path, **kwargs)

    # Check and decode attrs if necessary
    if hasattr(obj, "attrs"):
        obj.attrs = {k: (json.loads(v) if isinstance(v, str) else v) for k, v in obj.attrs.items()}

    # If it's a DataTree, reconstruct it
    if isinstance(obj, xr.DataTree):
        obj = xr.open_datatree(path, **kwargs)

    return obj
