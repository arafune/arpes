"""Provides the core IO facilities supported by PyARPES.

The most important here are the data loading functions (load_data, load_example_data).
and pickling utilities.

Heavy lifting is actually performed by the plugin definitions which know how to ingest
different data formats into the PyARPES data model.

TODO: An improvement could be made to the example data if served
over a network and someone was willing to host a few larger pieces
of data.
"""

from __future__ import annotations

import copy
import pickle
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from logging import DEBUG, INFO
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import xarray as xr

from .debug import setup_logger
from .endstations.registry import resolve_endstation
from .example_data.mock import build_mock_tarpes
from .provenance import provenance_multiple_parents
from .setting import CONFIG

if TYPE_CHECKING:
    from _typeshed import Incomplete

    from arpes._typing import ScanDesc, XrTypes


__all__ = ("easy_pickle", "list_pickles", "load_data", "load_example_data", "load_scan", "stitch")


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


def load_data(
    file: str | Path,
    location: str | None = None,
    **kwargs: Incomplete,
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


def file_for_pickle(name: str) -> Path | str:
    here = Path()
    if CONFIG["WORKSPACE"] and "path" in CONFIG["WORKSPACE"]:
        here = Path(CONFIG["WORKSPACE"]["path"])
    path = here / "picklejar" / f"{name}.pickle"
    path.parent.mkdir(exist_ok=True)
    return str(path)


def load_pickle(name: str) -> object:
    """Loads a workspace local pickle. Inverse to `save_pickle`."""
    with Path(file_for_pickle(name)).open("rb") as file:
        return pickle.load(file)  # noqa: S301


def save_pickle(data: object, name: str) -> None:
    """Saves a workspace local pickle. Inverse to `load_pickle`."""
    with Path(file_for_pickle(name)).open("wb") as pickle_file:
        pickle.dump(data, pickle_file)


def easy_pickle(data_or_str: str | object, name: str = "") -> object:
    """A convenience function around pickling.

    Provides a workspace scoped associative set of named pickles which
    can be used for

    Examples:
        Retaining analysis results between sessions.

        Sharing results between workspaces.

        Caching expensive or interim work.

    For reproducibility reasons, you should generally prefer to
    duplicate anaysis results using common code to prevent stale data
    dependencies, but there are good reasons to use pickling as well.

    This function knows whether we are pickling or unpickling depending on
    whether one or two arguments are provided.

    Args:
        data_or_str: If saving, the data to be pickled. If loading, the name of the pickle to load.
        name: If saving (non-None value), the name to associate. Defaults to None.

    Returns:
        None if name is not None, which indicates that we are saving data.
        Otherwise, returns the unpickled value associated to `name`.
    """
    # we are loading data
    if isinstance(data_or_str, str) or not name:
        assert isinstance(data_or_str, str)
        return load_pickle(data_or_str)
    # we are saving data
    assert isinstance(name, str)
    save_pickle(data_or_str, name)
    return None


def list_pickles() -> list[str]:
    """Generates a summary list of (workspace-local) pickled results and data.

    Returns:
        A list of the named pickles, suitable for passing to `easy_pickle`.
    """
    return [str(s.stem) for s in Path(file_for_pickle("just-a-pickle")).parent.glob("*.pickle")]


def load_scan(
    scan_desc: ScanDesc,
    *,
    retry: bool = True,
    **kwargs: Incomplete,
) -> xr.Dataset:
    """Resolves a plugin and delegates loading a scan.

    This is used internally by `load_data` and should not be invoked directly
    by users.

    Determines which data loading class is appropriate for the data,
    shuffles a bit of metadata, and calls the .load function on the
    retrieved class to start the data loading process.

    Args:
        scan_desc: Information identifying the scan, typically the full path.
        retry: Used to attempt a reload of plugins and subsequent data load attempt.
        kwargs: pass to the endstation.load(scan_dec, **kwargs)

    Returns:
        Loaded and normalized ARPES scan data.
    """
    note: dict[str, str | float] | ScanDesc = scan_desc.get("note", scan_desc)
    full_note: ScanDesc = copy.deepcopy(scan_desc)
    assert isinstance(note, dict)
    full_note.update(note)

    endstation_cls = resolve_endstation(retry=retry, **full_note)
    logger.debug(f"Using plugin class {endstation_cls}")

    key: Literal["file", "path"] = "file" if "file" in scan_desc else "path"

    file = scan_desc[key]
    try:
        file_number: int = int(str(file))
        file = endstation_cls.find_first_file(file_number)
        scan_desc[key] = file
    except ValueError:
        pass

    logger.debug(f"Loading {scan_desc}")
    endstation = endstation_cls()
    return endstation.load(scan_desc, **kwargs)
