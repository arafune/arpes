"""Provides data provenance for PyARPES.

Most analysis routines built into PyARPES support provenance.
Of course, Python is a dynamic language and nothing can be
done to prevent the experimenter from circumventing the provenance scheme.

All the same, between analysis notebooks and the data provenenace provided by PyARPES,
we provide an environment with much higher standard for reproducible analysis than many
other current analysis environments.

This provenenace record is automatically exported when using the built in
plotting utilities. Additionally, passing `used_data` to the PyARPES `savefig`
wrapper allows saving provenance information even for bespoke plots created in
a Jupyter cell.

PyARPES also makes it easy to opt into data provenance for new analysis
functions by providing convenient decorators. These decorators inspect data passed at runtime
to look for and update provenance entries on arguments and return values.
"""

from __future__ import annotations

import contextlib
import datetime
import functools
import json
import uuid
import warnings
from datetime import UTC
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import xarray as xr

from . import VERSION
from ._typing import xr_types

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np
    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from ._typing import WORKSPACETYPE


class PROVENANCE(TypedDict, total=False):
    """TypedDict class for provenance.

    While any values can be stored in attrs["provenance"], but some rules exist.
    """

    record: PROVENANCE
    jupyter_context: list[str]
    parent_id: str | int | None
    parents_provenance: list[PROVENANCE] | PROVENANCE | str | None
    time: str
    version: str
    file: str
    what: str
    by: str
    args: list[PROVENANCE]
    #
    alpha: float  # derivative.curvature
    weight2d: float  # derivative.curvature
    directions: tuple[str, str]  # derivative.curvature
    axis: str  # derivative.dn_along_axis
    order: int  # derivative.dn_along_axis
    #
    sigma: dict[str, float]  # analysis.filters
    size: dict[str, float]  # analysis.filters
    use_pixel: bool  # analysis.filters
    #
    correction: list[NDArray[np.float_]]  # fermi_edge_correction
    #
    dims: Sequence[str]
    #
    old_axis: str
    new_axis: str


def attach_id(data: xr.DataArray | xr.Dataset) -> None:
    """Ensures that an ID is attached to a piece of data, if it does not already exist.

    IDs are generated at the time of identification in an analysis notebook. Sometimes a piece of
    data is created from nothing, and we might need to generate one for it on the spot.

    Args:
        data: The data to attach an ID to.
    """
    if "id" not in data.attrs:
        data.attrs["id"] = str(uuid.uuid1())


def provenance_from_file(
    child_arr: xr.DataArray | xr.Dataset,
    file: str,
    record: PROVENANCE,
) -> None:
    """Builds a provenance entry for a dataset corresponding to loading data from a file.

    This is used by data loaders at the start of an analysis.

    Args:
        child_arr: The array to update. This argument is modified.
        file: The file which provided the data. Should be a path or collection thereof.
        record: An annotation to add.
    """
    from .utilities.jupyter import get_recent_history

    if "id" not in child_arr.attrs:
        attach_id(child_arr)
    chile_provenance_context: PROVENANCE = {
        "record": record,
        "file": file,
        "jupyter_context": get_recent_history(5),
        "parents_provenance": "filesystem",
        "time": datetime.datetime.now(UTC).isoformat(),
        "version": VERSION,
    }

    child_arr.attrs["provenance"] = chile_provenance_context


def update_provenance(
    what: str,
    *,
    keep_parent_ref: bool = False,
):
    """A decorator that promotes a function to one that records data provenance.

    Args:
        what: Description of what transpired, to put into the record.
        keep_parent_ref: Whether to keep a pointer to the parents in the hierarchy or not.

    Returns:
        A decorator which can be applied to a function.
    """

    def update_provenance_decorator(fn: Callable) -> Callable[..., xr.DataArray | xr.Dataset]:
        """[TODO:summary].

        Args:
            fn: [TODO:description]
        """

        @functools.wraps(fn)
        def func_wrapper(*args: Incomplete, **kwargs: Incomplete) -> xr.DataArray | xr.Dataset:
            arg_parents = [v for v in args if isinstance(v, xr_types) and "id" in v.attrs]
            kwarg_parents = {
                k: v for k, v in kwargs.items() if isinstance(v, xr_types) and "id" in v.attrs
            }
            all_parents = arg_parents + list(kwarg_parents.values())
            result = fn(*args, **kwargs)
            # we do not want to record provenance or change the id if ``f`` opted not to do anything
            # to its input. This reduces the burden on client code by allowing them to return the
            # input without changing the 'id' attr
            result_not_identity = not any(p is result for p in all_parents)
            if isinstance(result, xr_types) and result_not_identity:
                if "id" in result.attrs:
                    del result.attrs["id"]
                provenance_fn = provenance
                if len(all_parents) > 1:
                    provenance_fn = provenance_multiple_parents
                if all_parents:
                    provenance_context: PROVENANCE = {
                        "what": what,
                        "by": fn.__name__,
                        "time": datetime.datetime.now(UTC).isoformat(),
                        "version": VERSION,
                    }
                    provenance_fn(
                        result,
                        all_parents,
                        provenance_context,
                        keep_parent_ref=keep_parent_ref,
                    )
            return result

        return func_wrapper

    return update_provenance_decorator


def save_plot_provenance(plot_fn: Callable) -> Callable:
    """A decorator that automates saving the provenance information for a particular plot.

    A plotting function creates an image or movie resource at some location on the
    filesystem.

    In order to hook into this decorator appropriately, because there is no way that I know
    of of temporarily overriding the behavior of the open builtin in order to monitor
    for a write.

    Args:
        plot_fn: A plotting function to decorate.

    Returns:
        A decorated copy of the input function which additionally saves provenance information.
    """
    from .utilities.jupyter import get_recent_history

    @functools.wraps(plot_fn)
    def func_wrapper(*args: Incomplete, **kwargs: Incomplete) -> Incomplete:
        """[TODO:summary].

        Args:
            args: [TODO:description]
            kwargs: [TODO:description]

        Returns:
            [TODO:description]
        """
        import arpes.config

        path = plot_fn(*args, **kwargs)
        if isinstance(path, str) and Path(path).exists():
            workspace: WORKSPACETYPE = arpes.config.CONFIG["WORKSPACE"]

            with contextlib.suppress(TypeError, KeyError):
                workspace_name: str = workspace["name"]

            if not workspace_name or workspace_name not in path:
                warnings.warn(
                    (
                        f"Plotting function {plot_fn.__name__} appears not to abide by "
                        "practice of placing plots into designated workspaces."
                    ),
                    stacklevel=2,
                )

            provenance_context = {
                "VERSION": VERSION,
                "time": datetime.datetime.now(UTC).isoformat(),
                "jupyter_context": get_recent_history(5),
                "name": plot_fn.__name__,
                "args": [
                    arg.attrs.get("provenance", {}) for arg in args if isinstance(arg, xr.DataArray)
                ],
                "kwargs": {
                    k: v.attrs.get("provenance", {})
                    for k, v in kwargs.items()
                    if isinstance(v, xr.DataArray)
                },
            }

            provenance_path = path + ".provenance.json"
            with Path(provenance_path).open("w") as f:
                json.dump(provenance_context, f, indent=2)

        return path

    return func_wrapper


def provenance(
    child_arr: xr.DataArray | xr.Dataset,
    parent_arr: xr.DataArray | xr.Dataset | list[xr.DataArray | xr.Dataset],
    record: PROVENANCE,
    *,
    keep_parent_ref: bool = False,
) -> None:
    """Updates the provenance in place for a piece of data with a single parent.

    Args:
        child_arr: The array to update. This argument is modified.
        parent_arr: The parent array.
        record: An annotation to add.
        keep_parent_ref: Whether we should keep a reference to the parents.
    """
    from .utilities.jupyter import get_recent_history

    if isinstance(parent_arr, list):
        assert len(parent_arr) == 1
        parent_arr = parent_arr[0]

    if "id" not in child_arr.attrs:
        attach_id(child_arr)

    parent_id = parent_arr.attrs.get("id")
    if parent_id is None:
        warnings.warn("Parent array has no ID.", stacklevel=2)

    if child_arr.attrs["id"] == parent_id:
        warnings.warn("Duplicate id for dataset %s" % child_arr.attrs["id"], stacklevel=2)

    child_arr.attrs["provenance"] = {
        "record": record,
        "jupyter_context": get_recent_history(5),
        "parent_id": parent_id,
        "parents_provanence": parent_arr.attrs.get("provenance"),
        "time": datetime.datetime.now(UTC).isoformat(),
        "version": VERSION,
    }

    if keep_parent_ref:
        child_arr.attrs["provenance"]["parent"] = parent_arr


def provenance_multiple_parents(
    child_arr: xr.DataArray | xr.Dataset,
    parents: list[xr.DataArray | xr.Dataset],
    record: PROVENANCE,
    *,
    keep_parent_ref: bool = False,
) -> None:
    """Updates provenance in place when there are multiple array-like data inputs.

    Similar to `provenance` updates the data provenance information for data with
    multiple sources or "parents". For instance, if you normalize a piece of data "X" by a metal
    reference "Y", then the returned data would list both "X" and "Y" in its history.

    Args:
        child_arr: The array to update. This argument is modified.
        parents: The collection of parents.
        record: An annotation to add.
        keep_parent_ref: Whether we should keep a reference to the parents.
    """
    from .utilities.jupyter import get_recent_history

    if "id" not in child_arr.attrs:
        attach_id(child_arr)

    if child_arr.attrs["id"] in {p.attrs.get("id", None) for p in parents}:
        warnings.warn("Duplicate id for dataset %s" % child_arr.attrs["id"], stacklevel=2)

    child_arr.attrs["provenance"] = {
        "record": record,
        "jupyter_context": get_recent_history(5),
        "parent_id": [p.attrs["id"] for p in parents],
        "parents_provenance": [p.attrs["provenance"] for p in parents],
        "time": datetime.datetime.now(UTC).isoformat(),
    }

    if keep_parent_ref:
        child_arr.attrs["provenance"]["parent"] = parents
