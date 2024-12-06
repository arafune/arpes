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
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, TypedDict, TypeVar

import xarray as xr

from . import VERSION
from ._typing import XrTypes
from .config import CONFIG

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence

    import numpy as np
    from numpy.typing import NDArray

    from ._typing import WorkSpaceType, XrTypes

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


_Provenance = TypedDict("_Provenance", {"with": str}, total=False)


class Provenance(_Provenance, total=False):
    """TypedDict class for provenance.

    While any values can be stored in attrs["provenance"], but some rules exist.
    """

    VERSION: str
    jupyter_notebook_name: str
    name: str

    record: Provenance
    jupyter_context: list[str]
    parent_id: str | int | None
    parents_provenance: list[Provenance] | Provenance | None
    time: str
    version: str
    file: str
    what: str
    by: str
    args: list[Provenance]
    alpha: float  # derivative.curvature
    weight2d: float  # derivative.curvature
    axis: str  # derivative.dn_along_axis
    order: int  # derivative.dn_along_axis
    sigma: dict[Hashable, float]  # analysis.filters
    size: dict[Hashable, float]  # analysis.filters
    use_pixel: bool  # analysis.filters
    correction: list[NDArray[np.float64]]  # fermi_edge_correction
    dims: Sequence[str]
    dim: str
    old_axis: str
    new_axis: str
    transformed_vars: list[str]
    parant_id: tuple[str, str]
    occupation_ratio: float | None
    correlation: bool
    decomposition_cls: str
    parsed_interpolation_points: list[dict[Hashable, float]]
    interpolation_points: list[Hashable | dict[Hashable, float]]
    axes: list[str]
    enhance_a: float


def attach_id(data: XrTypes) -> None:
    """Ensures that an ID is attached to a piece of data, if it does not already exist.

    IDs are generated at the time of identification in an analysis notebook. Sometimes a piece of
    data is created from nothing, and we might need to generate one for it on the spot.

    Args:
        data: The data to attach an ID to.
    """
    if "id" not in data.attrs:
        data.attrs["id"] = str(uuid.uuid1())


def provenance_from_file(
    child_arr: XrTypes,
    file: str,
    record: Provenance,
) -> None:
    """Builds a provenance entry for a dataset corresponding to loading data from a file.

    This is used by data loaders at the start of an analysis.

    Args:
        child_arr: The array to update. This argument is modified.
        file: The file which provided the data. Should be a path or collection thereof.
        record: An annotation to add.
    """
    from .utilities.jupyter import get_recent_history

    logger.debug("provenance from file")
    if "id" not in child_arr.attrs:
        attach_id(child_arr)
    child_provenance_context: Provenance = {
        "record": record,
        "file": file,
        "jupyter_context": get_recent_history(5),
        "time": datetime.datetime.now(UTC).isoformat(),
        "version": VERSION,
    }
    logger.debug(f"child_provenance_context: {child_provenance_context}")
    child_arr.attrs["provenance"] = child_provenance_context


P = ParamSpec("P")
R = TypeVar("R")


def update_provenance(
    what: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """A decorator that promotes a function to one that records data provenance.

    Args:
        what (str): Description of what transpired, to put into the record.

    Returns:
        A decorator which can be applied to a function.
    """

    def update_provenance_decorator(
        fn: Callable[P, R],
    ) -> Callable[P, R]:
        """A wrapper function that records data provenance for the execution of a function.

        Args:
            fn (Callable): The function for which provenance will be recorded.

        Returns:
            Callable: A function that has been extended to record data provenance.[TODO:summary].
        """

        @functools.wraps(fn)
        def func_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            arg_parents = [
                v for v in args if isinstance(v, xr.DataArray | xr.Dataset) and "id" in v.attrs
            ]
            kwarg_parents = {
                k: v
                for k, v in kwargs.items()
                if isinstance(v, xr.Dataset | xr.DataArray) and "id" in v.attrs
            }
            all_parents = arg_parents + list(kwarg_parents.values())
            result = fn(*args, **kwargs)
            # we do not want to record provenance or change the id if ``f`` opted not to do anything
            # to its input. This reduces the burden on client code by allowing them to return the
            # input without changing the 'id' attr
            result_not_identity = not any(p is result for p in all_parents)
            if isinstance(result, xr.DataArray | xr.Dataset) and result_not_identity:
                if "id" in result.attrs:
                    del result.attrs["id"]
                provenance_fn = provenance
                if len(all_parents) > 1:
                    provenance_fn = provenance_multiple_parents
                if all_parents:
                    provenance_context: Provenance = {
                        "what": what,
                        "by": fn.__name__,
                        "time": datetime.datetime.now(UTC).isoformat(),
                        "version": VERSION,
                    }
                    provenance_fn(
                        child_arr=result,
                        parents=all_parents,
                        record=provenance_context,
                    )
            return result

        return func_wrapper

    return update_provenance_decorator


def save_plot_provenance(plot_fn: Callable[P, R]) -> Callable[P, R]:
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
    def func_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """A wrapper function that records provenance information after generating a plot.

        Args:
            args: Positional arguments passed to `plot_fn`
            kwargs: Keyword arguments passed to `plot_fn`

        Returns:
            str: The file path where the plot is saved[TODO:summary].
        """
        path = plot_fn(*args, **kwargs)
        if isinstance(path, str) and Path(path).exists():
            workspace: WorkSpaceType = CONFIG["WORKSPACE"]

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

            provenance_context: Provenance = {
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
            with Path(provenance_path).open("w", encoding="UTF-8") as f:
                json.dump(provenance_context, f, indent=2)

        return path

    return func_wrapper


def provenance(
    child_arr: XrTypes,
    parents: list[XrTypes] | XrTypes,
    record: Provenance,
) -> None:
    """Updates the provenance in place for a piece of data with a single parent.

    Args:
        child_arr: The array to update. This argument is modified.
        parents: The parent array.
        record: An annotation to add.
    """
    from .utilities.jupyter import get_recent_history

    if isinstance(parents, list):
        assert len(parents) == 1
        parents = parents[0]

    if "id" not in child_arr.attrs:
        attach_id(child_arr)

    parent_id = parents.attrs.get("id")
    if parent_id is None:
        warnings.warn(
            "Parent array has no ID.",
            stacklevel=2,
        )

    if child_arr.attrs["id"] == parent_id:
        warnings.warn(
            f"Duplicate id for dataset {child_arr.attrs['id']}",
            stacklevel=2,
        )

    child_arr.attrs["provenance"] = {
        "record": record,
        "jupyter_context": get_recent_history(5),
        "parent_id": parent_id,
        "parents_provanence": parents.attrs.get("provenance"),
        "time": datetime.datetime.now(UTC).isoformat(),
        "version": VERSION,
    }


def provenance_multiple_parents(
    child_arr: XrTypes,
    parents: list[XrTypes] | XrTypes,
    record: Provenance,
) -> None:
    """Updates provenance in place when there are multiple array-like data inputs.

    Similar to `provenance` updates the data provenance information for data with
    multiple sources or "parents". For instance, if you normalize a piece of data "X" by a metal
    reference "Y", then the returned data would list both "X" and "Y" in its history.

    Args:
        child_arr: The array to update. This argument is modified.
        parents: The collection of parents.
        record: An annotation to add.
    """
    from .utilities.jupyter import get_recent_history

    if isinstance(parents, xr.Dataset | xr.DataArray):
        parents = [parents]
    if "id" not in child_arr.attrs:
        attach_id(child_arr)

    if child_arr.attrs["id"] in {p.attrs.get("id", None) for p in parents}:
        warnings.warn(
            f"Duplicate id for dataset {child_arr.attrs['id']}",
            stacklevel=2,
        )

    child_arr.attrs["provenance"] = {
        "record": record,
        "jupyter_context": get_recent_history(5),
        "parent_id": [p.attrs["id"] for p in parents],
        "parents_provenance": [p.attrs["provenance"] for p in parents],
        "time": datetime.datetime.now(UTC).isoformat(),
        "version": VERSION,
    }
