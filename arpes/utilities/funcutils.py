"""Some functional and UI functional programming utilities."""

from __future__ import annotations

import functools
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator, Sequence

    from _typeshed import Incomplete
    from numpy import ndarray
    from numpy._typing import NDArray

    from arpes._typing import DataType

__all__ = [
    "Debounce",
    "lift_dataarray_to_generic",
    "iter_leaves",
    "group_by",
    "cycle",
]

T = TypeVar("T")

P = ParamSpec("P")
R = TypeVar("R")


def cycle(sequence: Sequence[T]) -> Generator[T, None, None]:
    """Infinitely cycles a sequence."""
    while True:
        yield from sequence


def group_by(grouping: int | Generator, sequence: Sequence) -> list:
    """Permits partitining a sequence into sets of items, for instance by taking two at a time."""
    if isinstance(grouping, int):
        base_seq = [False] * grouping
        base_seq[-1] = True

        grouping_cycle = cycle(base_seq)
    else:
        grouping_cycle = grouping

    groups = []
    current_group = []
    for elem in sequence:
        current_group.append(elem)

        if (callable(grouping_cycle) and grouping_cycle(elem)) or next(grouping_cycle):
            groups.append(current_group)
            current_group = []

    if len(current_group):
        groups.append(current_group)

    return groups


def collect_leaves(tree: dict[str, Any], is_leaf: Incomplete = None) -> dict:
    """Produces a flat representation of the leaves.

    Leaves with the same key are collected into a list in the order of appearance,
    but this depends on the dictionary iteration order.

    Example:
    collect_leaves({'a': 1, 'b': 2, 'c': {'a': 3, 'b': 4}}) -> {'a': [1, 3], 'b': [2, 4]}

    Args:
        tree: The nested dictionary structured tree
        is_leaf: A condition to determine whether the current node is a leaf

    Returns:
        A dictionary with the leaves and their direct parent key.
    """

    def reducer(dd: dict, item: tuple[str, NDArray[np.float_]]) -> dict:
        dd[item[0]].append(item[1])
        return dd

    return functools.reduce(reducer, iter_leaves(tree, is_leaf), defaultdict(list))


def iter_leaves(
    tree: dict[str, Any],
    is_leaf: Callable[..., bool] | None = None,
) -> Iterator[tuple[str, ndarray]]:
    """Iterates across the leaves of a nested dictionary.

    Whether a particular piece
    of data counts as a leaf is controlled by the predicate `is_leaf`. By default,
    all nested dictionaries are considered not leaves, i.e. an item is a leaf if and
    only if it is not a dictionary.

    Iterated items are returned as key value pairs.

    As an example, you can easily flatten a nested structure with
    `dict(leaves(data))`
    """
    if is_leaf is None:

        def is_leaf(x: dict) -> bool:
            return not isinstance(x, dict)

    for k, v in tree.items():
        if is_leaf(v):
            yield k, v
        else:
            yield from iter_leaves(v)


def lift_dataarray_to_generic(
    func: Callable[Concatenate[DataType, P], DataType],
) -> Callable[Concatenate[DataType, P], DataType]:
    """A functorial decorator that lifts functions to operate over xarray types.

    (xr.DataArray, *args, **kwargs) -> xr.DataArray

    to one with signature

    A = xr.DataArray | xr.Dataset
    (A, *args, **kwargs) -> A

    i.e. one that will operate either over xr.DataArrays or xr.Datasets.
    """

    @functools.wraps(func)
    def func_wrapper(data: DataType, *args: P.args, **kwargs: P.kwargs) -> DataType:
        if isinstance(data, xr.DataArray):
            return func(data, *args, **kwargs)
        assert isinstance(data, xr.Dataset)
        new_vars = {datavar: func(data[datavar], *args, **kwargs) for datavar in data.data_vars}

        for var_name, var in new_vars.items():
            if isinstance(var, xr.DataArray) and var.name is None:
                var.name = var_name

        merged: xr.Dataset = xr.merge(new_vars.values())
        return merged.assign_attrs(data.attrs)

    return func_wrapper


class Debounce:
    """Wraps a function so that it can only be called periodically.

    Very useful for preventing expensive recomputation of some UI state when a user
    is performing a continuous action like a mouse pan or scroll or manipulating a
    slider.
    """

    def __init__(self, period: float) -> None:
        """Sets up the internal state for debounce tracking."""
        self.period = period  # never call the wrapped function more often than this (in seconds)
        self.count = 0  # how many times have we successfully called the function
        self.count_rejected = 0  # how many times have we rejected the call
        self.last: float = np.nan  # the last time it was called

    def reset(self) -> None:
        """Force a reset of the timer, aka the next call will always work."""
        self.last = np.nan

    def __call__(self, func: Callable[P, Any]) -> Callable[P, None]:
        """The wrapper call which defers execution if the function was actually called recently."""

        @functools.wraps(func)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> None:
            now = time.time()
            willcall = False
            if not np.isnan(self.last):
                # amount of time since last call
                delta = now - self.last
                willcall = delta >= self.period
            else:
                willcall = True  # function has never been called before

            if willcall:
                # set these first in case we throw an exception
                self.last = now  # don't use time.time()
                self.count += 1
                func(*args, **kwargs)  # call wrapped function
            else:
                self.count_rejected += 1

        return wrapped
