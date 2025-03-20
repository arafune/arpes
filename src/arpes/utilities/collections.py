"""Utilities for comparing collections and some specialty collection types.

This module provides functions to perform deep updates on dictionaries,
ensuring that nested dictionaries are merged correctly without overwriting
existing keys.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TypeVar, Union, cast

__all__ = ("deep_update",)

T = TypeVar("T")
NestedDict = MutableMapping[str, Union[T, "NestedDict[T]"]]


def deep_update(destination: NestedDict[T], source: NestedDict[T]) -> NestedDict[T]:
    """Recursively updates the destination dictionary with values from the source dictionary.

    This function ensures that nested dictionaries are merged correctly without
    overwriting existing keys.

    Args:
        destination: dict object to be updated.
        source: source dictkj

    Returns:
        The updated destination dictionary.

    Example:
        >>> dest = {'a': 1, 'b': {'c': 2}}
        >>> src = {'b': {'d': 3}, 'e': 4}
        >>> deep_update(dest, src)
        {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': 4}
    """
    for k, v in source.items():
        if isinstance(v, MutableMapping):
            sub_dict = cast("NestedDict[T]", destination.get(k, {}))
            destination[k] = deep_update(sub_dict, v)
        else:
            destination[k] = v
    return destination
