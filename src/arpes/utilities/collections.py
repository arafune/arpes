"""Utilities for comparing collections and some specialty collection types."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

__all__ = ("deep_update",)

T = TypeVar("T")


def deep_update(destination: dict[str, T], source: dict[str, T]) -> dict[str, T]:
    """Doesn't clobber keys further down trees like doing a shallow update would.

    Instead recurse down from the root and update as appropriate.

    Args:
        destination: dict object to be updated.
        source: source dict

    Returns:
        The destination item
    """
    for k, v in source.items():
        if isinstance(v, Mapping):
            destination[k] = deep_update(destination.get(k, {}), v)
        else:
            destination[k] = v

    return destination
