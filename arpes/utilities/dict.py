"""Utilities for modifying, iterating over, and transforming dictionaries."""

from __future__ import annotations

import re
from typing import Any, TypeVar

from .xarray import lift_dataarray_attrs, lift_datavar_attrs

__all__ = (
    "rename_keys",
    "clean_keys",
    "rename_dataarray_attrs",
    "clean_datavar_attribute_names",
    "clean_attribute_names",
)


def _rename_key(
    d: dict[str, Any],
    original_name_k: str,
    new_name_k: str,
) -> None:
    if original_name_k in d:
        d[new_name_k] = d[original_name_k]
        del d[original_name_k]


def rename_keys(
    d: dict[str, Any],
    keys_dict: dict[str, str],
) -> dict[str, Any]:
    """Renames all the keys of `d` according to the remapping in `keys_dict`.

    Args:
        d (dict): dict object (Suppose the attrs)
        keys_dict(dict[str, str]):  {original_name_k: new_name_k}

    Returns:
        [TODO:description]
    """
    d = d.copy()
    for k, nk in keys_dict.items():
        _rename_key(d, k, nk)

    return d


T = TypeVar("T")


def clean_keys(d: dict[str, T]) -> dict[str, T]:
    """Renames dict key to fit Pythonic more.

    Args:
        d (dict): dictionary to be cleaned.

    Returns:
        dict object whose key is cleaned.
    """

    def clean_single_key(k: str) -> str:
        k = k.replace(" ", "_")
        k = k.replace(".", "_")
        k = k.lower()
        k = re.sub(r"[()/?]", "_", k)
        return k.replace("__", "_")

    return dict(zip([clean_single_key(k) for k in d], d.values(), strict=True))


rename_dataarray_attrs = lift_dataarray_attrs(rename_keys)
clean_attribute_names = lift_dataarray_attrs(clean_keys)

clean_datavar_attribute_names = lift_datavar_attrs(clean_keys)
