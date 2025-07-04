"""Utilities for modifying, iterating over, and transforming dictionaries."""

from __future__ import annotations

import re
from typing import Any, TypeVar

__all__ = (
    "clean_keys",
    "rename_keys",
)


T = TypeVar("T")


def _rename_key(
    d: dict[str, Any],
    original_name_k: str,
    new_name_k: str,
) -> None:
    if original_name_k in d:
        d[new_name_k] = d[original_name_k]
        del d[original_name_k]


def rename_keys(
    d: dict[str, T],
    keys_dict: dict[str, str],
) -> dict[str, T]:
    """Renames all the keys of `d` according to the remapping in `keys_dict`.

    Args:
        d (dict): dict object.
        keys_dict(dict[str, str]):  {original_name_k: new_name_k}

    Returns:
        dict[str, T]: A new dictionary with renamed keys.
    """
    d = d.copy()
    for k, nk in keys_dict.items():
        _rename_key(d, k, nk)

    return d


def clean_keys(d: dict[str, T]) -> dict[str, T]:
    """Renames dict key to fit Pythonic more.

    Args:
        d (dict): dictionary to be cleaned.

    Returns:
        dict[str, T]: A new dictionary with cleaned key.
    """

    def clean_single_key(k: str) -> str:
        k = k.replace(" ", "_")
        k = k.replace(".", "_")
        k = k.lower()
        k = re.sub(r"[()/?]", "_", k)
        return k.replace("__", "_")

    return dict(zip([clean_single_key(k) for k in d], d.values(), strict=True))
