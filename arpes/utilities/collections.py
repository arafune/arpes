"""Utilities for comparing collections and some specialty collection types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeVar

__all__ = (
    "deep_equals",
    "deep_update",
    "MappableDict",
)


class MappableDict(dict):
    """Like dict except that +, -, *, / are cascaded to values."""

    def __add__(self, other: MappableDict) -> MappableDict:
        """Applies `+` onto values."""
        if set(self.keys()) != set(other.keys()):
            msg = "You can only add two MappableDicts with the same keys."
            raise ValueError(msg)

        return MappableDict({k: self.get(k) + other.get(k) for k in self})

    def __sub__(self, other: MappableDict) -> MappableDict:
        """Applies `-` onto values."""
        if set(self.keys()) != set(other.keys()):
            msg = "You can only subtract two MappableDicts with the same keys."
            raise ValueError(msg)
        return MappableDict({k: self.get(k) - other.get(k) for k in self})

    def __mul__(self, other: MappableDict) -> MappableDict:
        """Applies `*` onto values."""
        if set(self.keys()) != set(other.keys()):
            msg = "You can only multiply two MappableDicts with the same keys."
            raise ValueError(msg)
        return MappableDict({k: self.get(k) * other.get(k) for k in self})

    def __truediv__(self, other: MappableDict) -> MappableDict:
        """Applies `/` onto values."""
        if set(self.keys()) != set(other.keys()):
            msg = "You can only divide two MappableDicts with the same keys."
            raise ValueError(msg)
        return MappableDict({k: self.get(k) / other.get(k) for k in self})

    def __floordiv__(self, other: MappableDict) -> MappableDict:
        """Applies `//` onto values."""
        if set(self.keys()) != set(other.keys()):
            msg = "You can only divide (//) two MappableDicts with the same keys."
            raise ValueError(msg)
        return MappableDict({k: self.get(k) // other.get(k) for k in self})

    def __neg__(self) -> MappableDict:
        """Applies unary negation onto values."""
        return MappableDict({k: -self.get(k) for k in self})


def deep_update(destination: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Doesn't clobber keys further down trees like doing a shallow update would.

    Instead recurse down from the root and update as appropriate.

    Args:
        destination:
        source:

    Returns:
        The destination item
    """
    for k, v in source.items():
        if isinstance(v, Mapping):
            destination[k] = deep_update(destination.get(k, {}), v)
        else:
            destination[k] = v

    return destination


T = TypeVar("T")


def deep_equals(
    a: T | Sequence[T] | set[T] | Mapping[str, T] | None,
    b: T | Sequence[T] | set[T] | Mapping[str, T] | None,
) -> bool:
    """An equality check that looks into common collection types."""
    if not isinstance(b, type(a)):
        return False

    if isinstance(a, str | float | int | None | set):
        return a == b

    if isinstance(a, Sequence) and isinstance(b, Sequence):
        if len(a) != len(b):
            return False
        return all(deep_equals(item_a, item_b) for item_a, item_b in zip(a, b, strict=True))

    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if set(a.keys()) != set(b.keys()):
            return False

        for k in a:
            item_a, item_b = a[k], b[k]

            if not deep_equals(item_a, item_b):
                return False

        return True
    raise TypeError
