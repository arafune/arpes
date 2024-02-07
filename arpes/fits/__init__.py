"""Utilities related to curve-fitting of ARPES data and xarray format data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict

from .fit_models import *
from .utilities import broadcast_model, result_to_hints

NAN_POLICY = Literal["raise", "propagate", "omit"]


class ModelARGS(TypedDict, total=False):
    """KWargs for lf.Model."""

    independent_vars: list[str]
    param_names: list[str]
    nan_policy: NAN_POLICY
    prefix: str
    name: str
