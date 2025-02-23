"""Utilities used in broadcast fitting."""

from __future__ import annotations

import functools
import operator
import warnings
from collections.abc import Iterable, Sequence
from functools import singledispatch
from logging import DEBUG, INFO
from string import ascii_lowercase
from typing import TYPE_CHECKING, Any, Literal, TypeGuard

import lmfit as lf
import xarray as xr

from arpes.debug import setup_logger

if TYPE_CHECKING:
    from arpes.fits import ParametersArgs, XModelMixin

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


def unwrap_params(
    params: dict[str, ParametersArgs],
    iter_coordinate: dict[str, slice | float],
) -> dict[str, Any]:
    """Inspects arraylike parameters and extracts appropriate value for current fit."""
    return {k: _transform_or_walk(v, iter_coordinate) for k, v in params.items()}


@singledispatch
def _transform_or_walk(v: object, iter_coordinate: dict[str, slice | float]) -> object:
    """Default case: return the value as is."""
    del iter_coordinate
    return v


@_transform_or_walk.register
def _(v: dict, iter_coordinate: dict[str, slice | float]) -> dict:
    return unwrap_params(v, iter_coordinate)


@_transform_or_walk.register
def _(v: xr.DataArray, iter_coordinate: dict[str, slice | float]) -> float:
    return v.sel(iter_coordinate, method="nearest").item()


@_transform_or_walk.register
def _(v: Iterable, iter_coordinate: dict[str, slice | float]) -> Iterable:
    del iter_coordinate
    return v


def apply_window(
    data: xr.DataArray,
    cut_coords: dict[str, float | slice],
    window: xr.DataArray | None = None,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Cuts data inside a specified window.

    Because we allow passing an array of windows, we need to first try to find
    the right one for the current fit before application.

    If there's no window, this acts as the identity function on the data.
    """
    cut_data = data.sel(cut_coords)
    original_cut_data = cut_data

    if isinstance(window, xr.DataArray):
        window_item = window.sel(cut_coords).item()
        if isinstance(window_item, slice):
            cut_data = cut_data.sel({cut_data.dims[0]: window_item})
    return cut_data, original_cut_data


def _parens_to_nested(items: list) -> list:
    """Turns a flat list with parentheses tokens into a nested list."""
    parens = [
        (
            t,
            idx,
        )
        for idx, t in enumerate(items)
        if isinstance(t, str) and t in "()"
    ]
    if parens:
        first_idx, last_idx = parens[0][1], parens[-1][1]
        if parens[0][0] != "(" or parens[-1][0] != ")":
            msg = "Parentheses do not match!"
            raise ValueError(msg)

        return (
            items[0:first_idx]
            + [_parens_to_nested(items[first_idx + 1 : last_idx])]
            + items[last_idx + 1 :]
        )
    return items


def reduce_model_with_operators(
    models: Sequence[lf.Model | Literal["+", "*", "-", "/"]],
) -> lf.Model:
    """Combine models according to mathematical operators."""
    if isinstance(models, tuple):
        return models[0](prefix=f"{models[1]}_", nan_policy="omit")

    if isinstance(models, list) and len(models) == 1:
        return reduce_model_with_operators(models[0])

    left, op, right = models[0], models[1], models[2:]
    left, right = reduce_model_with_operators(left), reduce_model_with_operators(right)
    assert left is not None
    assert right is not None
    operation = {
        "+": left + right,
        "*": left * right,
        "-": left - right,
        "/": left / right,
    }
    return operation.get(op, "None")


def compile_model(
    uncompiled_model: type[lf.Model]
    | Sequence[type[lf.Model]]
    | list[type[lf.Model] | float | Literal["+", "-", "*", "/", "(", ")"]],
    params: dict[str, ParametersArgs] | Sequence[dict[str, ParametersArgs]] | None = None,
    prefixes: Sequence[str] = "",
) -> XModelMixin:  # guess_fit is used in MPWorker
    """Generates an lmfit model instance from specification.

    Takes a model sequence, i.e. a Model class, a list of such classes, or a list
    of such classes with operators and instantiates an appropriate model.
    """
    params = params or {}
    assert isinstance(params, dict | Sequence)

    def _is_sequence_of_models(models: Sequence) -> TypeGuard[Sequence[type[lf.Model]]]:
        return all(issubclass(token, lf.Model) for token in models)

    prefix_compile = "{}"
    if not prefixes:
        prefixes = ascii_lowercase
        prefix_compile = "{}_"

    if isinstance(uncompiled_model, type) and issubclass(uncompiled_model, lf.Model):
        if prefixes == ascii_lowercase:
            return uncompiled_model()
        return uncompiled_model(prefix=prefixes[0])

    if isinstance(uncompiled_model, Sequence) and _is_sequence_of_models(uncompiled_model):
        return _compositemodel_from_model_sequence(
            uncompiled_model=uncompiled_model,
            params=params,
            prefixes=prefixes,
            prefix_compile=prefix_compile,
        )
    warnings.warn("Beware of equal operator precedence.", stacklevel=2)
    prefix = iter(prefixes)
    model = [m if isinstance(m, str) else (m, next(prefix)) for m in uncompiled_model]
    return reduce_model_with_operators(_parens_to_nested(model))


def _compositemodel_from_model_sequence(
    uncompiled_model: Sequence[type[lf.Model]],
    params: dict | Sequence,
    prefixes: Sequence[str],
    prefix_compile: str,
) -> XModelMixin:
    models: list[lf.Model] = [
        m(prefix=prefix_compile.format(prefixes[i]), nan_policy="omit")
        for i, m in enumerate(uncompiled_model)
    ]
    if isinstance(params, Sequence):
        for cs, m in zip(params, models, strict=True):
            for k, params_for_name in cs.items():
                m.set_param_hint(name=k, **params_for_name)

    built = functools.reduce(operator.add, models)
    if isinstance(params, dict):
        for k, v in params.items():
            built.set_param_hint(k, **v)
    return built
