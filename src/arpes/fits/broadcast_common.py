"""Utilities used in broadcast fitting."""

from __future__ import annotations

import functools
import operator
import warnings
from collections.abc import Sequence
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from string import ascii_lowercase
from typing import TYPE_CHECKING, Any, Literal, TypeGuard

import lmfit as lf
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Iterable

    from _typeshed import Incomplete

    from arpes.fits import ParametersArgs

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


def unwrap_params(
    params: dict[str, ParametersArgs],
    iter_coordinate: dict[str, slice | float],
) -> dict[str, Any]:
    """Inspects arraylike parameters and extracts appropriate value for current fit."""

    def transform_or_walk(
        v: dict | xr.DataArray | Iterable[float],
    ) -> Incomplete:
        """[TODO:summary].

        [TODO:description]

        Args:
            v: [TODO:description]
        """
        if isinstance(v, dict):
            return unwrap_params(v, iter_coordinate)

        if isinstance(v, xr.DataArray):
            return v.sel(iter_coordinate, method="nearest").item()

        return v

    return {k: transform_or_walk(v) for k, v in params.items()}


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
) -> lf.Model:
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
) -> lf.Model:
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
