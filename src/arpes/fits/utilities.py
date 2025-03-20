"""Provides broadcasted and process parallel curve fitting for PyARPES.

The core of this module is `broadcast_model` which is a serious workhorse in PyARPES for
analyses based on curve fitting. This allows simple multidimensional curve fitting by
iterative fitting across one or many axes. Currently basic strategies are implemented,
but in the future we would like to provide:

1. Passing xr.DataArray values to parameter guesses and bounds, which can be interpolated/selected
   to allow changing conditions throughout the curve fitting session.
2. A strategy allowing retries with initial guess taken from the previous fit. This is similar
   to some adaptive curve fitting routines that have been proposed in the literature.
"""

from __future__ import annotations

import warnings
from datetime import UTC, datetime
from logging import DEBUG, INFO
from os import cpu_count
from typing import TYPE_CHECKING, Literal, TypeGuard, TypeVar

import dill
import lmfit
import numpy as np
import xarray as xr

import arpes.fits.fit_models
from arpes import VERSION
from arpes.debug import setup_logger
from arpes.utilities import normalize_to_spectrum
from arpes.utilities.jupyter import get_tqdm

from . import mp_fits
from .hot_pool import hot_pool

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arpes.fits import ParametersArgs
    from arpes.provenance import Provenance


__all__ = ("broadcast_model", "result_to_hints")


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)

T = TypeVar("T")

tqdm = get_tqdm()


def result_to_hints(
    model_result: lmfit.model.ModelResult | None,
    defaults: dict[str, dict[Literal["value"], float]] | None = None,
) -> dict[str, dict[Literal["value"], float]] | None:
    """Turns an `lmfit.model.ModelResult` into a dictionary with initial guesses.

    Args:
        model_result: The model result to extract parameters from
        defaults: Returned if `model_result` is None, useful for cell re-evaluation in Jupyter

    Returns:
        A dict containing parameter specifications in key-value rather than `lmfit.Parameter`
        format, as you might pass as `params=` to PyARPES fitting code.
    """
    if model_result is None:
        return defaults
    return {k: {"value": model_result.params[k].value} for k in model_result.params}


def parse_model(
    model: str | type[lmfit.Model] | Sequence[type[lmfit.Model]],
) -> (
    type[lmfit.Model]
    | Sequence[type[lmfit.Model]]
    | list[type[lmfit.Model] | float | Literal["+", "-", "*", "/", "(", ")"]]
):
    """Takes a model string and turns it into a tokenized version.

    1. ModelClass -> ModelClass
    2. [ModelClass] -> [ModelClass]
    3. str -> [<ModelClass, operator as string>]

    i.e.

    A + (B + C) * D -> [A, '(', B, '+', C, ')', '*', D]

    Args:
        model: The model specification

    Returns:
        A tokenized specification of the model suitable for passing to the curve
        fitting routine.
    """
    if not isinstance(model, str):
        return model

    pad_all = ["+", "-", "*", "/", "(", ")"]

    for pad in pad_all:
        model = model.replace(pad, f" {pad} ")

    special = set(pad_all)

    def _token_check(token: str) -> TypeGuard[Literal["+", "-", "*", "/", "(", ")"]]:
        return token in special

    def read_token(token: str) -> Literal["+", "-", "*", "/", "(", ")"] | float | type[lmfit.Model]:
        if _token_check(token):
            return token
        try:
            return float(token)
        except ValueError as v_err:
            try:
                model_type = arpes.fits.fit_models.__dict__[token]
                assert issubclass(model_type, lmfit.Model)
            except KeyError:
                msg = f"Could not find model: {token}"
                raise ValueError(msg) from v_err
            else:
                return model_type

    return [read_token(token) for token in model.split()]


def broadcast_model(  # noqa: PLR0913
    model_cls: type[lmfit.Model] | Sequence[type[lmfit.Model]] | str,
    data: xr.DataArray,
    broadcast_dims: str | list[str],
    params: dict[str, ParametersArgs] | Sequence[dict[str, ParametersArgs]] | None = None,
    weights: xr.DataArray | None = None,
    prefixes: Sequence[str] = "",
    window: xr.DataArray | None = None,
    *,
    parallelize: bool | None = None,
    safe: bool = False,
) -> xr.Dataset:
    r"""Perform a fit across a number of dimensions.

    Allows composite models as well as models defined and compiled through strings.
    There are three ways in order to specify the model for fitting.

    1. Just specify the (single) model only.
    2. (Recommended) More than two models by the equence of model like:
        [AffineBroadenedFD, LorentzianModel]. By this style, the sum of these models is used as
        the composite model.
    3. As string: like "AffineBroadenedFD + LorentzianModel". Used when the composite model
        cannot be described as the sum of models.

    Args:
        model_cls: The model specification
        data: The data to curve fit (Should be DataArray)
        broadcast_dims: Which dimensions of the input should be iterated across as opposed
          to fit across
        params: Parameter hints, consisting of the dict-style values or arrays of parameter hints.
            **Keep consistensity with prefixes**.  Two styles can be used:

                * {"a_center": value=0.0, "b_width": {"value": 0.0, "vary": False}}
                * [{"a_center": value=0.0}, {"b_width": {"value": 0.0, "vary": False}}]

        weights: Weights to apply when curve fitting. Should have the same shape as the input data
        prefixes: Prefix for the parameter name.  Pass to MPWorker that pass to
          broadcast_common.compile_model.  When prefixes are specified, the number of prefixes must
          be same as the number of models for fitting. If not specified, the prefix automatically is
          determined as "a\_", "b\_",....  (We recommend to specify them explicitly.)
        window: A specification of cuts/windows to apply to each curve fit
        parallelize: Whether to parallelize curve fits, defaults to True if unspecified and more
          than 20 fits were requested.
        progress: Whether to show a progress bar
        safe: Whether to mask out nan values

    Returns: xr.Dataset
        An `xr.Dataset` containing the curve fitting results. These are data vars:

            - "results": Containing an `xr.DataArray` of the `lmfit.model.ModelResult` instances
            - "residual": The residual array, with the same shape as the input
            - "data": The original data used for fitting
            - "norm_residual": The residual array normalized by the data, i.e. the fractional error

    Note:
        Though there are many arguments, the essentials are model_cls, params, prefixes
        (and the data for fit, needless to say.)

    """
    warnings.warn(
        "This function has not been maintained.  Use S.modelfit() insted. "
        "(Migration is not so difficult)",
        DeprecationWarning,
        stacklevel=2,
    )

    params = params or {}

    if isinstance(broadcast_dims, str):
        broadcast_dims = [broadcast_dims]

    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    cs = {}
    for dim in broadcast_dims:
        cs[dim] = data.coords[dim]

    other_axes = set(data.dims).difference(set(broadcast_dims))
    results = data.sum(list(other_axes))
    results.values = np.ndarray(results.shape, dtype=object)

    n_fits = np.prod(np.array(list(results.sizes.values())))

    if parallelize is None:
        parallelize = bool(n_fits > 20)  # noqa: PLR2004
    residual = xr.DataArray(np.zeros_like(data.values), coords=data.coords, dims=data.dims)

    model: (
        type[lmfit.Model]
        | Sequence[type[lmfit.Model]]
        | list[type[lmfit.Model] | float | Literal["+", "-", "*", "/", "(", ")"]]
    ) = parse_model(model_cls)

    serialize = parallelize
    assert isinstance(serialize, bool)
    fitter = mp_fits.MPWorker(
        data=data,
        uncompiled_model=model,
        prefixes=prefixes,
        params=params,
        safe=safe,
        serialize=serialize,
        weights=weights,
        window=window,
    )

    if parallelize:
        logger.debug(f"Running fits (nfits={n_fits}) in parallel (n_threads={cpu_count()})")

        pool = hot_pool.pool
        exe_results = list(
            tqdm(
                pool.imap(fitter, results.G.iter_coords()),  # IMapIterator
                total=int(n_fits),
                desc="Fitting on pool...",
            ),
        )
    else:
        logger.debug(f"Running fits (nfits={n_fits}) serially")
        exe_results = []
        for cut_coords in tqdm(
            results.G.iter_coords(),
            desc="Fitting",
            total=int(n_fits),
        ):
            exe_results.append(fitter(cut_coords))

    if serialize:
        logger.debug("Deserializing...")

        def unwrap(result_data: str) -> object:  # (Unpickler)
            # using the lmfit deserialization and serialization seems slower than double pickling
            # with dill
            return dill.loads(result_data)  # noqa: S301

        exe_results = [(unwrap(res), residual, cs) for res, residual, cs in exe_results]

    for fit_result, fit_residual, coords in exe_results:
        results.loc[coords] = np.array(fit_result)
        residual.loc[coords] = fit_residual

    logger.debug(msg=f"fitter.model: {fitter.model}")
    provenance_context: Provenance = {
        "what": "Broadcast a curve fit along several dimensions",
        "by": "broadcast_common",
        "with": f"{fitter.model}",
    }
    fit_result_dataset = xr.Dataset(
        data_vars={
            "results": results,
            "data": data,
            "residual": residual,
            "norm_residual": residual / data,
        },
        coords=residual.coords,
    )

    fit_result_dataset.attrs["provenance"] = {
        "record": provenance_context,
        "parent_id": data.attrs.get("id", None),
        "time": datetime.now(tz=UTC).isoformat(),
        "version": VERSION,
    }

    return fit_result_dataset
