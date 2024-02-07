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

import os
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, Literal

import dill
import numpy as np
import xarray as xr
from tqdm.notebook import tqdm

import arpes.fits.fit_models
from arpes.provenance import update_provenance
from arpes.utilities import normalize_to_spectrum

from . import mp_fits

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import lmfit


__all__ = ("broadcast_model", "result_to_hints")


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


def result_to_hints(
    model_result: lmfit.model.ModelResult | None,
    defaults: dict[str, dict[Literal["value"], float]] | None = None,
) -> dict[str, dict[Literal["value"], float]] | None:
    """Turns an `lmfit.model.ModelResult` into a dictionary with initial guesses.

    Args:
        model_result: The model result to extract parameters from
        defaults: Returned if `model_result` is None, useful for cell re-evaluation in Jupyter

    Returns:
        A dict containing parameter specifications in key-value rathan than `lmfit.Parameter`
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

    def read_token(token: str) -> str | float:
        if token in special:
            return token
        try:
            return float(token)
        except ValueError as v_err:
            try:
                return arpes.fits.fit_models.__dict__[token]
            except KeyError:
                msg = f"Could not find model: {token}"
                raise ValueError(msg) from v_err

    return [read_token(token) for token in model.split()]


@update_provenance("Broadcast a curve fit along several dimensions")
def broadcast_model(
    model_cls: type[lmfit.Model] | Sequence[type[lmfit.Model]] | str,
    data: xr.DataArray | xr.Dataset,
    broadcast_dims: str | list[str],
    params: dict | None = None,
    weights: xr.DataArray | None = None,
    prefixes: Sequence[str] = "",
    window: xr.DataArray | None = None,
    parallelize: bool | None = None,
    *,
    progress: bool = True,
    safe: bool = False,
) -> xr.Dataset:
    """Perform a fit across a number of dimensions.

    Allows composite models as well as models defined and compiled through strings.

    Args:
        model_cls: The model specification
        data: The data to curve fit
        broadcast_dims: Which dimensions of the input should be iterated across as opposed
          to fit across
        params: Parameter hints, consisting of plain values or arrays for interpolation
        weights: Weights to apply when curve fitting. Should have the same shape as the input data
        prefixes: Prefix for the parameter name.  Pass to MPWorker that pass to
          broadcast_common.compile_model
        window: A specification of cuts/windows to apply to each curve fit
        parallelize: Whether to parallelize curve fits, defaults to True if unspecified and more
          than 20 fits were requested
        progress: Whether to show a progress bar
        safe: Whether to mask out nan values

    Returns:
        An `xr.Dataset` containing the curve fitting results. These are data vars:

        - "results": Containing an `xr.DataArray` of the `lmfit.model.ModelResult` instances
        - "residual": The residual array, with the same shape as the input
        - "data": The original data used for fitting
        - "norm_residual": The residual array normalized by the data, i.e. the fractional error
    """
    if params is None:
        params = {}

    if isinstance(broadcast_dims, str):
        broadcast_dims = [broadcast_dims]

    logger.debug("Normalizing to spectrum")
    data_array = normalize_to_spectrum(data)
    del data
    assert isinstance(data_array, xr.DataArray)
    cs = {}
    for dim in broadcast_dims:
        cs[dim] = data_array.coords[dim]

    other_axes = set(data_array.dims).difference(set(broadcast_dims))
    template = data_array.sum(list(other_axes))
    template.values = np.ndarray(template.shape, dtype=object)
    n_fits = np.prod(np.array(list(template.S.dshape.values())))
    if parallelize is None:
        parallelize = bool(n_fits > 20)  # noqa: PLR2004

    residual = data_array.copy(deep=True)
    logger.debug("Copying residual")
    residual.values = np.zeros(residual.shape)

    logger.debug("Parsing model")
    model = parse_model(model_cls)
    # <== when model_cls type is tpe or iterable[model]
    # parse_model just reterns model_cls as is.

    if progress:
        wrap_progress = tqdm
    else:

        def wrap_progress(x: Iterable[int], **kwargs: str | float) -> Iterable[int]:
            """Fake of tqdm.notebook.tqdm.

            Args:
                x (Iterable[int]): [TODO:description]
                kwargs: its a dummy parameter, which is not used.

            Returns:
                Same iterable.
            """
            del kwargs  # kwargs is dummy parameter
            return x

    serialize = parallelize
    assert isinstance(serialize, bool)
    fitter = mp_fits.MPWorker(
        data=data_array,
        uncompiled_model=model,
        prefixes=prefixes,
        params=params,
        safe=safe,
        serialize=serialize,
        weights=weights,
        window=window,
    )

    if parallelize:
        logger.debug(f"Running fits (nfits={n_fits}) in parallel (n_threads={os.cpu_count()})")

        from .hot_pool import hot_pool

        pool = hot_pool.pool
        exe_results = list(
            wrap_progress(
                pool.imap(fitter, template.G.iter_coords()),  # IMapIterator
                total=n_fits,
                desc="Fitting on pool...",
            ),
        )
    else:
        logger.debug(f"Running fits (nfits={n_fits}) serially")
        exe_results = []
        for _, cut_coords in wrap_progress(
            template.G.enumerate_iter_coords(),
            desc="Fitting",
            total=n_fits,
        ):
            exe_results.append(fitter(cut_coords))

    if serialize:
        logger.debug("Deserializing...")

        def unwrap(result_data: str) -> object:  # (Unpickler)
            # using the lmfit deserialization and serialization seems slower than double pickling
            # with dill
            return dill.loads(result_data)

        exe_results = [(unwrap(res), residual, cs) for res, residual, cs in exe_results]

    logger.debug("Finished running fits Collating")
    for fit_result, fit_residual, coords in exe_results:
        template.loc[coords] = np.array(fit_result)
        residual.loc[coords] = fit_residual

    logger.debug("Bundling into dataset")
    return xr.Dataset(
        {
            "results": template,
            "data": data_array,
            "residual": residual,
            "norm_residual": residual / data_array,
        },
        residual.coords,
    )
