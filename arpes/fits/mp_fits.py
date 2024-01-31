"""Worker for process parallel curve fitting with `lmfit`.

Uses dill for IPC due to issues with pickling `lmfit` instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, Any

import dill

from .broadcast_common import apply_window, compile_model, unwrap_params

if TYPE_CHECKING:
    from collections.abc import Sequence

    import lmfit as lf
    import numpy as np
    import xarray as xr
    from _typeshed import Incomplete
    from numpy.typing import NDArray

__all__ = ["MPWorker"]

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


@dataclass
class MPWorker:
    """Worker for performing curve fitting on a high dimensional dataset.

    Essentially represents a closure over the curve fitting parameters (the input data,
    model specification, etc.) which do not change for a batch of curve fits.

    This worker can then be treated as a function of coordinates only which:
    1. Subselects its data at those coordinates
    2. Performs the fit at those coordinates using the retained settings
    3. Returns the fit result

    There are also a few details related to serialization because we are dealing with
    IPC whenever we use multiprocessing.

    Attributes:
        data(xr.DataArray)
    """

    data: xr.DataArray
    uncompiled_model: Any

    prefixes: Sequence[str]
    params: Any

    safe: bool = False
    serialize: bool = False
    weights: xr.DataArray | None = None
    window: xr.DataArray | None = None

    _model: Any = field(init=False)

    def __post_init__(self) -> None:
        """Indicate that the model has not been compiled yet."""
        self._model = None

    @property
    def model(self) -> lf.Model:
        """Compiles and caches the model used for curve fitting.

        Because of pickling constraints, we send model specifications
        not model instances out to workers and let them compile the model.

        This also ends up being slightly more efficient because the specification
        for a model is just references to classes in code.
        """
        if self._model is not None:
            return self._model

        self._model = compile_model(
            self.uncompiled_model,
            params=self.params,
            prefixes=self.prefixes,
        )
        self._model.make_params()

        return self._model

    @property
    def fit_params(self) -> dict[str, float | NDArray[np.float_]] | None:
        """Builds or fetches the parameter hints from closed over attributes."""
        if isinstance(self.params, list | tuple):
            return {}

        return self.params

    def __call__(
        self,
        cut_coords: dict[str, slice],
    ) -> tuple[lf.model.ModelResult, Incomplete, dict[str, slice]]:
        """Performs a curve fit at the coordinates specified by `cut_coords`."""
        current_params = unwrap_params(self.fit_params, cut_coords)
        cut_data, original_cut_data = apply_window(self.data, cut_coords, self.window)

        if self.safe:
            cut_data = cut_data.G.drop_nan()

        weights_for = None
        if self.weights is not None:
            weights_for = self.weights.sel(cut_coords)

        try:
            fit_result = self.model.guess_fit(cut_data, params=current_params, weights=weights_for)
        except ValueError:
            fit_result = None

        if fit_result is None:
            true_residual = None
        elif self.window is None:
            true_residual = fit_result.residual
        else:
            true_residual = original_cut_data - fit_result.eval(
                x=original_cut_data.coords[original_cut_data.dims[0]].values,
            )

        if self.serialize:
            fit_result = dill.dumps(fit_result)

        return fit_result, true_residual, cut_coords
