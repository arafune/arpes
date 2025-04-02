"""Tools related to finding the Fermi momentum in a cut."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from lmfit.models import LinearModel

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from arpes._typing import XrTypes

__all__ = ("kfermi_from_mdcs",)


def kfermi_from_mdcs(mdc_results: XrTypes, param: str = "") -> NDArray[np.float64]:
    """Calculates a Fermi momentum using a series of MDCs and the known Fermi level (eV=0).

    This is especially useful to isolate an area for analysis.

    This method tolerates data that came from a prefixed model fit,
    but will always attempt to look for an attribute containing "center".

    Args:
        mdc_results: A DataArray or Dataset containing :obj:``lmfit.ModelResult``s.
        param: The name of the parameter to use as the Fermi momentum in the fit

    Returns:
        The k_fermi values for the input data.
    """
    real_param_name: set[str] | str = set()
    mdc_results = mdc_results if isinstance(mdc_results, xr.DataArray) else mdc_results.results
    assert isinstance(mdc_results, xr.DataArray)
    param_names = mdc_results.F.parameter_names

    if param in param_names and param is not None:
        real_param_name = param
    else:
        best_names = [p for p in param_names if "center" in p]
        if not param:
            best_names = [p for p in best_names if param in p]

        assert len(best_names) == 1
        real_param_name = best_names[0]

    def nan_sieve(_: xr.DataArray, x: xr.DataArray) -> bool:
        return not np.isnan(x.item())

    return (
        LinearModel()
        .fit(
            data=mdc_results.F.p(real_param_name).G.filter_coord(
                coordinate_name="eV",
                sieve=nan_sieve,
            ),
            x=mdc_results.coords["kp"],
        )
        .eval(x=0)
    )
