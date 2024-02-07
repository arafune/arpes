"""Utilities for plotting parameter data out of bulk fits."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt

from arpes.provenance import save_plot_provenance

from .utils import latex_escape

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    from matplotlib.axes import Axes
    from matplotlib.typing import RGBColorType
    from numpy.typing import NDArray

__all__ = ("plot_parameter",)


@save_plot_provenance
def plot_parameter(
    fit_data: xr.DataArray,
    param_name: str,
    ax: Axes | None = None,
    fillstyle: Literal["full", "left", "right", "bottom", "top", "none"] = "none",
    shift: float = 0,
    x_shift: float = 0,
    markersize: int = 8,
    *,
    two_sigma: bool = False,
    **kwargs: tuple | RGBColorType,
) -> Axes:
    """Makes a simple scatter plot of a parameter from an `broadcast_fit` result."""
    if ax is None:
        _, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 5)))

    ds = fit_data.F.param_as_dataset(param_name)
    x_name = ds.value.dims[0]
    x: NDArray[np.float_] = ds.coords[x_name].values

    color = kwargs.get("color")
    e_width = None
    l_width = None
    if two_sigma:
        _, __, lines = ax.errorbar(
            x + x_shift,
            ds.value.values + shift,
            yerr=2 * ds.error.values,
            fmt="",
            elinewidth=1,
            linewidth=0,
            c=color,
            **kwargs,
        )
        color = lines[0].get_color()[0]
        e_width = 2
        l_width = 0

    ax.errorbar(
        x + x_shift,
        ds.value.values + shift,
        yerr=ds.error.values,
        fmt="s",
        color=color,
        elinewidth=e_width,
        linewidth=l_width,
        markeredgewidth=e_width or 2,
        fillstyle=fillstyle,
        markersize=markersize,
        **kwargs,
    )

    ax.set_xlabel(latex_escape(x_name))
    ax.set_ylabel(latex_escape(param_name))
    return ax
