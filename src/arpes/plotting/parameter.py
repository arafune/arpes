"""Utilities for plotting parameter data out of bulk fits."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from arpes.provenance import save_plot_provenance

from .utils import latex_escape

if TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    from numpy.typing import NDArray

    from arpes._typing.plotting import MPLPlotKwargs

__all__ = ("plot_parameter",)


@save_plot_provenance
def plot_parameter(  # noqa: PLR0913
    fit_data: xr.DataArray,
    param_name: str,
    ax: Axes | None = None,
    shift: float = 0,
    x_shift: float = 0,
    *,
    two_sigma: bool = False,
    figsize: tuple[float, float] = (7, 5),
    **kwargs: Unpack[MPLPlotKwargs],
) -> Axes:
    """Creates a scatter plot of a parameter from a `broadcast_fit` result.

    Args:
        fit_data (xr.DataArray): The fitting result, typically from `broadcast_fit.results`.
        param_name (str): The name of the parameter to plot.
        ax (Axes, optional): The axes on which to plot. If not provided, a new set of axes will be
            created.
        shift (float, optional): A vertical shift for the plot. Default is 0.
        x_shift (float, optional): A horizontal shift for the x-values. Default is 0.
        two_sigma (bool, optional): If True, plots the error bars as two standard deviations.
            Default is False.
        figsize (tuple[float, float], optional): The size of the figure. Default is (7, 5).
        kwargs: Additional keyword arguments for the plot (e.g., `color`, `markersize`, etc.).

    Returns:
        Axes: The Axes object with the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    assert isinstance(ax, Axes)

    ds = fit_data.F.param_as_dataset(param_name)
    x_name = ds.value.dims[0]
    x: NDArray[np.float64] = ds.coords[x_name].values
    kwargs.setdefault("fillstyle", "none")
    kwargs.setdefault("markersize", 8)
    kwargs.setdefault("color", "#1f77b4")  # matplotlib.colors.TABLEAU_COLORS["tab:blue"]

    e_width = None
    if "fmt" not in kwargs:
        kwargs["fmt"] = ""
    if two_sigma:
        _, _, lines = ax.errorbar(
            x + x_shift,
            ds.value.values + shift,
            yerr=2 * ds.error.values,
            elinewidth=1,
            **kwargs,
        )
        e_width = 2
        kwargs["markeredgewidth"] = 2
        kwargs["color"] = lines[0].get_color()[0]
        kwargs["linewidth"] = 0

    kwargs["fmt"] = "s"
    kwargs.setdefault("markeredgewidth", 2)
    ax.errorbar(
        x + x_shift,
        ds.value.values + shift,
        yerr=ds.error.values,
        elinewidth=e_width,
        **kwargs,
    )

    ax.set_xlabel(latex_escape(x_name))
    ax.set_ylabel(latex_escape(param_name))
    return ax
