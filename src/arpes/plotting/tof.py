"""Plotting routines for count-like data from a ToF or delay line.

This module is a bit of a misnomer, in that it also applies perfectly well to data collected by a
delay line on a hemisphere, the important point is that the data in any given channel should
correspond to the true number of electrons that arrived in that channel.

Plotting routines here are ones that include statistical errorbars. Generally for datasets in
PyARPES, an xr.Dataset will hold the standard deviation data for a given variable on
`{var_name}_std`.
"""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Unpack

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from arpes.debug import setup_logger
from arpes.provenance import save_plot_provenance

from .utils import path_for_plot

if TYPE_CHECKING:
    from pathlib import Path

    import xarray as xr
    from matplotlib.figure import Figure

    from arpes._typing import MPLPlotKwargs, MPLPlotKwargsBasic

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


__all__ = (
    "plot_with_std",
    "scatter_with_std",
)


@save_plot_provenance
def plot_with_std(
    data_set: xr.Dataset,  # data_vars is used,
    name_to_plot: str = "",
    ax: Axes | None = None,
    out: str | Path = "",
    figsize: tuple[float, float] = (7, 5),
    **kwargs: Unpack[MPLPlotKwargs],
) -> Path | tuple[Figure | None, Axes]:
    """Makes a fill-between line plot with error bars from associated statistical errors.

    Args:
       data_set (xr.Dataset): ARPES data that 'mean_and_deviation' is applied.
       name_to_plot(str): data name to plot, in most case "spectrum" is used.
       ax: Matplotlib Axes object
       out: (str | Path): Path name to output figure.
       figsize (tuple[float, float]): figure size
       **kwargs: pass to subplots if figsize is set as tuple, other kwargs are pass to
           ax.fill_between/xr.DataArray.plot
    """
    if not name_to_plot:
        var_names = [k for k in data_set.data_vars if "_std" not in str(k)]
        assert len(var_names) == 1
        name_to_plot = str(var_names[0])
        assert (
            name_to_plot + "_std"
        ) in data_set.data_vars, "Has 'mean_and_deviation' been applied?"

    fig: Figure | None = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    assert isinstance(ax, Axes)

    data_set.data_vars[name_to_plot].S.plot(ax=ax, **kwargs)
    x, y = data_set.data_vars[name_to_plot].G.to_arrays()

    std = data_set.data_vars[name_to_plot + "_std"].values
    kwargs.setdefault("alpha", 0.3)
    ax.fill_between(x, y - std, y + std, **kwargs)

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    ax.set_xlim(left=np.min(x), right=np.max(x))

    return fig, ax


@save_plot_provenance
def scatter_with_std(
    data: xr.Dataset,  # data_vars is used.
    name_to_plot: str = "",
    ax: Axes | None = None,
    out: str | Path = "",
    figsize: tuple[float, float] = (7, 5),
    **kwargs: Unpack[MPLPlotKwargsBasic],
) -> Path | tuple[Figure | None, Axes]:
    """Makes a scatter plot of data with error bars generated from associated statistical errors.

    Args:
        data(xr.Dataset): ARPES data that 'mean_and_deviation' is applied.
        name_to_plot(str): data name to plot, in most case "spectrum" is used.
        ax: Matplotlib Axes object
        out: (str | Path): Path name to output figure.
        figsize (tuple[float, float]): tuple for figure size.
        **kwargs: pass to subplots if figsize is set as tuple, other kwargs are pass to ax.errorbar
    """
    if not name_to_plot:
        var_names = [k for k in data.data_vars if "_std" not in str(k)]
        assert len(var_names) == 1
        name_to_plot = str(var_names[0])
        assert (
            name_to_plot + "_std"
        ) in data.data_vars, "Has 'mean_and_deviation' been applied to the data?"

    fig: Figure | None = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    assert isinstance(ax, Axes)
    x, y = data.data_vars[name_to_plot].G.to_arrays()

    std = data.data_vars[name_to_plot + "_std"].values
    ax.errorbar(x, y, yerr=std, markeredgecolor="black", **kwargs)

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    ax.set_xlim(left=np.min(x), right=np.max(x))

    return fig, ax
