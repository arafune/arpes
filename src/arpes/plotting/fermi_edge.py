"""Simple plotting routines related to Fermi edges and Fermi edge fits."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Unpack

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from arpes.analysis.general import fit_fermi_edge
from arpes.constants import TWO_DIMENSION
from arpes.provenance import save_plot_provenance
from arpes.utilities import apply_dataarray

from .utils import label_for_dim, path_for_plot

if TYPE_CHECKING:
    from pathlib import Path

    from arpes._typing import MPLPlotKwargs

__all__ = ["fermi_edge_reference"]


@save_plot_provenance
def fermi_edge_reference(
    data_arr: xr.DataArray,
    energy_range: slice | None = None,
    title: str = "",
    ax: Axes | None = None,
    out: str | Path = "",
    **kwargs: Unpack[MPLPlotKwargs],
) -> Path | Axes:
    """Fits for and plots results for the Fermi edge on a piece of data.

    Args:
        data_arr: The data to be fit., this should be of type DataArray.
        energy_range (slice: None) : Energy range for fitting. if None, (-0.1, 0.1) is used.
            Default to None.
        title (str): A title to attach to the plot
        ax:  The axes to plot to, if not specified will be generated
        out:  Where to save the plot
        kwargs: pass to data.plot()

    Returns: Path | Axes
        Plot of the result about the Fermi edge fitting
    """
    warnings.warn(
        "Not automatically correcting for slit shape distortions to the Fermi edge",
        stacklevel=2,
    )
    assert isinstance(data_arr, xr.DataArray)
    if len(data_arr.dims) > TWO_DIMENSION:
        sum_dimensions: set[str] = {"cycle", "phi", "kp", "kx"}
        sum_dimensions.intersection_update(set(data_arr.dims))
        data_for_fit = data_arr.sum(
            *list(sum_dimensions),
            keep_attrs=True,
        )
        assert any(str(d) == "eV" for d in data_for_fit.dims)
    else:
        data_for_fit = data_arr
    edge_fit = fit_fermi_edge(data_for_fit, energy_range=energy_range)
    centers = apply_dataarray(
        edge_fit.modelfit_results,
        np.vectorize(lambda x: x.params["center"].value, otypes=[float]),
    )
    widths = apply_dataarray(
        edge_fit.modelfit_results,
        np.vectorize(lambda x: x.params["width"].value, otypes=[float]),
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    assert isinstance(ax, Axes)

    if not title:
        title = data_arr.S.label.replace("_", " ")

    centers.S.plot(ax=ax, **kwargs)
    widths.S.plot(ax=ax, **kwargs)

    if isinstance(ax, Axes):
        ax.set_xlabel(label_for_dim(data_arr, ax.get_xlabel()))
        ax.set_ylabel(label_for_dim(data_arr, ax.get_ylabel()))
        ax.set_title(title, font_size=14)

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)
    return ax
