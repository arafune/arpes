"""Simple plotting routines related to Fermi edges and Fermi edge fits."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Unpack

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from arpes.fits import GStepBModel, broadcast_model
from arpes.provenance import save_plot_provenance
from arpes.utilities import apply_dataarray

from .utils import label_for_dim, path_for_plot

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from arpes._typing import MPLPlotKwargs

__all__ = ["fermi_edge_reference", "plot_fit"]


@save_plot_provenance
def plot_fit(
    data: xr.DataArray,
    title: str = "",
    axes: None | NDArray[np.object_] = None,
    out: str | Path = "",
) -> Path | None:
    """Plots the results of a fit of some lmfit model to some data.

    We introspect the model to determine which attributes we should plot,
    as well as their uncertainties

    Args:
        data: The data, this should be of type DataArray <lmfit.model.ModelResult>
        title: A title to attach to the plot
        axes: The axes to plot to, if not specified will be generated
        out: Where to save the plot
    """
    # get any of the elements
    reference_fit = data.values.ravel()[0]
    model = reference_fit.model
    SKIP_NAMES = {"const_bkg", "lin_bkg", "erf_amp"}
    param_names = [p for p in model.param_names if p not in SKIP_NAMES]
    n_params = len(param_names)
    MAX_COLS = 3
    n_rows = int(np.ceil(n_params / MAX_COLS))
    n_cols = n_params if n_params < MAX_COLS else MAX_COLS

    if axes is None:
        _, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6))
    assert isinstance(axes, np.ndarray)
    for i, param in enumerate(param_names):
        row = i // MAX_COLS
        column = i - (row * MAX_COLS)

        try:
            ax = axes[row, column]
        except IndexError:
            ax = axes[column]  # n_rows = 1

        # extract the data for this param name
        # attributes are on .value and .stderr
        centers = data.G.map(lambda x: x.params[param].value)
        if "bootstrap" in data.dims:
            centers = centers.mean("bootstrap")

        centers.plot(ax=ax)

        ax.set_title(f"Fit var: {param}")

        if len(centers.dims) == 1:
            if "bootstrap" in data.dims:
                widths = data.G.map(lambda x: x.params[param].value).std("bootstrap")
            else:
                widths = data.G.map(lambda x: x.params[param].stderr)
            # then we can plot widths as well, otherwise we need more
            # figures, blergh
            x_coords = centers.coords[centers.dims[0]]
            ax.fill_between(
                x_coords,
                centers.values + widths.values,
                centers.values - widths.values,
                alpha=0.5,
            )

    if not title:
        title = data.S.label.replace("_", " ")

    # if multidimensional, we can share y axis as well

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    plt.show()
    return None


@save_plot_provenance
def fermi_edge_reference(
    data_arr: xr.DataArray,
    title: str = "",
    ax: Axes | None = None,
    out: str | Path = "",
    **kwargs: Unpack[MPLPlotKwargs],
) -> Path | Axes:
    """Fits for and plots results for the Fermi edge on a piece of data.

    Args:
        data_arr: The data, this should be of type DataArray<lmfit.model.ModelResult>
        title: A title to attach to the plot
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
    sum_dimensions: set[str] = {"cycle", "phi", "kp", "kx"}
    sum_dimensions.intersection_update(set(data_arr.dims))
    summed_data = data_arr.sum(*list(sum_dimensions))

    broadcast_dimensions = [str(d) for d in summed_data.dims if str(d) != "eV"]
    msg = f"Could not product fermi edge reference. Too many dimensions: {broadcast_dimensions}"
    assert len(broadcast_dimensions) == 1, msg
    edge_fit = broadcast_model(
        GStepBModel,
        summed_data.sel(eV=slice(-0.1, 0.1)),
        broadcast_dimensions[0],
    )
    centers = apply_dataarray(
        edge_fit.results,
        np.vectorize(lambda x: x.params["center"].value, otypes=[float]),
    )
    widths = apply_dataarray(
        edge_fit.results,
        np.vectorize(lambda x: x.params["width"].value, otypes=[float]),
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if not title:
        title = data_arr.S.label.replace("_", " ")

    centers.plot(ax=ax, **kwargs)
    widths.plot(ax=ax, **kwargs)

    if isinstance(ax, Axes):
        ax.set_xlabel(label_for_dim(data_arr, ax.get_xlabel()))
        ax.set_ylabel(label_for_dim(data_arr, ax.get_ylabel()))
        ax.set_title(title, font_size=14)

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)
    return ax
