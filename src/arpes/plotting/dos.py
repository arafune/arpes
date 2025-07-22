"""Plotting utilities related to density of states plots.

This module provides functions and utilities for creating and saving
density of states (DOS) plots using matplotlib and xarray.
"""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Literal, Unpack

import xarray as xr
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, LogNorm, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from arpes.analysis.xps import approximate_core_levels
from arpes.constants import TWO_DIMENSION
from arpes.debug import setup_logger
from arpes.provenance import save_plot_provenance
from arpes.utilities import normalize_to_spectrum

from .utils import path_for_plot, savefig

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure

    from arpes._typing.plotting import MPLPlotKwargs, QuadmeshParam

__all__ = (
    "plot_core_levels",
    "plot_dos",
)

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]

logger = setup_logger(__name__, LOGLEVEL)


def create_figure_and_axes(
    figsize: tuple[float, float],
    orientation: Literal["horizontal", "vertical"] = "horizontal",
) -> tuple[Figure, gridspec.GridSpec]:
    """Create a figure and axes for plotting.

    Args:
        figsize (tuple[float, float]): The size of the figure.
        orientation (Literal["horizontal", "vertical"]): The orientation of the figure.

    Returns:
        tuple[Figure, gridspec.GridSpec]: The figure and grid specification.
    """
    if orientation.startswith("h"):
        fig = plt.figure(figsize=figsize) if figsize else plt.figure(figsize=(8, 6))
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[5.5, 1.0])
    else:
        fig = plt.figure(figsize=figsize) if figsize else plt.figure(figsize=(6, 8))
        gs = gridspec.GridSpec(nrows=1, ncols=2, width_ratios=[5.5, 1.0])
    fig.subplots_adjust(hspace=0.00, wspace=0.00)
    return fig, gs


def add_colorbar(
    ax: Axes,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    norm: Normalize | None = None,
    cmap: str | Colormap | None = None,
) -> None:
    """Add a colorbar to the plot.

    Args:
        ax (Axes): The axes to add the colorbar to.
        orientation (str): The orientation of the colorbar.
        norm (Normalize | None): The normalization for the colorbar. If None, the default
            (full range) is used.
        cmap (str | Colormap): The colormap for the colorbar. If None, the default (viridis) is
            used.
    """
    if cmap is None:
        cmap = "viridis"
    if orientation.startswith("h"):
        axins = inset_axes(
            ax,
            width="2%",
            height="90%",
            loc="lower left",
            bbox_to_anchor=(1.01, 0.05, 1, 1),
            bbox_transform=ax.transAxes,
        )
        Colorbar(ax=axins, orientation="vertical", norm=norm, cmap=cmap)
    else:
        axins = inset_axes(
            ax,
            height="2%",
            width="90%",
            loc="lower left",
            bbox_to_anchor=(0.01, 1.05, 1, 1),
            bbox_transform=ax.transAxes,
        )
        Colorbar(ax=axins, orientation="horizontal", norm=norm, cmap=cmap)


@save_plot_provenance
def plot_core_levels(  # noqa: PLR0913
    data: xr.DataArray,
    ax: Axes | None = None,
    *,
    out: str | Path = "",
    core_levels: list[float] | None = None,
    binning: int = 3,
    promenance: int = 5,
    figsize: tuple[float, float] = (11, 5),
    **kwargs: Unpack[MPLPlotKwargs],
) -> Path | tuple[Figure | None, Axes]:
    """Plots an XPS curve and approximate core level locations.

    Args:
        data (xr.DataArray): The data array containing the XPS information.
        ax (Axes, optional): The matplotlib axes object to plot on. Defaults to None.
        out (str | Path, optional): The file path to save the plot. Defaults to "".
        core_levels (list[float], optional): List of core level energies. Defaults to None.
        binning (int, optional): Binning parameter for core level approximation. Defaults to 3.
        promenance (int, optional): Prominence parameter for core level approximation.
            Defaults to 5.
        figsize (tuple[float, float], optional): Figure size. Defaults to (11, 5).
        **kwargs: Additional keyword arguments for customization.

    Returns:
        Path | tuple[Figure | None, Axes]: The file path if saved, otherwise the figure and axes.
    """
    fig: Figure | None = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    assert isinstance(ax, Axes)
    data.S.plot(ax=ax, **kwargs)

    if core_levels is None:
        core_levels = approximate_core_levels(data, binning=binning, promenance=promenance)
    assert core_levels is not None
    for core_level in core_levels:
        ax.axvline(core_level, ymin=0.1, ymax=0.25, color="red", ls="-")

    if out:
        savefig(str(out), dpi=400)
        return path_for_plot(out)
    return fig, ax


@save_plot_provenance
def plot_dos(
    data: xr.DataArray,
    out: str | Path = "",
    figsize: tuple[float, float] | None = None,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    **kwargs: Unpack[QuadmeshParam],
) -> Path | tuple[Figure, tuple[Axes, Axes]]:
    """Plots the density of states next to the original spectr spectra.

    cf: https://matplotlib.org/stable/gallery/axes_grid1/demo_colorbar_with_inset_locator.html

    Args:
        data (xr.DataArray): ARPES data to plot.
        out (str | Path): Path to the figure.
        orientation (Literal["horizontal", "vertical"]): Orientation of the figures.
        figsize (tuple[float, float] | None): The figure size (arg of plt.figure()).
        kwargs: Additional keyword arguments for customization.

    Returns:
        Path | tuple[Figure, tuple[Axes, Axes]]: The file path if saved, otherwise the figure and
            axes.
    """
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    assert isinstance(data, xr.DataArray)
    assert data.ndim == TWO_DIMENSION
    dos = data.S.sum_other(["eV"], keep_attrs=True)
    kwargs.setdefault(
        "norm",
        Normalize(vmin=data.min().item(), vmax=data.max().item()),
    )
    kwargs.setdefault("cmap", "viridis")
    figsize = figsize or (7.0, 5.0)

    fig, gs = create_figure_and_axes(figsize, orientation)
    ax0 = fig.add_subplot(gs[0])
    data.S.plot(ax=ax0, add_labels=False, add_colorbar=False, **kwargs)
    add_colorbar(
        ax0,
        orientation,
        kwargs.get("norm", Normalize(vmin=data.min().item(), vmax=data.max().item())),
        kwargs.get("cmap", "viridis"),
    )

    if orientation.startswith("h"):
        ax1: Axes = fig.add_subplot(gs[1], sharex=ax0)
        plt.setp(ax0.get_xticklabels(), visible=False)
        if "norm" in kwargs and isinstance(kwargs["norm"], LogNorm):
            dos.S.plot(ax=ax1, _labels=False, yscale="log")
        else:
            dos.S.plot(ax=ax1, _labels=False)
        ax1.set_xlabel(str(data.dims[1]))
    else:  # Vertical orientation.
        ax1 = fig.add_subplot(gs[1], sharey=ax0)
        plt.setp(ax1.get_yticklabels(), visible=False)
        if "norm" in kwargs and isinstance(kwargs["norm"], LogNorm):
            dos.S.plot(
                ax=ax1,
                _labels=False,
                y="eV",
                xscale="log",
            )
        else:
            dos.S.plot(
                ax=ax1,
                _labels=False,
                y="eV",
            )
        ax0.set_xlabel(str(data.dims[1]))
        ax1.set_xlabel("Intensity")
    ax0.set_ylabel(str(data.dims[0]))
    assert "norm" in kwargs
    assert isinstance(kwargs["norm"], Normalize | None)

    if out:
        savefig(str(out), dpi=400)
        return path_for_plot(out)
    return fig, (ax0, ax1)
