"""Plotting utilities related to density of states plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import xarray as xr
from matplotlib import colors, gridspec
from matplotlib import pyplot as plt

from arpes.analysis.xps import approximate_core_levels
from arpes.provenance import save_plot_provenance
from arpes.utilities import normalize_to_spectrum

from .utils import path_for_plot, savefig

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

__all__ = (
    "plot_dos",
    "plot_core_levels",
)


@save_plot_provenance
def plot_core_levels(  # noqa: PLR0913
    data: xr.DataArray,
    title: str = "",
    out: str | Path = "",
    norm: Normalize | None = None,
    dos_pow: float = 1,
    core_levels: list[float] | None = None,
    binning: int = 1,
    promenance: int = 5,
) -> Path | tuple[Axes, Colorbar]:
    """Plots an XPS curve and approximate core level locations."""
    plotdos = plot_dos(data=data, title=title, out="", norm=norm, dos_pow=dos_pow)
    assert isinstance(plotdos, tuple)
    _, axes, cbar = plotdos

    if core_levels is None:
        core_levels = approximate_core_levels(data, binning=binning, promenance=promenance)
    assert core_levels is not None
    for core_level in core_levels:
        axes[1].axvline(core_level, color="red", ls="--")

    if out:
        savefig(str(out), dpi=400)
        return path_for_plot(out)
    return axes, cbar


@save_plot_provenance
def plot_dos(
    data: xr.DataArray,
    title: str = "",
    out: str | Path = "",
    norm: Normalize | None = None,
    dos_pow: float = 1,
) -> Path | tuple[Figure, tuple[Axes], Colorbar]:
    """Plots the density of states (momentum integrated) image next to the original spectrum."""
    data_arr = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)

    assert isinstance(data_arr, xr.DataArray)

    fig = plt.figure(figsize=(14, 6))
    fig.subplots_adjust(hspace=0.00)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

    ax0 = plt.subplot(gs[0])
    axes = (ax0, plt.subplot(gs[1], sharex=ax0))

    data_arr.fillna(0)
    cbar_axes = mpl.colorbar.make_axes(axes, pad=0.01)
    mesh = data_arr.plot(ax=axes[0], norm=norm or colors.PowerNorm(gamma=0.15))

    axes[1].set_facecolor((0.95, 0.95, 0.95))
    density_of_states = data_arr.S.sum_other(["eV"])
    (density_of_states**dos_pow).plot(ax=axes[1])

    cbar = plt.colorbar(mesh, cax=cbar_axes[0])
    cbar.set_label("Photoemission Intensity (Arb.)")

    axes[1].set_ylabel("Spectrum DOS", labelpad=12)
    axes[0].set_title(title)

    if out:
        savefig(str(out), dpi=400)
        return path_for_plot(out)
    return fig, axes, cbar
