"""This module contains functions for plotting spin-resolved ARPES data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import matplotlib as mpl
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colorbar, colors
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from arpes.analysis.sarpes import to_intensity_polarization
from arpes.analysis.statistics import mean_and_deviation
from arpes.bootstrap import bootstrap
from arpes.provenance import save_plot_provenance

from .tof import scatter_with_std
from .utils import label_for_dim, path_for_plot, savefig

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import xarray as xr
    from _typeshed import Incomplete
    from numpy.typing import ArrayLike, NDArray

__all__ = (
    "spin_colored_spectrum",
    "spin_difference_spectrum",
    "spin_polarized_spectrum",
)


@save_plot_provenance
def spin_colored_spectrum(
    spin_dr: xr.Dataset,
    title: str = "",
    ax: Axes | None = None,
    out: str | Path = "",
    *,
    scatter: bool = False,
) -> Path | None:
    """Plots a spin spectrum using total intensity.

    Assigning color with the spin polarization.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    assert isinstance(ax, Axes)
    as_intensity = to_intensity_polarization(spin_dr)
    intensity = as_intensity.intensity
    pol = as_intensity.polarization.copy(deep=True)

    if len(intensity.dims) == 1:
        inset_ax = inset_axes(ax, width="30%", height="5%", loc="upper right")
        coord = intensity.coords[intensity.dims[0]]
        points = np.array([coord.values, intensity.values]).reshape(-1, 1, 2)
        pol.values[np.isnan(pol.values)] = 0
        pol.values[pol.values > 1] = 1
        pol.values[pol.values < -1] = -1
        pol_colors = mpl.colormaps.get_cmap("RdBu")(pol.values[:-1])

        if scatter:
            pol_colors = mpl.colormaps.get_cmap("RdBu")(pol.values)
            ax.scatter(coord.values, intensity.values, c=pol_colors, s=1.5)
        else:
            segments: Sequence[ArrayLike] = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, colors=pol_colors)

            ax.add_collection(lc)

        ax.set_xlim(coord.min().item(), coord.max().item())
        ax.set_ylim(0, intensity.max().item() * 1.15)
        ax.set_ylabel("ARPES Spectrum Intensity (arb.)")
        ax.set_xlabel(label_for_dim(spin_dr, dim_name=intensity.dims[0]))
        ax.set_title(title or "Spin Polarization")
        _polarization_colorbar(inset_ax)

    if out:
        savefig(str(out), dpi=400)
        plt.clf()
        return path_for_plot(out)
    plt.show()
    return None


def _polarization_colorbar(ax: Axes | None = None) -> colorbar.Colorbar:
    """Makes a colorbar which is appropriate for "polarization" (e.g. spin) data."""
    assert isinstance(ax, Axes)
    return colorbar.Colorbar(
        ax,
        cmap="RdBu",
        norm=colors.Normalize(vmin=-1, vmax=1),
        orientation="horizontal",
        label="Polarization",
        ticks=[-1, 0, 1],
    )


@save_plot_provenance
def spin_difference_spectrum(
    spin_dr: xr.Dataset,
    title: str = "",
    ax: Axes | None = None,
    out: str | Path = "",
    *,
    scatter: bool = False,
) -> Path | None:
    """Plots a spin difference spectrum."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    assert isinstance(ax, Axes)
    try:
        as_intensity = to_intensity_polarization(spin_dr)
    except AssertionError:
        as_intensity = spin_dr
    intensity = as_intensity.intensity
    pol = as_intensity.polarization.copy(deep=True)

    if len(intensity.dims) == 1:
        inset_ax = inset_axes(ax, width="30%", height="5%", loc="upper right")
        coord = intensity.coords[intensity.dims[0]]
        points = np.array([coord.values, intensity.values]).reshape(-1, 1, 2)
        pol.values[np.isnan(pol.values)] = 0
        pol.values[pol.values > 1] = 1
        pol.values[pol.values < -1] = -1
        pol_colors = mpl.colormaps.get_cmap("RdBu")(pol.values[:-1])

        if scatter:
            pol_colors = mpl.colormaps.get_cmap("RdBu")(pol.values)
            ax.scatter(coord.values, intensity.values, c=pol_colors, s=1.5)
        else:
            segments: Sequence[ArrayLike] = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, colors=pol_colors)

            ax.add_collection(lc)

        ax.set_xlim(coord.min().item(), coord.max().item())
        ax.set_ylim(0, intensity.max().item() * 1.15)
        ax.set_ylabel("ARPES Spectrum Intensity (arb.)")
        ax.set_xlabel(label_for_dim(spin_dr, dim_name=intensity.dims[0]))
        ax.set_title(title or "Spin Polarization")
        _polarization_colorbar(inset_ax)

    if out:
        savefig(str(out), dpi=400)
        plt.clf()
        return path_for_plot(out)
    plt.show()
    return None


@save_plot_provenance
def spin_polarized_spectrum(  # noqa: PLR0913
    spin_dr: xr.Dataset,
    title: str = "",
    ax: list[Axes] | None = None,
    out: str | Path = "",
    component: Literal["x", "y", "z"] = "y",
    *,
    scatter: bool = False,
    stats: bool = False,
) -> Path | list[Axes]:
    """Plots a simple spin polarized spectrum using curves for the up and down components."""
    if ax is None:
        _, ax = plt.subplots(2, 1, sharex=True)
    assert ax is not None

    if stats:
        spin_dr = bootstrap(lambda x: x)(spin_dr, N=100)
        pol = mean_and_deviation(to_intensity_polarization(spin_dr))
        counts = mean_and_deviation(spin_dr)
    else:
        counts = spin_dr
        pol = to_intensity_polarization(counts)

    ax_left, ax_right = ax[0], ax[1]

    down, up = counts.down.data, counts.up.data

    energies = spin_dr.coords["eV"].values

    # Plot the spectra
    if stats:
        if scatter:
            scatter_with_std(counts, "up", color="red", ax=ax_left)
            scatter_with_std(counts, "down", color="blue", ax=ax_left)
            scatter_with_std(pol, "polarization", ax=ax_right, color="black")
        else:
            v, s = counts.up.values, counts.up_std.values
            ax_left.plot(energies, v, "r")
            ax_left.fill_between(energies, v - s, v + s, color="r", alpha=0.25)
            v, s = counts.down.values, counts.down_std.values
            ax_left.plot(energies, v, "b")
            ax_left.fill_between(energies, v - s, v + s, color="b", alpha=0.25)
            v, s = pol.polarization.data, pol.polarization_std.data
            ax_right.plot(energies, v, color="black")
            ax_right.fill_between(energies, v - s, v + s, color="black", alpha=0.25)
    else:
        ax_left.plot(energies, up, "r"), ax_left.plot(energies, down, "b")
        ax_right.plot(energies, pol.polarization.data, color="black")
    # Modify axes
    # left
    ax_left.set_title(title or "Spin spectrum {}".format(""))
    (
        ax_left.set_ylabel(
            r"\textbf{Spectrum Intensity}",
        ),
        ax_left.set_xlabel(
            r"\textbf{Kinetic energy} (eV)",
        ),
    )
    ax_left.set_xlim(left=np.min(energies), right=np.max(energies))

    max_up, max_down = np.max(up), np.max(down)
    ax_left.set_ylim(0, max(max_down, max_up) * 1.2)

    # right
    ax_right.fill_between(energies, 0, 1, facecolor="blue", alpha=0.1)
    ax_right.fill_between(energies, -1, 0, facecolor="red", alpha=0.1)

    ax_right.set_title("Spin polarization, $\\text{S}_\\textbf{" + component + "}$")
    ax_right.set_ylabel(
        r"\textbf{Polarization}",
    )
    ax_right.set_xlabel(
        r"\textbf{Kinetic Energy} (eV)",
    )
    ax_right.set_xlim(left=np.min(energies), right=np.max(energies))
    ax_right.axhline(0, color="white", linestyle=":")

    ax_right.set_ylim(-1, 1)
    ax_right.grid(visible=True, axis="y")

    plt.tight_layout()

    if out:
        savefig(str(out), dpi=400)
        plt.clf()
        return path_for_plot(out)

    return ax


def polarization_intensity_to_color(
    data: xr.Dataset,
    vmax: float = 0,
    pmax: float = 1,
) -> NDArray[np.float64]:
    """Converts a dataset with intensity and polarization into a RGB colorarray.

    This consists of a few steps:
    1. first we take the polarization to get a RdBu RGB value
    2. We convert the RGB value to HSV
    3. We use the relative intensity to compute a new value for the V ('value') channel
    4. We convert back to RGB

    Args:
        data: The input intensity/data to convert to a color representation.
        vmax: maximum value for polarization
        pmax: ??.

    Returns:
        The rgb color data.
    """
    if not vmax:
        # use the 98th percentile data if not provided
        vmax = np.percentile(data.intensity.values, 98)

    rgbas = mpl.colormaps["RdBu"]((data.polarization.values / pmax + 1) / 2)
    slices = [slice(None) for _ in data.polarization.dims] + [slice(0, 3)]
    rgbs = rgbas[slices]

    hsvs = matplotlib.colors.rgb_to_hsv(rgbs)

    intensity_values = data.intensity.values.copy() / vmax
    intensity_values[intensity_values > 1] = 1
    hsvs[:, :, 2] = intensity_values

    return matplotlib.colors.hsv_to_rgb(hsvs)


@save_plot_provenance
def hue_brightness_plot(
    data: xr.Dataset,
    ax: Axes | None = None,
    out: str | Path = "",
    **kwargs: Incomplete,
) -> Path | tuple[Figure | None, Axes]:
    """Plog by hue brightness.

    Args:
        data(xr.Dataset): ARPES data
        ax(Axes | None): matplotlib Axes object
        out(str | Path): path string for figure output
        **kwargs: pass to subplot by figsize or pass to "polarization_intensity_to_color".


    """
    assert "intensity" in data
    assert "polarization" in data

    fig: Figure | None = None
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 5)))
    assert isinstance(ax, Axes)
    assert isinstance(fig, Figure)
    x, y = data.coords[data.intensity.dims[0]].values, data.coords[data.intensity.dims[1]].values
    extent = (y[0], y[-1], x[0], x[-1])
    ax.imshow(
        polarization_intensity_to_color(data, **kwargs),
        extent=extent,
        aspect="auto",
        origin="lower",
    )
    ax.set_xlabel(data.intensity.dims[1])
    ax.set_ylabel(data.intensity.dims[0])

    ax.grid(visible=False)

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax
