"""Simple plotting routes related constant energy slices and Fermi surfaces."""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING

import holoviews as hv
import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from arpes.debug import setup_logger
from arpes.provenance import save_plot_provenance
from arpes.utilities import normalize_to_spectrum

from .utils import path_for_holoviews, path_for_plot

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure
    from matplotlib.typing import ColorType
    from numpy.typing import NDArray


__all__ = (
    "fermi_surface_slices",
    "magnify_circular_regions_plot",
)

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


@save_plot_provenance
def fermi_surface_slices(
    arr: xr.DataArray,
    n_slices: int = 9,
    ev_per_slice: float = 0.02,
    binning: float = 0.01,
    out: str | Path = "",
) -> hv.Layout | Path:
    """Plots many constant energy slices in an axis grid."""
    slices = []
    for i in range(n_slices):
        high = -ev_per_slice * i
        low = high - binning
        image = hv.Image(
            arr.sum(
                [d for d in arr.dims if d not in {"theta", "beta", "phi", "eV", "kp", "kx", "ky"}],
            )
            .sel(eV=slice(low, high))
            .sum("eV"),
            label=f"{high:g} eV",
        )

        slices.append(image)

    layout = hv.Layout(slices).cols(3)
    if out:
        renderer = hv.renderer("matplotlib").instance(fig="svg", holomap="gif")
        filename = path_for_plot(out)
        renderer.save(layout, path_for_holoviews(str(filename)))
        return filename
    return layout


@save_plot_provenance
def magnify_circular_regions_plot(  # noqa: PLR0913
    data: xr.DataArray,
    magnified_points: NDArray[np.float64] | list[float],
    mag: float = 10,
    radius: float = 0.05,
    # below this two can be treated as kwargs?
    cmap: Colormap | ColorType = "viridis",
    color: ColorType | list[ColorType] = "blue",
    edgecolor: ColorType | list[ColorType] = "red",
    out: str | Path = "",
    ax: Axes | None = None,
    **kwargs: tuple[float, float],
) -> tuple[Figure | None, Axes] | Path:
    """Plots a Fermi surface with magnified circular regions as insets.

    This function highlights specified points on a Fermi surface plot by magnifying
    their corresponding regions and displaying them as inset circular regions.

    Args:
        data (xr.DataArray): ARPES data to plot.
        magnified_points: Points on the surface to magnify.
        mag: Magnification factor for the inset regions.
        radius: Radius for the circular regions.
        cmap: Colormap for the plot.
        color: Color of the magnified points.
        edgecolor: Color of the borders around the magnified regions.
        out: File path to save the plot.
        ax: Matplotlib axes to plot on.
        kwargs: Additional keyword arguments for customization.

    Returns:
        A tuple of figure and axes, or the path to the saved plot.
    """
    data_arr = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    assert isinstance(data_arr, xr.DataArray)

    fig: Figure | None = None
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 5)))

    assert isinstance(ax, Axes)

    mesh = data_arr.S.plot(ax=ax, cmap=cmap)
    clim = list(mesh.get_clim())
    clim[1] = clim[1] / mag

    pts = np.zeros(
        shape=(
            len(data_arr.values.ravel()),
            2,
        ),
    )
    mask = np.zeros(shape=len(data_arr.values.ravel())) > 0

    raveled = data_arr.G.ravel()
    pts[:, 0] = raveled[data_arr.dims[0]]
    pts[:, 1] = raveled[data_arr.dims[1]]

    x0, y0 = ax.transAxes.transform((0, 0))  # lower left in pixels
    x1, y1 = ax.transAxes.transform((1, 1))  # upper right in pixels
    dx = x1 - x0
    dy = y1 - y0
    maxd = max(dx, dy)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    width = radius * maxd / dx * (xlim[1] - xlim[0])
    height = radius * maxd / dy * (ylim[1] - ylim[0])

    if not isinstance(edgecolor, list):
        edgecolor = [edgecolor for _ in range(len(magnified_points))]

    if not isinstance(color, list):
        color = [color for _ in range(len(magnified_points))]

    pts[:, 1] = (pts[:, 1]) / (xlim[1] - xlim[0])
    pts[:, 0] = (pts[:, 0]) / (ylim[1] - ylim[0])
    logger.debug(np.min(pts[:, 1]), np.max(pts[:, 1]))
    logger.debug(np.min(pts[:, 0]), np.max(pts[:, 0]))

    for c, ec, point in zip(color, edgecolor, magnified_points, strict=True):
        patch = matplotlib.patches.Ellipse(
            point,
            width,
            height,
            color=c,
            edgecolor=ec,
            fill=False,
            linewidth=2,
            zorder=4,
        )
        patchfake = matplotlib.patches.Ellipse((point[1], point[0]), radius, radius)
        ax.add_patch(patch)
        mask = np.logical_or(mask, patchfake.contains_points(pts))

    data_masked = data_arr.copy(deep=True)
    data_masked.values = np.array(data_masked.values, dtype=np.float64)

    cm = matplotlib.colormaps.get_cmap(cmap="viridis")
    cm.set_bad(color=(1, 1, 1, 0))
    data_masked.values[
        np.swapaxes(np.logical_not(mask.reshape(data_arr.values.shape[::-1])), 0, 1)
    ] = np.nan

    aspect = ax.get_aspect()
    extent = (xlim[0], xlim[1], ylim[0], ylim[1])
    ax.imshow(
        data_masked.values,
        cmap=cm,
        extent=extent,
        zorder=3,
        clim=clim,
        origin="lower",
    )
    ax.set_aspect(aspect)

    for spine in ["left", "top", "right", "bottom"]:
        ax.spines[spine].set_zorder(5)

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax
