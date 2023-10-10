"""Simple plotting routes related constant energy slices and Fermi surfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

import holoviews as hv  # pylint: disable=import-error
import matplotlib.patches
import matplotlib.path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from .utils import path_for_holoviews, path_for_plot
from ..provenance import save_plot_provenance
from ..utilities import normalize_to_spectrum

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure
    from matplotlib.typing import ColorType
    from numpy.typing import NDArray

    from arpes._typing import DataType


__all__ = (
    "fermi_surface_slices",
    "magnify_circular_regions_plot",
)


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
                [d for d in arr.dims if d not in ["theta", "beta", "phi", "eV", "kp", "kx", "ky"]],
            )
            .sel(eV=slice(low, high))
            .sum("eV"),
            label="%g eV" % high,
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
def magnify_circular_regions_plot(
    data: DataType,
    magnified_points: NDArray[np.float_] | list[float],
    mag: float = 10,
    radius: float = 0.05,
    # below this can be treated as kwargs?
    cmap: Colormap | ColorType = "viridis",
    color: ColorType | None = None,
    edgecolor: ColorType = "red",
    out: str | Path = "",
    ax: Axes | None = None,
    **kwargs: tuple[float, float],
) -> tuple[Figure | None, Axes] | Path:
    """Plots a Fermi surface with inset points magnified in an inset.

    Args:
        data: [TODO:description]
        magnified_points: [TODO:description]
        mag: [TODO:description]
        radius: [TODO:description]
        cmap: [TODO:description]
        color: [TODO:description]
        edgecolor: [TODO:description]
        out: [TODO:description]
        ax: [TODO:description]
        kwargs: [TODO:description]

    Returns:
        [TODO:description]
    """
    data_arr = normalize_to_spectrum(data)
    assert isinstance(data_arr, xr.DataArray)

    fig: Figure | None = None
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 5)))

    assert isinstance(ax, Axes)

    mesh = data_arr.S.plot(ax=ax, cmap=cmap)
    clim = list(mesh.get_clim())
    clim[1] = clim[1] / mag

    mask = np.zeros(shape=(len(data_arr.values.ravel()),))
    pts = np.zeros(
        shape=(
            len(data_arr.values.ravel()),
            2,
        ),
    )
    mask = mask > 0

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
    assert isinstance(color, list)

    pts[:, 1] = (pts[:, 1]) / (xlim[1] - xlim[0])
    pts[:, 0] = (pts[:, 0]) / (ylim[1] - ylim[0])
    print(np.min(pts[:, 1]), np.max(pts[:, 1]))
    print(np.min(pts[:, 0]), np.max(pts[:, 0]))

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
        patchfake = matplotlib.patches.Ellipse([point[1], point[0]], radius, radius)
        ax.add_patch(patch)
        mask = np.logical_or(mask, patchfake.contains_points(pts))

    data_masked = data_arr.copy(deep=True)
    data_masked.values = np.array(data_masked.values, dtype=np.float_)

    cm = matplotlib.colormaps.get_cmap(cmap="viridis")
    cm.set_bad(color=(1, 1, 1, 0))
    data_masked.values[
        np.swapaxes(np.logical_not(mask.reshape(data_arr.values.shape[::-1])), 0, 1)
    ] = np.nan

    aspect = ax.get_aspect()
    extent = (xlim[0], xlim[1], ylim[0], ylim[1])
    ax.imshow(data_masked.values, cmap=cm, extent=extent, zorder=3, clim=clim, origin="lower")
    ax.set_aspect(aspect)

    for spine in ["left", "top", "right", "bottom"]:
        ax.spines[spine].set_zorder(5)

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax
