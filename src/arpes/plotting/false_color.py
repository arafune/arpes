"""Provides RGB (false color) plotting for spectra."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes

from arpes.provenance import save_plot_provenance
from arpes.utilities import normalize_to_spectrum

from .utils import imshow_arr, path_for_plot

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.figure import Figure
    from numpy.typing import NDArray


@save_plot_provenance
def false_color_plot(  # noqa: PLR0913
    data_rgb: tuple[xr.Dataset, xr.Dataset, xr.Dataset],
    ax: Axes | None = None,
    out: str | Path = "",
    *,
    invert: bool = False,
    pmin_pmax: tuple[float, float] = (0, 1),
    figsize: tuple[float, float] = (7, 5),
) -> Path | tuple[Figure | None, Axes]:
    """Plots a spectrum in false color after conversion to R, G, B arrays.

    Args:
        data_rgb (tuple[xr.Dataset, xr.Dataset, xr.Dataset]): Tuple containing the R, G, B datasets.
        ax (Axes | None, optional): Matplotlib Axes object. If None, a new figure and axes are
            created.
        out (str | Path, optional): Path to save the plot. If empty, the plot is not saved.
        invert (bool, optional): If True, inverts the colors in the HSV space.
        pmin_pmax (tuple[float, float], optional): Percentile range for normalization.
        figsize (tuple[float, float], optional): Size of the figure if a new one is created.

    Returns:
        Path: If `out` is specified, returns the path where the plot is saved.
        tuple[Figure | None, Axes]: If `out` is not specified, returns the figure and axes objects.
    """
    data_r_arr, data_g_arr, data_b_arr = (normalize_to_spectrum(d) for d in data_rgb)
    pmin, pmax = pmin_pmax

    fig: Figure | None = None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    assert isinstance(ax, Axes)

    def normalize_channel(channel: NDArray[np.float64]) -> NDArray[np.float64]:
        channel -= np.percentile(channel, 100 * pmin)
        channel[channel > np.percentile(channel, 100 * pmax)] = np.percentile(channel, 100 * pmax)
        return channel / np.max(channel)

    cs = dict(data_r_arr.coords)
    cs["dim_color"] = [1, 2, 3]

    arr = xr.DataArray(
        np.stack(
            [
                normalize_channel(data_r_arr.values),
                normalize_channel(data_g_arr.values),
                normalize_channel(data_b_arr.values),
            ],
            axis=-1,
        ),
        coords=cs,
        dims=[*list(data_r_arr.dims), "dim_color"],
    )

    if invert:
        vs = arr.values
        vs[vs > 1] = 1
        hsv = matplotlib.colors.rgb_to_hsv(vs)
        hsv[:, :, 2] = 1 - hsv[:, :, 2]
        arr.values = matplotlib.colors.hsv_to_rgb(hsv)

    imshow_arr(arr, ax=ax)

    if out:
        plt.savefig(path_for_plot(out), dpi=400)
        return path_for_plot(out)

    return fig, ax
