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

    from _typeshed import Incomplete
    from matplotlib.figure import Figure
    from numpy.typing import NDArray


@save_plot_provenance
def false_color_plot(  # noqa: PLR0913
    data_rgb: tuple[xr.Dataset, xr.Dataset, xr.Dataset],
    ax: Axes | None = None,
    out: str | Path = "",
    *,
    invert: bool = False,
    pmin: float = 0,
    pmax: float = 1,
    **kwargs: Incomplete,
) -> Path | tuple[Figure | None, Axes]:
    """Plots a spectrum in false color after conversion to R, G, B arrays."""
    data_r_arr, data_g_arr, data_b_arr = (normalize_to_spectrum(d) for d in data_rgb)
    fig: Figure | None = None
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 5)))
    assert isinstance(ax, Axes)

    def normalize_channel(channel: NDArray[np.float_]) -> NDArray[np.float_]:
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
