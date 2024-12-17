"""Utilities and an example of how to make an animated plot to export as a movie."""

from __future__ import annotations

from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, Unpack

import numpy as np
import xarray as xr
from IPython.display import HTML
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import arpes.config
from arpes.constants import TWO_DIMENSION
from arpes.provenance import save_plot_provenance
from arpes.utilities import normalize_to_spectrum

from .utils import path_for_plot

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from matplotlib.artist import Artist

    from arpes._typing import PColorMeshKwargs

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


__all__ = ("plot_movie",)


@save_plot_provenance
def plot_movie(  # noqa: PLR0913
    data: xr.DataArray,
    time_dim: str = "delay",
    interval_ms: float = 100,
    fig_ax: tuple[Figure | None, Axes | None] | None = None,
    out: str | Path = "",
    figsize: tuple[float, float] | None = None,
    **kwargs: Unpack[PColorMeshKwargs],
) -> Path | HTML:
    """Create an animated moview of a 3D dataset using one dimension as "time".

    Args:
        data (xr.DataArray): ARPES data containing time-series data to animate.
        time_dim (str): Dimension name for time, default is "delay"
        interval_ms (float): Delay between frames in milliseconds,  default 100
        fig_ax (tuple[Figure, Axes]): matplotlib Figure and Axes objects, optional
        out (str | Path): Output path for saving the animation, optional.
        figsize (tuple[float, float]): Size of the movie figure, optional
        kwargs: Additional keyword arguments for `pcolormesh`

    Returns:
        Path | animation.FuncAnimation: The path to the saved animation or the animation object
            itself

    Raises:
        TypeError: If the argument types are incorrect.
        RuntimeError: If saving the movie file fails.
    """
    figsize = figsize or (7.0, 7.0)
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    fig, ax = fig_ax or plt.subplots(figsize=figsize)

    assert isinstance(ax, Axes)
    assert isinstance(fig, Figure)
    assert isinstance(data, xr.DataArray)
    assert isinstance(arpes.config.SETTINGS, dict)
    assert data.ndim == TWO_DIMENSION + 1

    kwargs.setdefault(
        "cmap",
        arpes.config.SETTINGS.get("interactive", {}).get(
            "palette",
            "viridis",
        ),
    )
    kwargs.setdefault("vmax", data.max().item())
    kwargs.setdefault("vmin", data.min().item())
    assert "vmax" in kwargs
    assert "vmin" in kwargs
    if data.S.is_subtracted:
        kwargs["cmap"] = "RdBu"
        kwargs["vmax"] = np.max([np.abs(kwargs["vmin"]), np.abs(kwargs["vmax"])])
        kwargs["vmin"] = -kwargs["vmax"]

    def init() -> Iterable[Artist]:
        data.isel({time_dim: 0}).plot.pcolormesh(ax=ax, add_colorbar=True, **kwargs)
        return ax

    def update(frame: int) -> Iterable[Artist]:
        ax.clear()
        data.isel({time_dim: frame}).plot.pcolormesh(ax=ax, add_colorbar=False, **kwargs)
        return ax

    anim: animation.FuncAnimation = animation.FuncAnimation(
        fig=fig,
        func=update,
        init_func=init,
        frames=data.sizes[time_dim],
        interval=interval_ms,
    )

    if out:
        logger.debug(msg=f"path_for_plot is {path_for_plot(out)}")
        anim.save(str(path_for_plot(out)), writer="ffmpeg")
        return path_for_plot(out)

    return HTML(anim.to_html5_video())  # HTML(anim.to_jshtml())
