"""Utilities and an example of how to make an animated plot to export as a movie."""

from __future__ import annotations

from typing import TYPE_CHECKING, Unpack

import numpy as np
import xarray as xr
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import arpes.config
from arpes.provenance import save_plot_provenance
from arpes.utilities import normalize_to_spectrum

from .utils import path_for_plot

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.collections import QuadMesh

    from arpes._typing import PColorMeshKwargs

__all__ = ("plot_movie",)


@save_plot_provenance
def plot_movie(  # noqa: PLR0913
    data: xr.DataArray,
    time_dim: str = "delay",
    interval_ms: float = 100,
    fig_ax: tuple[Figure | None, Axes | None] = (None, None),
    out: str | Path = "",
    figsize: tuple[float, float] | None = None,
    **kwargs: Unpack[PColorMeshKwargs],
) -> Path | animation.FuncAnimation:
    """Creates an animated plot of a 3D dataset using one dimension as "time".

    Args:
        data (xr.DataArray): ARPES data
        time_dim (str): Dimension name for time, default is "delay"
        interval_ms (float): Delay between frames in milliseconds
        fig_ax (tuple[Figure, Axes]): matplotlib Figure and Axes objects
        out (str | Path): Output path for saving the animation (optional)
        figsize (tuple[float, float]): Size of the movie figure
        kwargs: Additional keyword arguments for the plot

    Returns:
        Path | animation.FuncAnimation: The path to the saved animation or the animation object
            itself

    Raises:
        TypeError: If the argument types are incorrect.

    """
    figsize = figsize or (7.0, 7.0)
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    fig, ax = fig_ax
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    assert isinstance(ax, Axes)
    assert isinstance(fig, Figure)
    assert isinstance(arpes.config.SETTINGS, dict)

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

    plot: QuadMesh = (
        data.mean(time_dim)
        .transpose()
        .S.plot(
            **kwargs,
        )
    )

    def init() -> tuple[QuadMesh]:
        plot.set_array(np.asarray([]))
        return (plot,)

    animation_coords = data.coords[time_dim].values

    def animate(i: int) -> tuple[QuadMesh]:
        coordinate = animation_coords[i]
        data_for_plot = data.sel({time_dim: coordinate})
        plot.set_array(data_for_plot.values.ravel())
        return (plot,)

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        repeat=500,
        frames=len(animation_coords),
        interval=interval_ms,
        blit=True,
    )

    animation_writer = animation.writers["ffmpeg"]
    writer = animation_writer(
        fps=int(1000 / interval_ms),
        metadata={"artist": "Me"},
        bitrate=1800,
    )

    if out:
        anim.save(str(path_for_plot(out)), writer=writer)
        return path_for_plot(out)

    return anim
