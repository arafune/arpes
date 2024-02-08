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

from .utils import path_for_plot

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.collections import QuadMesh

    from arpes._typing import PColorMeshKwargs

__all__ = ("plot_movie",)


@save_plot_provenance
def plot_movie(
    data: xr.DataArray,
    time_dim: str = "delay",
    interval: float = 100,
    fig_ax: tuple[Figure | None, Axes | None] = (None, None),
    out: str | Path = "",
    **kwargs: Unpack[PColorMeshKwargs],
) -> Path | animation.FuncAnimation:
    """Make an animated plot of a 3D dataset using one dimension as "time".

    Args:
        data (xr.DataArray): ARPES data
        time_dim (str): dimension name for time
        interval: [TODO:description]
        fig_ax (tuple[Figure, Axes]): matplotlib object
        out: [TODO:description]
        kwargs: [TODO:description]

    Raises:
        TypeError: [TODO:description]
    """
    if not isinstance(data, xr.DataArray):
        msg = "You must provide a DataArray"
        raise TypeError(msg)
    fig, ax = fig_ax
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
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

    if data.S.is_subtracted:
        kwargs["cmap"] = "RdBu"
        kwargs["vmax"] = np.max([np.abs(kwargs["vmin"]), np.abs(kwargs["vmax"])])
        kwargs["vmin"] = -kwargs["vmax"]

    plot: QuadMesh = (
        data.mean(time_dim)
        .transpose()
        .plot(
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
        plot.set_array(data_for_plot.values.G.ravel())
        return (plot,)

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        repeat=500,
        frames=len(animation_coords),
        interval=interval,
        blit=True,
    )

    animation_writer = animation.writers["ffmpeg"]
    writer = animation_writer(
        fps=1000 / interval,
        metadata={"artist": "Me"},
        bitrate=1800,
    )

    if out:
        anim.save(str(path_for_plot(out)), writer=writer)
        return path_for_plot(out)

    return anim
