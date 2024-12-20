"""Utilities and an example of how to make an animated plot to export as a movie."""

from __future__ import annotations

from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, Unpack

import numpy as np
import xarray as xr
from IPython.display import HTML
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
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
    from matplotlib.collections import QuadMesh
    from numpy.typing import NDArray

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


__all__ = ("plot_movie", "plot_movie_and_evolution")


@save_plot_provenance
def plot_movie_and_evolution(  # noqa: PLR0913
    data: xr.DataArray,
    time_dim: str = "delay",
    interval_ms: float = 100,
    fig_ax: tuple[Figure | None, NDArray[np.object_] | None] | None = None,
    out: str | Path = "",
    figsize: tuple[float, float] | None = None,
    width_ratio: tuple[float, float] | None = None,
    evolution_at: tuple[str, float] | tuple[str, tuple[float, float]] = ("phi", 0.0),
    **kwargs: Unpack[PColorMeshKwargs],
) -> Path | HTML:
    """Create an animatied plot of ARPES data with time evolution at certain position.

    Args:
        data (xr.DataArray): ARPES data containing time-series data to animate.
        time_dim (str): Dimension name for time, default is "delay"
        interval_ms (float): Delay between frames in milliseconds,  default 100
        fig_ax (tuple[Figure, Axes]): matplotlib Figure and Axes objects, optional
        out (str | Path): Output path for saving the animation, optional.
        figsize (tuple[float, float]): Size of the movie figure, optional
        width_ratio (tuple[float, float]): Width ratio of ARPES data and Time evolution data.
        evolution_at (tuple[str, float] | tuple[str, tuple[float, float]): [TODO:description]
        kwargs: Additional keyword arguments for `pcolormesh`

    Returns:
        Path | animation.FuncAnimation: The path to the saved animation or the animation object
            itself
    """
    figsize = figsize or (9.0, 6.0)
    width_ratio = width_ratio or (1.0, 3.0)
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    axes_list = list(data.dims)
    axes_list.remove("delay")
    axes_list.remove(evolution_at[0])

    y_axis_evolution_mesh = axes_list[0]
    fig, ax = fig_ax or plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        width_ratios=width_ratio,
    )
    assert ax is not None
    assert isinstance(ax[0], Axes)
    assert isinstance(ax[1], Axes)
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

    if isinstance(evolution_at[1], float):
        evolution_data: xr.DataArray = data.sel(
            {evolution_at[0]: evolution_at[1]},
            method="nearest",
        )
    else:
        start, width = evolution_at[1]
        evolution_data = data.sel(
            {
                evolution_at[0]: slice(
                    start - width,
                    start + width,
                ),
            },
        ).mean(dim=evolution_at[0], keep_attrs=True)

    if data.S.is_subtracted:
        kwargs["cmap"] = "RdBu"
        kwargs["vmax"] = np.max([np.abs(kwargs["vmin"]), np.abs(kwargs["vmax"])])
        kwargs["vmin"] = -kwargs["vmax"]

    arpes_mesh: QuadMesh = data.isel({time_dim: 0}).plot.pcolormesh(
        ax=ax[0],
        add_colorbar=False,
        animated=True,
        **kwargs,
    )

    evolution_mesh: QuadMesh = evolution_data.plot.pcolormesh(
        ax=ax[1],
        add_colorbar=True,
        animated=True,
        **kwargs,
    )

    def init() -> Iterable[Artist]:
        ax[1].set_ylabel("")
        return (arpes_mesh, evolution_mesh)

    def update(frame: int) -> Iterable[Artist]:
        ax[0].set_title("")
        ax[1].set_title(f"pump probe delay={data.coords[time_dim].values[frame]: >9.3f}")
        arpes_mesh.set_array(data.isel({time_dim: frame}).values.ravel())
        evolution_mesh.set_array(
            _replace_after_col(evolution_data.values, col_num=frame + 1).ravel(),
        )
        return (arpes_mesh, evolution_mesh)

    anim: FuncAnimation = FuncAnimation(
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


@save_plot_provenance
def plot_movie(  # noqa: PLR0913
    data: xr.DataArray,
    time_dim: str = "delay",
    interval_ms: float = 100,
    fig_ax: tuple[Figure | None, Axes | None] | None = None,
    out: str | Path = "",
    figsize: tuple[float, float] | None = None,
    *,
    dark_bg: bool = False,
    **kwargs: Unpack[PColorMeshKwargs],
) -> Path | HTML:
    """Create an animated movie of a 3D dataset using one dimension as "time".

    Args:
        data (xr.DataArray): ARPES data containing time-series data to animate.
        time_dim (str): Dimension name for time, default is "delay"
        interval_ms (float): Delay between frames in milliseconds,  default 100
        fig_ax (tuple[Figure, Axes]): matplotlib Figure and Axes objects, optional
        out (str | Path): Output path for saving the animation, optional.
        figsize (tuple[float, float]): Size of the movie figure, optional
        dark_bg (bool): If true, the frame and font color changes to white, default False.
        kwargs: Additional keyword arguments for `pcolormesh`

    Returns:
        Path | animation.FuncAnimation: The path to the saved animation or the animation object
            itself

    Raises:
        TypeError: If the argument types are incorrect.
        RuntimeError: If saving the movie file fails.
    """
    figsize = figsize or (9.0, 5.0)
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
        kwargs["cmap"] = "RdBu_r"
        kwargs["vmax"] = np.max([np.abs(kwargs["vmin"]), np.abs(kwargs["vmax"])])
        kwargs["vmin"] = -kwargs["vmax"]

    arpes_data = data.isel({time_dim: 0})
    arpes_mesh: QuadMesh = ax.pcolormesh(
        arpes_data.coords[arpes_data.dims[1]].values,
        arpes_data.coords[arpes_data.dims[0]].values,
        arpes_data.values,
        **kwargs,
    )
    ax.set_xlabel(str(arpes_data.dims[1]))
    ax.set_ylabel(str(arpes_data.dims[0]))
    arpes_mesh.set_animated(True)
    cbar = fig.colorbar(arpes_mesh, ax=ax)
    if dark_bg:
        color_for_darkbackground(obj=cbar)
        color_for_darkbackground(obj=ax)

    def init() -> Iterable[Artist]:
        ax.set_title(f"pump probe delay={data.coords[time_dim].values[0]: >9.3f}")
        return (arpes_mesh,)

    def update(frame: int) -> Iterable[Artist]:
        ax.set_title(
            f"pump probe delay={data.coords[time_dim].values[frame]: >9.3f}",
        )
        arpes_mesh.set_array(data.isel({time_dim: frame}).values.ravel())
        arpes_mesh.set_animated(True)
        return (arpes_mesh,)

    anim: FuncAnimation = FuncAnimation(
        fig=fig,
        func=update,
        init_func=init,
        frames=data.sizes[time_dim],
        blit=True,
        interval=interval_ms,
    )

    if out:
        logger.debug(msg=f"path_for_plot is {path_for_plot(out)}")
        anim.save(str(path_for_plot(out)), writer="ffmpeg")
        return path_for_plot(out)

    return HTML(anim.to_html5_video())  # HTML(anim.to_jshtml())


def color_for_darkbackground(obj: Colorbar | Axes) -> None:
    """Change color to fit the dark background."""
    if isinstance(obj, Colorbar):
        obj.ax.yaxis.set_tick_params(color="white")
        obj.ax.yaxis.label.set_color("white")
        obj.outline.set_edgecolor("white")
        for label in obj.ax.get_yticklabels():
            label.set_color("white")
    if isinstance(obj, Axes):
        obj.spines["bottom"].set_color("white")
        obj.spines["top"].set_color("white")
        obj.spines["right"].set_color("white")
        obj.spines["left"].set_color("white")
        obj.tick_params(axis="both", colors="white")
        obj.xaxis.label.set_color("white")
        obj.yaxis.label.set_color("white")
        obj.title.set_color("white")


def _replace_after_col(array: NDArray[np.float64], col_num: int) -> NDArray[np.float64]:
    """Replace elements in the array with NaN af ter a specified column.

    Args:
        array (NDArray[np.float64): The input array.
        col_num (int): The column number after which elements will be replaced with NaN.

    Returns:
        NDArray[np.float64]: The modified array with NaN values after the specified column.
    """
    return np.where(np.arange(array.shape[1])[:, None] >= col_num, np.nan, array.T).T
