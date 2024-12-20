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
    *,
    dark_bg: bool = False,
    **kwargs: Unpack[PColorMeshKwargs],
) -> Path | HTML:
    """Create an animatied plot of ARPES data with time evolution at certain position.

    This function uses matplotlib's pcolormesh to create the plots.

    Args:
        data (xr.DataArray): ARPES data containing time-series data to animate.
        time_dim (str): Dimension name for time, default is "delay"
        interval_ms (float): Delay between frames in milliseconds,  default 100.
        fig_ax (tuple[Figure, Axes]): matplotlib Figure and Axes objects, optional.
        out (str | Path): Output path for saving the animation, optional.
        figsize (tuple[float, float]): Size of the movie figure, optional.
        width_ratio (tuple[float, float]): Width ratio of ARPES data and Time evolution data.
        evolution_at (tuple[str, float] | tuple[str, tuple[float, float]): Position for time
            evolution data, and the value.  if when the latter is tuple of two floats, the first
            value is the center value and the second value is the radius of the range.
        dark_bg (bool): If true, the frame and font color changes to white, default False.
        kwargs: Additional keyword arguments for `pcolormesh`

    Returns:
        Path | HTML: The path to the saved animation or the animation object itself
    """
    figsize = figsize or (9.0, 5.0)
    width_ratio = width_ratio or (1.0, 4.4)
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)

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
        ).transpose(..., time_dim)
    else:
        start, width = evolution_at[1]
        evolution_data = (
            data.sel(
                {
                    evolution_at[0]: slice(
                        start - width,
                        start + width,
                    ),
                },
            )
            .mean(dim=evolution_at[0], keep_attrs=True)
            .transpose(..., time_dim)
        )

    if data.S.is_subtracted:
        kwargs["cmap"] = "RdBu"
        kwargs["vmax"] = np.max([np.abs(kwargs["vmin"]), np.abs(kwargs["vmax"])])
        kwargs["vmin"] = -kwargs["vmax"]

    arpes_data = data.isel({time_dim: 0})
    arpes_mesh: QuadMesh = ax[0].pcolormesh(
        arpes_data.coords[arpes_data.dims[1]].values,
        arpes_data.coords[arpes_data.dims[0]].values,
        arpes_data.values,
        **kwargs,
    )
    ax[0].set_xlabel(str(arpes_data.dims[1]))
    ax[0].set_ylabel(str(arpes_data.dims[0]))
    arpes_mesh.set_animated(True)

    evolution_mesh: QuadMesh = ax[1].pcolormesh(
        evolution_data.coords[evolution_data.dims[1]].values,
        evolution_data.coords[evolution_data.dims[0]].values,
        evolution_data.values,
        **kwargs,
    )
    evolution_mesh.set_animated(True)
    ax[1].set_xlabel(str(evolution_data.dims[1]))
    if evolution_data.dims[0] == arpes_data.dims[0]:
        ax[1].yaxis.set_ticks([])
    cbar: Colorbar = fig.colorbar(evolution_mesh, ax=ax[1])
    fig.tight_layout()

    if dark_bg:
        color_for_darkbackground(obj=cbar)
        color_for_darkbackground(obj=ax[0])
        color_for_darkbackground(obj=ax[1])

    def init() -> Iterable[Artist]:
        ax[1].set_ylabel("")
        return (arpes_mesh, evolution_mesh)

    def update(frame: int) -> Iterable[Artist]:
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

    This function uses matplotlib's pcolormesh to create the plots.

    Args:
        data (xr.DataArray): ARPES data containing time-series data to animate.
        time_dim (str): Dimension name for time, default is "delay".
        interval_ms (float): Delay between frames in milliseconds,  default 100.
        fig_ax (tuple[Figure, Axes]): matplotlib Figure and Axes objects, optional.
        out (str | Path): Output path for saving the animation, optional.
        figsize (tuple[float, float]): Size of the movie figure, optional.
        dark_bg (bool): If true, the frame and font color changes to white, default False.
        kwargs: Additional keyword arguments for `pcolormesh`.

    Returns:
        Path | HTML: The path to the saved animation or the animation object itself

    Raises:
        TypeError: If the argument types are incorrect.
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
    """Change color to fit the dark background.

    This function adjusts the colors of the given Matplotlib Colorbar or Axes
    object to make them suitable for a dark background.

    Args:
        obj (Colorbar | Axes): The Matplotlib Colorbar or Axes object to adjust.
    """
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


def _replace_after_row(array: NDArray[np.float64], row_num: int) -> NDArray[np.float64]:
    """Replace elements in the array with NaN after a specified row.

    Args:
        array (NDArray[np.float64]): The input array.
        row_num (int): The row number after which elements will be replaced with NaN.

    Returns:
        NDArray[np.float64]: The modified array with NaN values after the specified row.
    """
    return np.where(np.arange(array.shape[0])[:, None] >= row_num, np.nan, array)
