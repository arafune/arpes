"""Helper functions for plotting ARPES data with Holoviews."""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Unpack, cast

import holoviews as hv
import numpy as np
from holoviews import DynamicMap

from arpes.debug import setup_logger

if TYPE_CHECKING:
    import xarray as xr
    from holoviews.streams import PointerX, PointerY

    from arpes._typing import ProfileViewParam

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


def fix_xarray_to_fit_with_holoview(dataarray: xr.DataArray) -> xr.DataArray:
    """Sanitize xarray object for Holoviews plotting.

    Removes non-dimension coordinates and reassigns only the dimensional ones to ensure
    compatibility with Holoviews' plotting logic (e.g., for `Image` or `QuadMesh`).

    Args:
        dataarray(xr.DataArray): Input data to be sanitized for Holoviews.

    Returns: xr.DataArray
        Cleaned data array with only dimension-coordinates.
    """
    for coord_name in dataarray.coords:
        if coord_name not in dataarray.dims:
            dataarray = dataarray.drop_vars(str(coord_name))
    return dataarray.assign_coords({dim: dataarray.coords[dim] for dim in dataarray.dims})


def default_plot_kwargs(**kwargs: Unpack[ProfileViewParam]) -> ProfileViewParam:
    """Set default plotting keyword arguments.

    Args:
        **kwargs: Optional plotting parameters such as width, height, etc.

    Returns: dict
        Updated keyword arguments with defaults filled in.
    """
    kwargs.setdefault("width", 300)
    kwargs.setdefault("height", 300)
    kwargs.setdefault("cmap", "viridis")
    kwargs.setdefault("log", False)
    return cast("ProfileViewParam", kwargs)


def get_image_options(
    *,
    log: bool,
    cmap: str,
    width: int,
    height: int,
    clim: tuple[float, float] | None = None,
) -> dict:
    """Construct Holoviews .opts dictionary for plotting images.

    Args:
        log(bool): Whether to use log scaling on z-axis.
        cmap  (str): Colormap to use.
        width(int): Width of the plot in pixels.
        height(int): Height of the plot in pixels.
        clim(tuple[float, float] | None): Color limit range for z-axis.

    Returns: dict
        Dictionary of options for Holoviews plotting.
    """
    image_options = {
        "width": width,
        "height": height,
        "logz": log,
        "cmap": cmap,
        "active_tools": ["box_zoom"],
        "default_tools": ["save", "box_zoom", "reset", "hover"],
        "framewise": True,
    }
    if clim:
        image_options["clim"] = clim
    return image_options


def get_plot_lim(data: xr.DataArray, *, log: bool) -> tuple[float | None, float]:
    """Compute appropriate color scale limits for ARPES intensity image.

    Args:
        data (xr.DataArray): The 2D dataset to be plotted.
        log (bool): Whether to use logarithmic color scaling.

    Returns:
        tuple[float | None, float]: Color scale limits (clim) for plotting.
            - If `log` is True: returns (second_min * 0.1, max_val * 10)
            - If `log` is False: returns (None, max_val * 1.1)
    """
    flat_vals = data.values.flatten()
    second_min = np.partition(np.unique(flat_vals), 1)[1]
    max_val = data.max().item()
    if log:
        return (second_min * 0.1, max_val * 10)
    return (None, max_val * 1.1)


def make_profile_curve(  # noqa: PLR0913
    data: xr.DataArray,
    dim: str,
    stream: PointerX | PointerY,
    orientation: str,
    plot_lim: tuple[float | None, float],
    profile_size: int,
    *,
    log: bool,
) -> DynamicMap:
    """Generate a dynamic cross-sectional profile curve from a 2D DataArray.

    Args:
        data (xr.DataArray): The ARPES dataset to extract profiles from.
        dim (str): Dimension along which the profile is taken ('kx' or 'E', etc.).
        stream (PointerX | PointerY): Holoviews pointer stream for interactive tracking.
        orientation (str): Either 'x' or 'y', determines if the plot controls width or height.
        plot_lim (tuple[float | None, float]): Limits for the y-axis (intensity).
        profile_size (int): Width or height of the profile plot in pixels.
        log (bool): Whether to apply logarithmic scale to the x-axis.

    Returns:
        holoviews.DynamicMap: Interactive 1D profile plot updated with pointer movement.
    """

    def callback(v: float) -> hv.Curve:
        return hv.Curve(data.sel({dim: v}, method="nearest"))

    opts: dict = {
        "ylim": plot_lim,
        "logx": log,
    }
    if orientation == "x":
        opts["width"] = profile_size
    else:
        opts["height"] = profile_size

    return hv.DynamicMap(callback, streams=[stream]).opts(**opts)
