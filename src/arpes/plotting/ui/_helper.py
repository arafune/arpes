"""Helper functions for plotting ARPES data with Holoviews."""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Unpack, cast

from arpes.debug import setup_logger

if TYPE_CHECKING:
    import xarray as xr

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
