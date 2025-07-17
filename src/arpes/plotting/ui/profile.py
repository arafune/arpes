"""Provides a Holoviews-based implementation of ARPES image inspection and manipulation tools.

This module defines interactive visualization tools based on Holoviews for use in ARPES data
analysis workflows. It supports tasks such as:

- Interactive profile viewing of 2D datasets

All visualizations are designed to work with `xarray.DataArray` or `xarray.Dataset` and are
rendered via the `bokeh` backend of Holoviews.

Dependencies:
    - holoviews
    - numpy
    - xarray
"""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Unpack

import holoviews as hv
import numpy as np
import xarray as xr
from holoviews import AdjointLayout, DynamicMap, Image, QuadMesh

from arpes.constants import TWO_DIMENSION
from arpes.debug import setup_logger
from arpes.utilities.normalize import normalize_to_spectrum

from ._helper import default_plot_kwargs, fix_xarray_to_fit_with_holoview, get_image_options

if TYPE_CHECKING:
    from holoviews.streams import PointerX, PointerY

    from arpes._typing import ProfileViewParam
LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)

hv.extension("bokeh", logo=False)


def profile_view(
    dataarray: xr.DataArray,
    *,
    use_quadmesh: bool = False,
    **kwargs: Unpack[ProfileViewParam],
) -> AdjointLayout:
    """Generates an interactive 2D profile view with cross-sectional analysis.

    Enables pointer-based inspection of a 2D ARPES dataset along both axes,
    showing intensity profiles intersecting at the pointer location.

    Args:
        dataarray (xr.DataArray): 2D ARPES dataset.
        use_quadmesh (bool, optional): If True, uses Holoviews QuadMesh instead of Image.
            Useful for irregular coordinate grids. Defaults to False.
        **kwargs: Additional keyword arguments for visualization.
            - width (int): Image width in pixels.
            - height (int): Image height in pixels.
            - cmap (str): Colormap name.
            - log (bool): Whether to use log scale for intensity.
            - profile_view_height (int): Size of the profile views.

    Returns:
        holoviews.AdjointLayout: Combined Holoviews layout with image and profile views.

    Todo:
    There are some issues.

    * 2024/07/08: On Jupyterlab on safari, it may not work correctly.
    * 2024/07/10: Incompatibility between bokeh and matplotlib about which is "x-" axis about
      plotting xarray data.
    """
    kwargs = default_plot_kwargs(**kwargs)
    kwargs.setdefault("profile_view_height", 100)

    assert dataarray.ndim == TWO_DIMENSION
    dataarray = fix_xarray_to_fit_with_holoview(dataarray)
    max_coords = dataarray.G.argmax_coords()
    posx: PointerX = hv.streams.PointerX(x=max_coords[dataarray.dims[0]])
    posy: PointerY = hv.streams.PointerY(y=max_coords[dataarray.dims[1]])

    second_weakest_intensity = np.partition(np.unique(dataarray.values.flatten()), 1)[1]
    dataarray = (
        dataarray if isinstance(dataarray, xr.DataArray) else normalize_to_spectrum(dataarray)
    )
    plot_lim: tuple[None | float, float] = (
        (second_weakest_intensity * 0.1, dataarray.max().item() * 10)
        if kwargs["log"]
        else (None, dataarray.max().item() * 1.1)
    )
    vline: DynamicMap = hv.DynamicMap(
        lambda x: hv.VLine(x=x or max_coords[dataarray.dims[0]]),
        streams=[posx],
    )
    hline: DynamicMap = hv.DynamicMap(
        lambda y: hv.HLine(y=y or max_coords[dataarray.dims[1]]),
        streams=[posy],
    )
    image_options = get_image_options(
        log=kwargs["log"],
        cmap=kwargs["cmap"],
        width=kwargs["width"],
        height=kwargs["height"],
        clim=plot_lim,
    )
    if use_quadmesh:
        img: QuadMesh | Image = hv.QuadMesh(dataarray).opts(**image_options)
    else:
        img = hv.Image(dataarray).opts(**image_options)

    profile_x = hv.DynamicMap(
        callback=lambda x: hv.Curve(
            dataarray.sel(
                **{str(dataarray.dims[0]): x},
                method="nearest",
            ),
        ),
        streams=[posx],
    ).opts(
        ylim=plot_lim,
        width=kwargs["profile_view_height"],
        logx=kwargs["log"],
    )
    profile_y = hv.DynamicMap(
        callback=lambda y: hv.Curve(
            dataarray.sel(
                **{str(dataarray.dims[1]): y},
                method="nearest",
            ),
        ),
        streams=[posy],
    ).opts(
        ylim=plot_lim,
        height=kwargs["profile_view_height"],
        logx=kwargs["log"],
    )

    return img * hline * vline << profile_x << profile_y
