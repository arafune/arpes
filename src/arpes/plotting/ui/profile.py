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
import panel as pn
import xarray as xr
from holoviews import AdjointLayout, DynamicMap, Image, QuadMesh
from holoviews.streams import PointerX, PointerY

from arpes.constants import TWO_DIMENSION
from arpes.debug import setup_logger
from arpes.utilities.normalize import normalize_to_spectrum

from ._helper import default_plot_kwargs, fix_xarray_to_fit_with_holoview, get_image_options
from .base import BaseUI

if TYPE_CHECKING:
    from arpes._typing import ProfileViewParam

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)

hv.extension("bokeh", logo=False)


class ProfileApp(BaseUI):
    """Interactive ARPES profile viewer application.

    This class provides a user interface for inspecting 2D ARPES datasets interactively.
    It allows users to view intensity profiles along both axes of the dataset by moving the pointer.
    """

    def __init__(
        self,
        data: xr.DataArray,
        *,
        use_quadmesh: bool = False,
        **kwargs: Unpack[ProfileViewParam],
    ) -> None:
        """Initialize the SmoothingApp with data and parameters.

        Args:
            data (xr.DataArray): Input data to be smoothed.
            use_quadmesh (bool, optional): If True, uses Holoviews QuadMesh instead of Image.
                Useful for irregular coordinate grids. Defaults to False.
            **kwargs: Additional parameters for the UI, such as pane_kwargs.
        """
        super().__init__(data, **kwargs)

        self.use_quadmesh = use_quadmesh

        max_coords = data.G.argmax_coords()
        self.posx = PointerX(x=max_coords[data.dims[0]])
        self.posy = PointerY(y=max_coords[data.dims[1]])

        self._build()

    def _build(self) -> None:
        """Builds the interactive profile view layout."""
        self.pane_kwargs["height"] = 400
        self.pane_kwargs["width"] = 450
        self.pane_kwargs.setdefault("colorbar", False)
        self.pane_kwargs.setdefault("profile_view_height", 100)

        self.coord_display = pn.bind(
            self._show_coords,
            self.posx.param.x,
            self.posy.param.y,
        )

        self.output_pane = pn.pane.HoloViews()

        self._update_plot()

        self.layout = pn.Row(
            self.output_pane,
            pn.Column(
                pn.panel(self.coord_display),
            ),
        )

    def _show_coords(self, x: float, y: float) -> str:
        """Displays the current coordinates of the pointer in the plot.

        Args:
            x (float): Current x-coordinate of the pointer.
            y (float): Current y-coordinate of the pointer.

        Returns:
            str: Formatted string showing the current coordinates.
        """
        return f"Coordinates: ({x:.2e}, {y:.2e})"

    def _update_plot(self) -> None:
        """Updates the plot with the current data and parameters."""
        self.output_pane.object = profile_view(
            self.data,
            use_quadmesh=self.use_quadmesh,
            posx=self.posx,
            posy=self.posy,
            **self.pane_kwargs,
        )


def profile_view(
    data: xr.DataArray,
    *,
    use_quadmesh: bool = False,
    posx: PointerX | None = None,
    posy: PointerY | None = None,
    **kwargs: Unpack[ProfileViewParam],
) -> AdjointLayout:
    """Generates an interactive 2D profile view with cross-sectional analysis.

        Enables pointer-based inspection of a 2D ARPES dataset along both axes,
        showing intensity profiles intersecting at the pointer location.

    Args:
        data (xr.DataArray): 2D ARPES dataset.
        use_quadmesh (bool, optional): If True, uses Holoviews QuadMesh instead of Image.
            Useful for irregular coordinate grids. Defaults to False.
        posx (PointerX | None, optional): PointerX stream for x-axis interaction.
        posy (PointerY | None, optional): PointerY stream for y-axis interaction.
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

    assert data.ndim == TWO_DIMENSION
    data = fix_xarray_to_fit_with_holoview(data)
    max_coords = data.G.argmax_coords()

    posx = posx if posx else PointerX(x=max_coords[data.dims[0]])
    posy = posy if posy else PointerY(y=max_coords[data.dims[1]])

    assert isinstance(posx, PointerX)
    assert isinstance(posy, PointerY)

    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)

    plot_lim = _get_plot_lim(
        data,
        log=kwargs["log"],
    )

    vline: DynamicMap = DynamicMap(
        lambda x: hv.VLine(x=x or max_coords[data.dims[0]]),
        streams=[posx],
    )
    hline: DynamicMap = DynamicMap(
        lambda y: hv.HLine(y=y or max_coords[data.dims[1]]),
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
        img: QuadMesh | Image = QuadMesh(data).opts(**image_options)
    else:
        img = Image(data).opts(**image_options)

    profile_x = hv.DynamicMap(
        callback=lambda x: hv.Curve(
            data.sel(
                {str(data.dims[0]): x},
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
            data.sel(
                {str(data.dims[1]): y},
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


def _make_profile_curve(
    dataarray: xr.DataArray,
    dim: str,
    stream: PointerX | PointerY,
    orientation: str,
    plot_lim: tuple[float | None, float],
    profile_size: int,
    log: bool,
) -> DynamicMap:
    """Generate a dynamic cross-sectional profile curve from a 2D DataArray.

    Args:
        dataarray (xr.DataArray): The ARPES dataset to extract profiles from.
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
        return hv.Curve(dataarray.sel({dim: v}, method="nearest"))

    opts: dict[str, Any] = {
        "ylim": plot_lim,
        "logx": log,
    }
    if orientation == "x":
        opts["width"] = profile_size
    else:
        opts["height"] = profile_size

    return hv.DynamicMap(callback, streams=[stream]).opts(**opts)


def _get_plot_lim(dataarray: xr.DataArray, *, log: bool) -> tuple[float | None, float]:
    """Compute appropriate color scale limits for ARPES intensity image.

    Args:
        dataarray (xr.DataArray): The 2D dataset to be plotted.
        log (bool): Whether to use logarithmic color scaling.

    Returns:
        tuple[float | None, float]: Color scale limits (clim) for plotting.
            - If `log` is True: returns (second_min * 0.1, max_val * 10)
            - If `log` is False: returns (None, max_val * 1.1)
    """
    flat_vals = dataarray.values.flatten()
    second_min = np.partition(np.unique(flat_vals), 1)[1]
    max_val = dataarray.max().item()
    if log:
        return (second_min * 0.1, max_val * 10)
    return (None, max_val * 1.1)
