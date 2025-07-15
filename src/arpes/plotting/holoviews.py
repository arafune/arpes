"""Provides a Holoviews-based implementation of ARPES image inspection and manipulation tools.

This module defines interactive visualization tools based on Holoviews for use in ARPES data
analysis workflows. It supports tasks such as:

- Concatenating two ARPES datasets along the polar angle (`phi`)
- Interactive profile viewing of 2D datasets
- Inspection of fitted model results alongside residuals

All visualizations are designed to work with `xarray.DataArray` or `xarray.Dataset` and are
rendered via the `bokeh` backend of Holoviews.

Dependencies:
    - holoviews
    - numpy
    - xarray
"""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Unpack, cast

import holoviews as hv
import numpy as np
import xarray as xr
from holoviews import AdjointLayout, DynamicMap, Image, QuadMesh

from arpes.constants import TWO_DIMENSION
from arpes.debug import setup_logger
from arpes.utilities.combine import concat_along_phi
from arpes.utilities.normalize import normalize_to_spectrum

if TYPE_CHECKING:
    from holoviews.streams import PointerX, PointerY

    from arpes._typing import ProfileViewParam
LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)

hv.extension("bokeh")


def _fix_xarray_to_fit_with_holoview(dataarray: xr.DataArray) -> xr.DataArray:
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


def _default_plot_kwargs(**kwargs: Unpack[ProfileViewParam]) -> ProfileViewParam:
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


def _get_image_options(
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
    return {
        "width": width,
        "height": height,
        "logz": log,
        "cmap": cmap,
        "clim": clim,
        "active_tools": ["box_zoom"],
        "default_tools": ["save", "box_zoom", "reset", "hover"],
        "framewise": True,
    }


def concat_along_phi_ui(
    dataarray_a: xr.DataArray,
    dataarray_b: xr.DataArray,
    **kwargs: Unpack[ProfileViewParam],
) -> hv.util.Dynamic:
    """Creates an interactive UI to visualize concatenation along the phi axis.

    Allows the user to dynamically adjust the occupation ratio and enhancement
    factor to visualize how two ARPES datasets can be combined along the phi axis.

    Args:
        dataarray_a (xr.DataArray): First ARPES dataset
        dataarray_b (xr.DataArray): Second ARPES dataset
        **kwargs: Additional keyword arguments for visualization settings.
            Supported keys include:
            - width (int): Plot width in pixels.
            - height (int): Plot height in pixels.
            - cmap (str): Colormap name.
            - log (bool): Whether to use log scaling on z-axis.

    Returns:
        holoviews.DynamicMap: A Holoviews DynamicMap with interactive sliders.
    """
    dataarray_a = _fix_xarray_to_fit_with_holoview(dataarray_a)
    dataarray_b = _fix_xarray_to_fit_with_holoview(dataarray_b)
    kwargs.setdefault("width", 300)
    kwargs.setdefault("height", 300)
    kwargs.setdefault("cmap", "viridis")
    kwargs.setdefault("log", False)

    def concate_along_phi_view(
        ratio: float = 0,
        magnification: float = 1,
    ) -> hv.QuadMesh | hv.Image:
        concatenated_data = concat_along_phi(
            dataarray_a,
            dataarray_b,
            occupation_ratio=ratio,
            enhance_a=magnification,
        )
        image_options = {
            "width": kwargs["width"],
            "height": kwargs["height"],
            "logz": kwargs["log"],
            "cmap": kwargs["cmap"],
            "active_tools": ["box_zoom"],
            "default_tools": ["save", "box_zoom", "reset", "hover"],
        }
        return hv.QuadMesh(data=concatenated_data).opts(
            **image_options,
        )

    dmap: DynamicMap = hv.DynamicMap(
        callback=concate_along_phi_view,
        kdims=["ratio", "magnification"],
    )
    return dmap.redim.values(
        ratio=np.linspace(0, 1, 201),
        magnification=np.linspace(0, 2, 201),
    ).redim.default(
        ratio=0,
        magnification=1,
    )


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
    kwargs.setdefault("width", 300)
    kwargs.setdefault("height", 300)
    kwargs.setdefault("cmap", "viridis")
    kwargs.setdefault("log", False)
    kwargs.setdefault("profile_view_height", 100)

    assert dataarray.ndim == TWO_DIMENSION
    dataarray = _fix_xarray_to_fit_with_holoview(dataarray)
    max_coords = dataarray.G.argmax_coords()
    posx: PointerX = hv.streams.PointerX(x=max_coords[dataarray.dims[0]])
    posy: PointerY = hv.streams.PointerY(y=max_coords[dataarray.dims[1]])

    second_weakest_intensity = np.partition(np.unique(dataarray.values.flatten()), 1)[1]
    dataarray = (
        dataarray if isinstance(dataarray, xr.DataArray) else normalize_to_spectrum(dataarray)
    )
    plot_lim: tuple[None | np.float64, np.float64] = (
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
    image_options = {
        "width": kwargs["width"],
        "height": kwargs["height"],
        "logz": kwargs["log"],
        "cmap": kwargs["cmap"],
        "clim": plot_lim,
        "active_tools": ["box_zoom"],
        "default_tools": ["save", "box_zoom", "reset", "hover"],
    }
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


def fit_inspection(
    dataset: xr.Dataset,
    spectral_name: str = "spectrum",
    *,
    use_quadmesh: bool = False,
    **kwargs: Unpack[ProfileViewParam],
) -> AdjointLayout:
    """Displays interactive visualization of fitted ARPES data with residuals.

    This function creates a panel for inspecting model fitting results in ARPES data,
    showing the experimental data, best-fit model, and residuals. A vertical slice view
    enables interactive inspection across energy or momentum axes.

    Args:
        dataset (xr.Dataset): xarray Dataset containing at least modelfit_data and
            modelfit_best_fit.
        spectral_name (str, optional): Prefix for spectral variables, e.g., 'spectrum'.
            Defaults to "spectrum".
        use_quadmesh (bool, optional): If True, use Holoviews QuadMesh for plotting.
            Useful for non-uniform coordinate spacing. Defaults to False.
        **kwargs: Visualization options.
            - width (int): Image width in pixels.
            - height (int): Image height in pixels.
            - cmap (str): Colormap.
            - log (bool): Use logarithmic z-axis.
            - profile_view_height (int): Height/width of profile views.

    Returns:
        holoviews.AdjointLayout: Layout with data image, vline, fit and residual profiles.
    """
    kwargs.setdefault("width", 300)
    kwargs.setdefault("height", 300)
    kwargs.setdefault("cmap", "viridis")
    kwargs.setdefault("log", False)
    kwargs.setdefault("profile_view_height", 200)

    assert any(str(i).endswith("modelfit_data") for i in dataset.data_vars)
    if any(str(i).startswith("modelfit_data") for i in dataset.data_vars):
        exp_data = dataset["modelfit_data"]
    else:
        exp_data = dataset[f"{spectral_name}_modelfit_data"]
    arpes_measured: xr.DataArray = _fix_xarray_to_fit_with_holoview(
        exp_data.transpose(..., "eV"),
    )

    if any(str(i).startswith("modelfit_best_fit") for i in dataset.data_vars):
        fit_data = dataset["modelfit_best_fit"]
    else:
        fit_data = dataset[f"{spectral_name}modelfit_best_fit"]
    fit = _fix_xarray_to_fit_with_holoview(
        fit_data.transpose(..., "eV"),
    )
    residual = arpes_measured - fit

    max_coords = arpes_measured.G.argmax_coords()
    posx = hv.streams.PointerX(x=max_coords[arpes_measured.dims[0]])
    second_weakest_intensity = np.partition(np.unique(arpes_measured.values.flatten()), 1)[1]
    max_height = np.max((fit.max().item(), arpes_measured.max().item()))
    max_residual_abs = np.max((np.abs(residual.min().item()), np.abs(residual.max().item())))
    plotlim_residual = (-max_residual_abs * 1.1, max_residual_abs * 1.1)

    plot_lim: tuple[None | np.float64, np.float64] = (
        (second_weakest_intensity * 0.1, arpes_measured.max().item() * 10)
        if kwargs["log"]
        else (None, max_height * 1.1)
    )
    vline: DynamicMap = hv.DynamicMap(
        lambda x: hv.VLine(x=x or max_coords[arpes_measured.dims[0]]),
        streams=[posx],
    )
    image_options = {
        "width": kwargs["width"],
        "height": kwargs["height"],
        "logz": kwargs["log"],
        "cmap": kwargs["cmap"],
        "clim": plot_lim,
        "active_tools": ["box_zoom"],
        "default_tools": ["save", "box_zoom", "reset", "hover"],
        "framewise": True,
    }
    if use_quadmesh:
        img: QuadMesh | Image = hv.QuadMesh(arpes_measured).opts(**image_options)
    else:
        img = hv.Image(arpes_measured).opts(**image_options)
    profile_arpes = hv.DynamicMap(
        callback=lambda x: hv.Curve(
            arpes_measured.sel(
                **{str(arpes_measured.dims[0]): x},
                method="nearest",
            ),
        ),
        streams=[posx],
    ).opts(
        width=kwargs["profile_view_height"],
        ylim=plot_lim,
        yticks=0,
        xticks=3,
        xlabel="",
    )
    profile_fit = hv.DynamicMap(
        callback=lambda x: hv.Curve(
            fit.sel(
                **{str(arpes_measured.dims[0]): x},
                method="nearest",
            ),
        ),
        streams=[posx],
    )
    profile_residual = hv.DynamicMap(
        callback=lambda x: hv.Curve(
            residual.sel(
                **{str(arpes_measured.dims[0]): x},
                method="nearest",
            ),
        ),
        streams=[posx],
    ).opts(
        invert_axes=True,
        xlabel="",
        width=int(kwargs["profile_view_height"] / 3),
        ylim=plotlim_residual,
        xticks=3,
        yticks=0,
        color="darkgray",
        fontscale=0.5,
        show_grid=True,
        gridstyle={"grid_bounds": (-1, 1), "xgrid_line_dash": [4, 2, 2]},
    )
    return (img * vline << (profile_arpes * profile_fit)) + profile_residual
