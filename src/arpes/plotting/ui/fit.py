"""Provides a Holoviews-based implementation of ARPES image inspection and manipulation tools.

This module defines interactive visualization tools based on Holoviews for use in ARPES data
analysis workflows. It supports tasks such as:

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
from typing import TYPE_CHECKING, Unpack

import holoviews as hv
import numpy as np
from holoviews import AdjointLayout, DynamicMap, Image, QuadMesh

from arpes.debug import setup_logger

from ._helper import default_plot_kwargs, fix_xarray_to_fit_with_holoview, get_image_options

if TYPE_CHECKING:
    import xarray as xr

    from arpes._typing import ProfileViewParam

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


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
    kwargs = default_plot_kwargs(**kwargs)
    kwargs.setdefault("profile_view_height", 200)

    assert any(str(i).endswith("modelfit_data") for i in dataset.data_vars)
    if any(str(i).startswith("modelfit_data") for i in dataset.data_vars):
        exp_data = dataset["modelfit_data"]
    else:
        exp_data = dataset[f"{spectral_name}_modelfit_data"]
    arpes_measured: xr.DataArray = fix_xarray_to_fit_with_holoview(
        exp_data.transpose(..., "eV"),
    )

    if any(str(i).startswith("modelfit_best_fit") for i in dataset.data_vars):
        fit_data = dataset["modelfit_best_fit"]
    else:
        fit_data = dataset[f"{spectral_name}modelfit_best_fit"]
    fit = fix_xarray_to_fit_with_holoview(
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
    image_options = get_image_options(
        log=kwargs["log"],
        cmap=kwargs["cmap"],
        width=kwargs["width"],
        height=kwargs["height"],
        clim=plot_lim,
    )
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
