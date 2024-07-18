"""Provides a holoviews based implementation of ImageTool."""

from __future__ import annotations

from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, Unpack

import holoviews as hv
import numpy as np
import xarray as xr
from holoviews import AdjointLayout, DynamicMap, Image, QuadMesh

from arpes.constants import TWO_DIMENSION
from arpes.utilities.combine import concat_along_phi
from arpes.utilities.normalize import normalize_to_spectrum

if TYPE_CHECKING:
    from arpes._typing import ProfileViewParam

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

hv.extension("bokeh")


def _fix_xarray_to_fit_with_holoview(dataarray: xr.DataArray) -> xr.DataArray:
    """Helper function to overcome the problem (#6327) in holoview.

    Args:
        dataarray (xr.DataArray): input Dataarray

    Returns:
        xr.DataArray, whose coordinates is regularly orderd determined by dataarray.dims.
    """
    for coord_name in dataarray.coords:
        if coord_name not in dataarray.dims:
            dataarray = dataarray.drop_vars(str(coord_name))
    return dataarray.assign_coords(
        coords={dim_name: dataarray.coords[dim_name] for dim_name in dataarray.dims},
    )


def concat_along_phi_ui(
    dataarray_a: xr.DataArray,
    dataarray_b: xr.DataArray,
    **kwargs: Unpack[ProfileViewParam],
) -> hv.util.Dynamic:
    """UI for determination of appropriate parameters of concat_along_phi.

    Args:
        dataarray_a: An AREPS data.
        dataarray_b: Another ARPES data.
        use_quadmesh (bool): If true, use hv.QuadMesh instead of hv.Image.
            In most case, hv.Image is sufficient. However, if the coords is irregulaly spaced,
            hv.QuadMesh would be more accurate mapping, but slow.
        kwargs: Options for hv.Image/hv.QuadMesh (width, height, cmap, log)

    Returns:
        [TODO:description]
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
    return dmap.redim.values(ratio=np.linspace(0, 1, 201), magnification=np.linspace(0, 2, 201))


def profile_view(
    dataarray: xr.DataArray,
    *,
    use_quadmesh: bool = False,
    **kwargs: Unpack[ProfileViewParam],
) -> AdjointLayout:
    """Show Profile view interactively.

    Args:
        dataarray: An AREPS data.
        use_quadmesh (bool): If true, use hv.QuadMesh instead of hv.Image.
            In most case, hv.Image is sufficient. However, if the coords is irregulaly spaced,
            hv.QuadMesh would be more accurate mapping, but slow.
        kwargs: Options for hv.Image/hv.QuadMesh (width, height, cmap, log)

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
    posx = hv.streams.PointerX(x=max_coords[dataarray.dims[0]])
    posy = hv.streams.PointerY(y=max_coords[dataarray.dims[1]])

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
    *,
    use_quadmesh: bool = False,
    **kwargs: Unpack[ProfileViewParam],
) -> AdjointLayout:
    """Fit results inspector.

    Args:
        dataset: [TODO:description]
        use_quadmesh (bool): If true, use hv.QuadMesh instead of hv.Image.
            In most case, hv.Image is sufficient. However, if the coords is irregulaly spaced,
            hv.QuadMesh would be more accurate mapping, but very slow.
        kwargs: [TODO:description]

    Returns:
        [TODO:description]
    """
    kwargs.setdefault("width", 300)
    kwargs.setdefault("height", 300)
    kwargs.setdefault("cmap", "viridis")
    kwargs.setdefault("log", False)
    kwargs.setdefault("profile_view_height", 200)

    assert "data" in dataset.data_vars
    arpes_measured: xr.DataArray = _fix_xarray_to_fit_with_holoview(
        dataset.data.S.transpose_to_back("eV"),
    )
    fit = arpes_measured + _fix_xarray_to_fit_with_holoview(
        dataset.residual.S.transpose_to_back("eV"),
    )
    residual = _fix_xarray_to_fit_with_holoview(
        dataset.residual.S.transpose_to_back("eV"),
    )
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
