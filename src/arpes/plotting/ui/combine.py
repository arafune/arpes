"""Provides a Holoviews-based implementation of ARPES image inspection and manipulation tools.

This module defines interactive visualization tools based on Holoviews for use in ARPES data
analysis workflows. It supports tasks such as:

- Concatenating two ARPES datasets along the polar angle (`phi`)

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
from holoviews import DynamicMap

from arpes.debug import setup_logger
from arpes.utilities.combine import concat_along_phi

from ._helper import default_plot_kwargs, fix_xarray_to_fit_with_holoview, get_image_options

if TYPE_CHECKING:
    import xarray as xr

    from arpes._typing import ProfileViewParam

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)

hv.extension("bokeh", logo=False)


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
    dataarray_a = fix_xarray_to_fit_with_holoview(dataarray_a)
    dataarray_b = fix_xarray_to_fit_with_holoview(dataarray_b)
    kwargs = default_plot_kwargs(**kwargs)

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

        image_options = get_image_options(
            log=kwargs["log"],
            cmap=kwargs["cmap"],
            width=kwargs["width"],
            height=kwargs["height"],
        )
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
