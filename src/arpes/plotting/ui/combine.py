"""Provides a Holoviews-based implementation of ARPES image inspection and manipulation tools.

This module defines interactive visualization tools based on Holoviews for use in ARPES data
analysis workflows. It supports tasks such as:

- Concatenating two ARPES datasets along the polar angle (`phi`)

All visualizations are designed to work with `xarray.DataArray`, and are
rendered via the `bokeh` backend of Holoviews.

Dependencies:
    - panel
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
from holoviews import DynamicMap

from arpes.constants import TWO_DIMENSION
from arpes.debug import setup_logger
from arpes.utilities.combine import concat_along_phi

from ._helper import default_plot_kwargs, fix_xarray_to_fit_with_holoview, get_image_options
from .base import BaseUI

if TYPE_CHECKING:
    from param.parameterized import Event

    from arpes._typing.plotting import ProfileViewParam

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)

hv.extension("bokeh", logo=False)


class TailorApp(BaseUI):
    """An interactive UI to stick two ARPES data using Panel and HoloViews."""

    def __init__(
        self,
        data: xr.DataArray,
        data_b: xr.DataArray,
        **kwargs: Unpack[ProfileViewParam],
    ) -> None:
        """Initialize the SmoothingApp with data and parameters.

        Args:
            data (xr.DataArray): Input data to be smoothed.
            data_b (xr.DataArray): Second ARPES dataset to concatenate.
            **kwargs: Additional parameters for the UI, such as pane_kwargs.
        """
        super().__init__(data, data_b, **kwargs)
        assert isinstance(data_b, xr.DataArray), "Data must be an xarray DataArray."
        assert data.ndim == TWO_DIMENSION, "Data must be two dimensions."
        assert data_b.ndim == TWO_DIMENSION, "Data must be two dimensions."

        self._build()

    def _build(self) -> None:
        self.pane_kwargs["height"] = 400
        self.pane_kwargs["width"] = 450
        self.pane_kwargs.setdefault("colorbar", True)

        self.output_name = pn.widgets.TextInput(
            name="Output Name",
            placeholder="e.g. id1_3",
        )

        self.toggle_laminate_mode = pn.widgets.Checkbox(
            name="Laminate mode",
            value=False,
        )
        self.toggle_laminate_mode.param.watch(
            self._toggle_ratio_slider,
            "value",
        )

        self.output_name = pn.widgets.TextInput(
            name="Output Name",
            placeholder="e.g., smoothed1",
        )

        self.output_button = pn.widgets.Button(
            name="Output",
            button_type="primary",
        )
        self.output_button.on_click(self._on_apply)

        self.ratio_slider = pn.widgets.FloatSlider(
            name="ratio",
            start=0,
            end=1,
            step=0.01,
            value=0,
        )
        self.ratio_slider.param.watch(
            self._on_slider_change,
            "value",
        )

        self.magnification_slider = pn.widgets.FloatSlider(
            name="magnification",
            start=0,
            end=2,
            step=0.0001,
            value=1,
            format="0.0000",
        )
        self.magnification_slider.param.watch(
            self._on_slider_change,
            "value",
        )

        self._update_plot()

        self.widgets_panel = pn.Column(
            self.toggle_laminate_mode,
            self.ratio_slider,
            self.magnification_slider,
        )

        self.layout = pn.Row(
            self.output_pane,
            pn.Column(
                self.widgets_panel,
                pn.layout.Divider(),
                self.output_name,
                self.output_button,
                self.message_pane,
            ),
        )

    def _on_apply(self, _: Event) -> None:
        """Callback when Output button is clicked."""
        name = self.output_name.value
        if name:
            self.named_output[name] = self.output.copy()
            self.message_log.append(f"Output stored: '{name}'\n")
            self.message_pane.object = "\n".join(self.message_log)

    def _toggle_ratio_slider(
        self,
        event: Event,
    ) -> None:
        """Toggle the visibility of the ratio slider based on checkbox state."""
        self.ratio_slider.disabled = event.new
        self._update_plot()

    def _update_plot(self) -> None:
        """Update the Holoviews plot withthe current stiched data."""
        ratio: float = float(self.ratio_slider.value)
        magnification: float = float(self.magnification_slider.value)

        laminate: bool = bool(self.toggle_laminate_mode.value)

        image_options = get_image_options(
            log=self.pane_kwargs["log"],
            cmap=self.pane_kwargs["cmap"],
            width=self.pane_kwargs["width"],
            height=self.pane_kwargs["height"],
        )
        if laminate:
            self.output = concat_along_phi(
                arr_a=self.data,
                arr_b=self.data_b,
                occupation_ratio=None,
                enhance_a=magnification,
            )
        else:
            self.output = concat_along_phi(
                self.data,
                self.data_b,
                ratio,
                magnification,
            )

        plot_data = self.output
        img = hv.QuadMesh(
            (
                plot_data.coords[plot_data.dims[1]].values,
                plot_data.coords[plot_data.dims[0]].values,
                plot_data.values,
            ),
        )
        self.output_pane.object = img.opts(**image_options)

    def _on_slider_change(self, _: Event) -> None:
        """Handles changes in the slider values and updates the output pane."""
        self._update_plot()


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
