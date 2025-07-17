"""Interactive smoothing application for xarray DataArray using Panel and HoloViews.

This module defines a `SmoothingApp` class which provides a user interface for
applying smoothing filters (e.g., Gaussian) to 1D or 2D xarray DataArrays.
Users can interactively control which axes to smooth and filter parameters,
and visualize the results.

Dependencies:
    - panel
    - holoviews
    - xarray
    - arpes.analysis gaussian_filter_arr, savitzky_golay_filter, boxcar_filter_arr

"""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING

import holoviews as hv
import panel as pn
from holoviews.operation.datashader import regrid

from arpes.analysis import boxcar_filter_arr, gaussian_filter_arr
from arpes.analysis.filters import savgol_filter_multi
from arpes.constants import TWO_DIMENSION
from arpes.debug import setup_logger

from .base import BaseUI

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    import xarray as xr
    from param.parameterized import Event

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[0]
logger = setup_logger(__name__, LOGLEVEL)

hv.extension("bokeh", logo=False)
pn.extension()


class SmoothingApp(BaseUI):
    """An interactive smoothing UI for xarray DataArray using Panel and HoloViews."""

    def _build(self) -> None:
        self.smoothing_funcs: dict[
            str,
            tuple[
                Callable[..., xr.DataArray],
                dict[Hashable, pn.widgets.Widget],
            ],
        ] = {
            "None": (lambda x: x, {}),
            "Gaussian": (
                self._gaussian_smoothing,
                _gaussian_slider(self.data),
            ),
            "Savitzky-Golay": (
                self._savitzky_golay_smoothing,
                _savgol_slider(self.data),
            ),
            "Boxcar": (
                self._boxcar_smoothing,
                _boxcar_slider(self.data),
            ),
        }

        self.smoothing_select = pn.widgets.Select(
            name="Smoothing Function",
            options=list(
                self.smoothing_funcs,
            ),
        )

        self.param_widgets_box = pn.Column()
        self.output_button = pn.widgets.Button(name="Apply", button_type="primary")
        self.output_button.on_click(self._on_apply)

        self._update_param_widgets()
        self.smoothing_select.param.watch(self._update_param_widgets, "value")

        self.output_name = pn.widgets.TextInput(name="Output Name", placeholder="e.g., smoothed1")
        self.output_pane = pn.pane.HoloViews(height=400)
        self.widgets_panel = pn.Column(
            self.smoothing_select,
            self.param_widgets_box,
            self.output_name,
            self.output_button,
        )
        self.layout = pn.Row(
            self.output_pane,
            self.widgets_panel,
        )

        self._update_plot()

    def _get_current_params(self) -> dict[str, float | int]:
        """Retrieve current values from parameter widgets.

        Returns:
            dict[str, Any]: Parameter names and their current values.
        """
        _, param_widgets = self.smoothing_funcs[str(self.smoothing_select.value)]
        return {name: widget.value for name, widget in param_widgets.items()}

    def _update_param_widgets(self, *_: Event) -> None:
        """Update the parameter widgets based on the selected smoothing function."""
        _, param_widgets = self.smoothing_funcs[str(self.smoothing_select.value)]
        self.param_widgets_box.objects = list(param_widgets.values())

    def _on_apply(self, _: Event) -> None:
        """Callback when Apply button is clicked. Applies the selected filter."""
        func, __ = self.smoothing_funcs[str(self.smoothing_select.value)]
        kwargs = self._get_current_params()
        self.output = func(self.data, **kwargs)
        name = self.output_name.value
        if name:
            self.named_output[name] = self.output
        self._update_plot()

    def panel(self) -> pn.layout.Panel:
        """Return the Panel layout for the smoothing application.

        Returns:
            pn.layout.Pane: The Panel layout containing the widgets and output plot.
        """
        return self.layout

    def _update_plot(self) -> None:
        """Update the HoloViews plot with the current (smoothed) data."""
        plot_data = self.output
        if plot_data.ndim == 1:
            curve = hv.Curve(plot_data, kdims=[plot_data.dims[0]])
            self.output_pane.object = curve.opts(height=400)
        elif plot_data.ndim == TWO_DIMENSION:
            img = hv.Image(
                (
                    plot_data.coords[plot_data.dims[1]],
                    plot_data.coords[plot_data.dims[0]],
                    plot_data.values,
                ),
            )
            self.output_pane.object = regrid(img).opts(
                cmap="viridis",
                colorbar=True,
                height=400,
                width=450,
                xlabel=plot_data.dims[1],
                ylabel=plot_data.dims[0],
            )

    def _gaussian_smoothing(self, data: xr.DataArray, **kwargs: float) -> xr.DataArray:
        iteration = kwargs.pop("iteration", 1)
        return gaussian_filter_arr(
            arr=data,
            sigma=kwargs,
            iteration_n=iteration,
        )

    def _savitzky_golay_smoothing(self, data: xr.DataArray, **kwargs: float) -> xr.DataArray:
        axis_params = {}
        for k, v in kwargs.items():
            param_name, axis_name = k.rsplit("_", 1)
            if axis_name not in axis_params:
                axis_params[axis_name] = [1, 0]
            if param_name == "window_length":
                axis_params[axis_name][0] = int(v)
            else:  # polyorder
                axis_params[axis_name][1] = int(v)
        return savgol_filter_multi(data, axis_params=axis_params)

    def _boxcar_smoothing(self, data: xr.DataArray, **kwargs: float) -> xr.DataArray:
        iteration = int(kwargs.pop("iteration", 1))
        return boxcar_filter_arr(
            arr=data,
            size=kwargs,
            iteration_n=iteration,
        )


def _iteration_slider() -> dict[Hashable, pn.widgets.Widget]:
    """Generate a dictionary of iteration sliders.

    Returns:
        dict[str, pn.widgets.Widget]: A dictionary of slider widgets.
    """
    return {
        "iteration": pn.widgets.IntSlider(
            name="Iteration",
            value=1,
            start=1,
            end=10,
            step=1,
        ),
    }


def _gaussian_slider(data: xr.DataArray) -> dict[Hashable, pn.widgets.Widget]:
    """Generate a dictionary of Gaussian smoothing sliders.

    Args:
        data(xr.DataArray): DataArray to be smoothed.

    Returns:
        dict[str, pn.widgets.Widget]: A dictionary of slider widgets.
    """
    sliders = _iteration_slider()
    for dim in data.dims:
        sliders[dim] = pn.widgets.FloatSlider(
            name=f"Sigma {dim}",
            start=0,
            end=3.0,
            step=0.001,
            value=0.1,
        )
    return sliders


def _boxcar_slider(data: xr.DataArray) -> dict[Hashable, pn.widgets.Widget]:
    """Generate a dictionary of boxcar smoothing sliders.

    Args:
        data(xr.DataArray): DataArray to be smoothed.

    Returns:
        dict[str, pn.widgets.Widget]: A dictionary of slider widgets.
    """
    sliders = _iteration_slider()
    for dim in data.dims:
        sliders[dim] = pn.widgets.FloatSlider(
            name=f"Kernel Size {dim}",
            start=0.0,
            end=3.0,
            step=0.001,
            value=0.1,
        )
    return sliders


def _savgol_slider(data: xr.DataArray) -> dict[Hashable, pn.widgets.Widget]:
    """Generate a dictionary of Savitzky-Golay smoothing sliders.

    Args:
        data(xr.DataArray): DataArray to be smoothed.

    Returns:
        dict[str, pn.widgets.Widget]: A dictionary of slider widgets.
    """
    sliders: dict[Hashable, pn.widgets.Widget] = {}
    for dim in data.dims:
        sliders[f"window_length_{dim}"] = pn.widgets.IntSlider(
            name=f"Window Length {dim}",
            start=1,
            end=20,
            step=2,
            value=5,
        )
        sliders[f"polyorder_{dim}"] = pn.widgets.IntSlider(
            name=f"Polyorder {dim}",
            start=0,
            end=6,
            step=1,
            value=1,
        )
    return sliders
