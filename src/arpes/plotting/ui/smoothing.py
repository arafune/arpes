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
import xarray as xr

from arpes.constants import TWO_DIMENSION
from arpes.debug import setup_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from param.parameterized import Event

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[0]
logger = setup_logger(__name__, LOGLEVEL)

hv.extension("bokeh", logo=False)
pn.extension()


class SmoothingApp:
    """An interactive smoothing UI for xarray DataArray using Panel and HoloViews."""

    def __init__(
        self,
        data: xr.DataArray,
        output_var_name: str = "smoothed",
    ) -> None:
        """Initialize the SmoothingApp.

        Args:
            data (xr.DataArray): The input data array to be smoothed.
            output_var_name (str): Variable name used to store the smoothed xr.DataArray.
        """
        assert len(data.dims) <= TWO_DIMENSION
        self.data: xr.DataArray = data
        self.output_var_name = output_var_name
        self.output = data.copy()
        self.output_var_name = self.output_var_name

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
                _gaussian_slider(data),
            ),
            "Savitzky-Golay": (
                self._savitzky_golay_smoothing,
                _savgol_slider(data),
            ),
            "Boxcar": (
                self._boxcar_smoothing,
                _boxcar_slider(data),
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

        self.output_pane = pn.pane.HoloViews(height=400)
        self.widgets_panel = pn.Column(
            self.smoothing_select,
            self.param_widgets_box,
            self.output_button,
        )

        self.panel_layout = pn.Row(self.widgets_panel, self.output_pane)
        self._update_plot()

    def _get_current_params(self) -> dict[str, float | int]:
        """Retrieve current values from parameter widgets.

        Returns:
            dict[str, Any]: Parameter names and their current values.
        """
        func, param_widgets = self.smoothing_funcs[str(self.smoothing_select.value)]
        return {name: widget.value for name, widget in param_widgets.items()}

    def _update_param_widgets(self, *_: Event) -> None:
        """Update the parameter widgets based on the selected smoothing function."""
        _, param_widgets = self.smoothing_funcs[str(self.smoothing_select.value)]
        self.param_widgets_box.objects = list(param_widgets.values())

    def _on_apply(self, _: Event) -> None:
        """Callback when Apply button is clicked. Applies the selected filter."""
        func, __ = self.smoothing_funcs[self.smoothing_select.value]
        kwargs = self._get_current_params()
        self.output = func(self.data, **kwargs)
        self._update_plot()

    def panel(self) -> pn.layout.Panel:
        """Return the Panel layout for the smoothing application.

        Returns:
            pn.layout.Pane: The Panel layout containing the widgets and output plot.
        """
        return self.panel_layout

    def _update_plot(self) -> None:
        """Update the HoloViews plot with the current (smoothed) data."""

    def _gaussian_smoothing(self, data: xr.DataArray, **kwargs) -> xr.DataArray:
        pass

    def _savitzky_golay_smoothing(self, data: xr.DataArray, **kwargs) -> xr.DataArray:
        pass

    def _boxcar_smoothing(self, data: xr.DataArray, **kwargs) -> xr.DataArray:
        pass


def _generation_iteration_slider() -> dict[Hashable, pn.widgets.Widget]:
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
    sliders = _generation_iteration_slider()
    for dim in data.dims:
        sliders[dim] = pn.widgets.FloatSlider(
            name=f"Sigma {dim}",
            start=0.1,
            end=10.0,
            step=0.1,
            value=1.0,
        )
    return sliders


def _boxcar_slider(data: xr.DataArray) -> dict[Hashable, pn.widgets.Widget]:
    """Generate a dictionary of boxcar smoothing sliders.

    Args:
        data(xr.DataArray): DataArray to be smoothed.

    Returns:
        dict[str, pn.widgets.Widget]: A dictionary of slider widgets.
    """
    sliders = _generation_iteration_slider()
    for dim in data.dims:
        sliders[dim] = pn.widgets.FloatSlider(
            name=f"Kernel Size {dim}",
            start=0.1,
            end=10.0,
            step=0.1,
            value=1.0,
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
            start=1,
            end=20,
            step=1,
            value=2,
        )
    return sliders
