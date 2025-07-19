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
from typing import TYPE_CHECKING, Unpack, cast

import holoviews as hv
import panel as pn
from holoviews.operation.datashader import regrid

from arpes.analysis import (
    boxcar_filter_arr,
    curvature1d,
    curvature2d,
    dn_along_axis,
    gaussian_filter_arr,
    minimum_gradient,
)
from arpes.analysis.filters import savgol_filter_multi
from arpes.constants import TWO_DIMENSION
from arpes.debug import setup_logger

from ._helper import get_image_options
from .base import BaseUI

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    import xarray as xr
    from param.parameterized import Event

    from arpes._typing import ProfileViewParam

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[0]
logger = setup_logger(__name__, LOGLEVEL)

hv.extension("bokeh", logo=False)
pn.extension()


class SmoothingApp(BaseUI):
    """An interactive smoothing UI for xarray DataArray using Panel and HoloViews."""

    def __init__(self, data: xr.DataArray, **kwargs: Unpack[ProfileViewParam]) -> None:
        """Initialize the SmoothingApp with data and parameters.

        Args:
            data (xr.DataArray): Input data to be smoothed.
            **kwargs: Additional parameters for the UI, such as pane_kwargs.
        """
        super().__init__(data, **kwargs)

        self._build()

    def _build(self) -> None:
        self.pane_kwargs["height"] = 400
        self.pane_kwargs["width"] = 450
        self.pane_kwargs.setdefault("colorbar", True)

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

        self.output_name = pn.widgets.TextInput(
            name="Output Name",
            placeholder="e.g., smoothed1",
        )
        self.output_pane = pn.pane.HoloViews(
            height=self.pane_kwargs["height"],
            width=self.pane_kwargs["width"],
        )

        self._update_plot()

        self.output_button = pn.widgets.Button(
            name="Apply",
            button_type="primary",
        )
        self.output_button.on_click(self._on_apply)

        self.param_widgets_box = pn.Column()
        self._update_smooth_param_widgets()

        self.smoothing_select.param.watch(
            self._update_smooth_param_widgets,
            "value",
        )

        self.widgets_panel = pn.Column(
            self.smoothing_select,
            self.param_widgets_box,
            self.output_name,
            self.output_button,
        )

        self.layout = pn.Row(
            self.output_pane,
            pn.Column(
                self.widgets_panel,
                pn.layout.Divider(),
                self.message_pane,
            ),
        )

    def _get_current_params(self) -> dict[str, float | int]:
        """Retrieve current values from parameter widgets.

        Returns:
            dict[str, float | int]: Parameter names and their current values.
        """
        _, param_widgets = self.smoothing_funcs[str(self.smoothing_select.value)]
        return {name: widget.value for name, widget in param_widgets.items()}

    def _update_smooth_param_widgets(self, *_: Event) -> None:
        """Update the parameter widgets based on the selected smoothing function."""
        _, param_widgets = self.smoothing_funcs[str(self.smoothing_select.value)]
        self.param_widgets_box.objects = list(param_widgets.values())

    def _on_apply(self, _: Event) -> None:
        """Callback when Apply button is clicked. Applies the selected filter."""
        smooth_func, __ = self.smoothing_funcs[str(self.smoothing_select.value)]
        kwargs = self._get_current_params()
        self.output = smooth_func(self.data, **kwargs)
        name = self.output_name.value
        if name:
            self.named_output[name] = self.output
        self._update_plot()

    def _update_plot(self) -> None:
        """Update the HoloViews plot with the current (smoothed) data."""
        plot_data = self.output

        if plot_data.ndim == 1:
            curve = hv.Curve(plot_data, kdims=[plot_data.dims[0]])
            self.output_pane.object = curve.opts(height=self.pane_kwargs["height"])
        elif plot_data.ndim == TWO_DIMENSION:
            image_options = get_image_options(
                log=self.pane_kwargs["log"],
                cmap=self.pane_kwargs["cmap"],
                width=self.pane_kwargs["width"],
                height=self.pane_kwargs["height"],
            )
            image_options["xlabel"] = plot_data.dims[1]
            image_options["ylabel"] = plot_data.dims[0]
            image_options["colorbar"] = self.pane_kwargs["colorbar"]

            img = hv.Image(
                (
                    plot_data.coords[plot_data.dims[1]],
                    plot_data.coords[plot_data.dims[0]],
                    plot_data.values,
                ),
            )
            self.output_pane.object = regrid(img).opts(**image_options)

    def _gaussian_smoothing(self, data: xr.DataArray, **kwargs: float) -> xr.DataArray:
        iteration = kwargs.pop("iteration", 1)
        sigma = cast("dict[Hashable, float]", kwargs)
        return gaussian_filter_arr(
            arr=data,
            sigma=sigma,
            iteration_n=int(iteration),
        )

    def _savitzky_golay_smoothing(self, data: xr.DataArray, **kwargs: float) -> xr.DataArray:
        axis_params: dict[Hashable, tuple[int, int]] = {}
        for k, v in kwargs.items():
            param_name, axis_name = k.rsplit("_", 1)
            if axis_name not in axis_params:
                axis_params[axis_name] = (1, 0)
            if param_name == "window_length":
                axis_params[axis_name] = (int(v), axis_params[axis_name][1])
            else:  # polyorder
                axis_params[axis_name] = (axis_params[axis_name][0], int(v))
        axis_params = {k: tuple(v) for k, v in axis_params.items()}
        for v in axis_params.values():
            if v[0] % 2 == 0:
                self.log_message("❌ Window length must be odd for Savitzky-Golay filter.\n")
                return data
            if v[0] < v[1]:
                self.log_message("❌ Polyorder must be less than window_length.\n")
                return data
        return savgol_filter_multi(data, axis_params=axis_params)

    def _boxcar_smoothing(self, data: xr.DataArray, **kwargs: float) -> xr.DataArray:
        iteration = int(kwargs.pop("iteration", 1))
        size = cast("dict[Hashable, float]", kwargs)
        return boxcar_filter_arr(
            arr=data,
            size=size,
            iteration_n=iteration,
        )


class DifferentiateApp(SmoothingApp):
    """An interactive differentiation UI for xarray DataArray using Panel and HoloViews.

    After smoothing, Differentiate, Maximum curvaure (1D, 2D) and Minimum gradient techniques
    applied to find the peak position.
    """

    def __init__(self, data: xr.DataArray, **kwargs: Unpack[ProfileViewParam]) -> None:
        """Initialize the DifferentiationApp with data and parameters.

        Args:
            data (xr.DataArray): Input data to be differentiated.
            **kwargs: Additional parameters for the UI, such as pane_kwargs.
        """
        super().__init__(data, **kwargs)

    def _build(self) -> None:
        """Build the differentiation UI components."""
        super()._build()
        self.derivative_funcs: dict[
            str,
            tuple[
                Callable[..., xr.DataArray],
                dict[Hashable, pn.widgets.Widget],
            ],
        ] = {
            "None": (lambda x: x, {}),
            "Derivative": (
                self._derivative,
                _derivative_slider(self.data),
            ),
            "Maximum curvature (1D)": (
                self._maximum_curvature_1d,
                _max_curvature_1d_slider(self.data),
            ),
            "Maximum curvature (2D)": (
                self._maximum_curvature_2d,
                _max_curvature_2d_slider(),
            ),
            "Minimum Gradient": (
                self._minimum_gradient,
                {},
            ),
        }

        self.derivation_select = pn.widgets.Select(
            name="Derivative Function",
            options=list(
                self.derivative_funcs,
            ),
        )

        self.derivative_param_widgets_box = pn.Column()
        self._update_derivative_param_widgets()

        self.derivation_select.param.watch(self._update_derivative_param_widgets, "value")

        self.widgets_panel = pn.Column(
            self.smoothing_select,
            self.param_widgets_box,
            self.output_name,
            self.derivation_select,
            self.derivative_param_widgets_box,
            self.output_button,
        )

        self.layout = pn.Row(
            self.output_pane,
            pn.Column(
                self.widgets_panel,
                pn.layout.Divider(),
                self.message_pane,
            ),
        )

    def _update_derivative_param_widgets(self, *_: Event) -> None:
        """Update the parameter widgets based on the selected smoothing function."""
        _, param_widgets = self.derivative_funcs[str(self.derivation_select.value)]
        self.derivative_param_widgets_box.objects = list(param_widgets.values())

    def _on_apply(self, _: Event) -> None:
        """Callback when Apply button is clicked.ArithmeticError.

        Applies the selected filter and then selected derivative procedure.
        """
        smooth_func, __ = self.smoothing_funcs[str(self.smoothing_select.value)]
        kwargs = self._get_current_params()
        self.output = smooth_func(self.data, **kwargs)

        derivative_func, __ = self.derivative_funcs[str(self.derivation_select.value)]
        derivative_kwargs = self._get_current_derivative_params()
        self.output = derivative_func(self.output, **derivative_kwargs)

        name = self.output_name.value
        if name:
            self.named_output[name] = self.output
        self._update_plot()

    def _get_current_derivative_params(self) -> dict[str, float | int | str]:
        """Retrieve current values from parameter widgets.

        Returns:
            dict[str, float | int | str]: Parameter names and their current values.
        """
        _, param_widgets = self.derivative_funcs[str(self.derivation_select.value)]
        return {k: v.value for k, v in param_widgets.items()}

    def _derivative(self, data: xr.DataArray, **kwargs: int) -> xr.DataArray:
        axis = kwargs.get("axis", data.dims[0])
        return dn_along_axis(data, dim=axis, order=kwargs.get("derivative_order", 1))

    def _maximum_curvature_1d(self, data: xr.DataArray, **kwargs: int) -> xr.DataArray:
        axis = kwargs.get("axis", data.dims[0])
        return curvature1d(data, dim=axis, alpha=kwargs.get("coefficient a", 0.1))

    def _maximum_curvature_2d(self, data: xr.DataArray, **kwargs: int) -> xr.DataArray:
        dims = cast("tuple[Hashable, Hashable]", kwargs.get("dims", data.dims))
        if kwargs.get("weight_2D", 1.0) == 0:
            self.log_message("❌ weight 2D must not be 0\n")
            return data
        return curvature2d(
            data,
            dims=dims,
            alpha=kwargs.get("coefficient a", 0.1),
            weight2d=kwargs.get("weight_2D", 1.0),
        )

    def _minimum_gradient(self, data: xr.DataArray, **kwargs: int) -> xr.DataArray:
        del kwargs
        return minimum_gradient(data)


# --------- Helper Functions ---------#


def _derivative_slider(data: xr.DataArray) -> dict[Hashable, pn.widgets.Widget]:
    """Generate a dictionary of sliders for derivative.

    Args:
        data(xr.DataArray): DataArray to be processed.

    Returns:
        dict[str, pn.widgets.Widget]: A dictionary of slider widgets.
    """
    return {
        "axis": pn.widgets.Select(name="axis", options=list(data.dims)),
        "derivative_order": pn.widgets.IntSlider(
            name="Derivative Order",
            value=1,
            start=1,
            end=10,
            step=1,
        ),
    }


def _max_curvature_1d_slider(data: xr.DataArray) -> dict[Hashable, pn.widgets.Widget]:
    """Generate a dictionary of sliders for 1D maximum curvature.

    Args:
        data(xr.DataArray): DataArray to be processed.

    Returns:
        dict[str, pn.widgets.Widget]: A dictionary of slider widgets.
    """
    return {
        "axis": pn.widgets.Select(name="axis", options=list(data.dims)),
        "coefficient a": pn.widgets.FloatSlider(
            name="Coefficient a",
            value=0.1,
            start=0.0,
            end=1,
            step=0.0001,
            format="0.0000",
        ),
    }


def _max_curvature_2d_slider() -> dict[Hashable, pn.widgets.Widget]:
    """Generate a dictionary of sliders for 2D maximum curvature.

    Returns:
        dict[str, pn.widgets.Widget]: A dictionary of slider widgets.
    """
    return {
        "coefficient a": pn.widgets.FloatSlider(
            name="Coefficient a",
            value=0.1,
            start=0.0,
            end=1,
            step=0.0001,
            format="0.0000",
        ),
        "weight_2D": pn.widgets.FloatSlider(
            name="Weight 2D",
            start=-10.0,
            end=10.0,
            step=0.001,
            value=1.0,
            format="0.000",
        ),
    }


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
            format="0.000",
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
            format="0.000",
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
