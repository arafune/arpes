"""Test smoothing operations stores result in named output and match expected output.

This test sets the output_name widget, triggers the apply action, and
verifies that the named output is present and correct.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import holoviews as hv
import numpy as np
import panel as pn
import pytest
import xarray as xr

import arpes.xarray_extensions  # pyright: ignore[reportUnusedImport]  # noqa: F401
from arpes.plotting import DifferentiateApp
from arpes.plotting.ui import smoothing
from arpes.plotting.ui.smoothing import (
    SmoothingApp,
    _boxcar_slider,
    _derivative_slider,
    _gaussian_slider,
    _iteration_slider,
    _max_curvature_1d_slider,
    _max_curvature_2d_slider,
    _savgol_slider,
)

# Assuming your SmoothingApp and helper functions are in a file named 'smoothing_app_module.py'
# from smoothing_app_module import SmoothingApp, _gaussian_slider, _boxcar_slider, _savgol_slider, _iteration_slider
# For this example, I'll put them directly in the test file for simplicity.


# Mocking arpes.analysis and arpes.utilities for testing purposes
# In a real scenario, you would import them and mock in tests.
class MockARPESAnalysis:
    """Mock class for ARPESAnalysis providing dummy smoothing filter methods for testing."""

    def gaussian_filter_arr(self, arr, sigma, iteration_n):
        """Simulate a Gaussian filter by scaling array values for testing."""
        return arr.copy(data=arr.values * 0.5)  # Dummy smoothing for test

    def savitzky_golay_filter(self, arr, window_length, polyorder):
        """Simulate a Savitzky-Golay filter by scaling array values for testing."""
        return arr.copy(data=arr.values * 0.6)  # Dummy smoothing for test

    def boxcar_filter_arr(self, arr, size, iteration_n):
        """Simulate a boxcar filter by scaling array values for testing."""
        return arr.copy(data=arr.values * 0.7)  # Dummy smoothing for test


class MockARPESUtilities:
    """Mock class for ARPESUtilities providing dummy function for testing."""

    def normalize_to_spectrum(self, data):
        if isinstance(data, xr.DataArray):
            return data
        # Simulate conversion to DataArray if not already
        return xr.DataArray([1, 2, 3], dims=["test_dim"])


arpes_analysis = MockARPESAnalysis()
arpes_utilities = MockARPESUtilities()


# Using a simplified setup_logger to avoid file I/O during tests
def setup_logger(name, level):
    class MockLogger:
        def info(self, message):
            pass  # Do nothing

    return MockLogger()


# Fixtures for common test data


@pytest.fixture
def sample_data_1d():
    x = np.linspace(0, 10, 100)
    data = xr.DataArray(np.sin(x), dims=["x"], coords={"x": x})
    return data


@pytest.fixture
def sample_data_2d():
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 5, 25)
    data = np.outer(np.sin(x), np.cos(y))
    return xr.DataArray(data, dims=["x", "y"], coords={"x": x, "y": y})


# Patching external modules for unit tests
@pytest.fixture(autouse=True)
def mock_arpes_modules():
    with (
        patch(
            "arpes.analysis.gaussian_filter_arr",
            new=MagicMock(side_effect=arpes_analysis.gaussian_filter_arr),
        ) as mock_gaussian_filter,
        patch(
            "arpes.analysis.savitzky_golay_filter",
            new=MagicMock(side_effect=arpes_analysis.savitzky_golay_filter),
        ) as mock_savgol_filter,
        patch(
            "arpes.analysis.boxcar_filter_arr",
            new=MagicMock(side_effect=arpes_analysis.boxcar_filter_arr),
        ) as mock_boxcar_filter,
        patch(
            "arpes.utilities.normalize_to_spectrum",
            new=MagicMock(side_effect=arpes_utilities.normalize_to_spectrum),
        ) as mock_normalize_to_spectrum,
    ):
        yield (
            mock_gaussian_filter,
            mock_savgol_filter,
            mock_boxcar_filter,
            mock_normalize_to_spectrum,
        )


@pytest.fixture(autouse=True)
def mock_holoviews_regrid():
    with patch(
        "holoviews.operation.datashader.regrid",
        new=MagicMock(side_effect=lambda x: x),
    ) as mock_regrid:
        yield mock_regrid


class TestSmoothingApp:
    def test_init_1d_data(self, sample_data_1d):
        app = SmoothingApp(sample_data_1d)
        assert isinstance(app.data, xr.DataArray)
        assert app.data.equals(sample_data_1d)
        assert app.output.equals(sample_data_1d)
        assert "None" in app.smoothing_funcs
        assert isinstance(app.smoothing_select, pn.widgets.Select)
        assert isinstance(app.output_button, pn.widgets.Button)
        assert isinstance(app.output_name, pn.widgets.TextInput)
        assert isinstance(app.output_pane, pn.pane.HoloViews)
        assert isinstance(app.widgets_panel, pn.Column)
        assert isinstance(app.layout, pn.Row)
        assert app.output_pane.object is not None  # Check if plot was updated

    def test_init_2d_data(self, sample_data_2d):
        app = SmoothingApp(sample_data_2d)
        assert isinstance(app.data, xr.DataArray)
        assert app.data.equals(sample_data_2d)
        assert app.output.equals(sample_data_2d)
        assert app.output_pane.object is not None

    @pytest.mark.skip
    def test_init_non_dataarray_input(self, mock_arpes_modules):
        # Test case where input is not xr.DataArray, triggering normalize_to_spectrum
        mock_data = [1, 2, 3]
        app = SmoothingApp(mock_data)
        mock_arpes_modules[3].assert_called_once_with(mock_data)
        assert isinstance(app.data, xr.DataArray)

    def test_init_raises_assertion_for_high_dim_data(self):
        high_dim_data = xr.DataArray([[[[1, 2], [3, 4]]]], dims=["a", "b", "c", "d"])
        with pytest.raises(AssertionError):
            SmoothingApp(high_dim_data)

    def test_smoothing_func_selection_updates_widgets(self, sample_data_1d):
        app = SmoothingApp(sample_data_1d)
        initial_widgets = app.param_widgets_box.objects

        app.smoothing_select.value = "Gaussian"
        assert len(app.param_widgets_box.objects) > len(initial_widgets)
        assert any(
            isinstance(w, pn.widgets.FloatSlider) and "Sigma" in w.name
            for w in app.param_widgets_box.objects
        )

        app.smoothing_select.value = "Boxcar"
        assert len(app.param_widgets_box.objects) > len(initial_widgets)
        assert any(
            isinstance(w, pn.widgets.FloatSlider) and "Kernel Size" in w.name
            for w in app.param_widgets_box.objects
        )

        app.smoothing_select.value = "Savitzky-Golay"
        assert len(app.param_widgets_box.objects) > len(initial_widgets)
        assert any(
            isinstance(w, pn.widgets.IntSlider) and "Window Length" in w.name
            for w in app.param_widgets_box.objects
        )
        assert any(
            isinstance(w, pn.widgets.IntSlider) and "Polyorder" in w.name
            for w in app.param_widgets_box.objects
        )

        app.smoothing_select.value = "None"
        assert len(app.param_widgets_box.objects) == 0

    def test_get_current_params(self, sample_data_1d):
        app = SmoothingApp(sample_data_1d)
        app.smoothing_select.value = "Gaussian"
        # Manually change a slider value
        for widget in app.param_widgets_box.objects:
            if "Sigma energy" in widget.name:
                widget.value = 0.5
            if "Iteration" in widget.name:
                widget.value = 2

        params = app._get_current_params()
        assert params["x"] == 0.1
        assert params["iteration"] == 2

    def test_on_apply_none_filter(self, sample_data_1d):
        app = SmoothingApp(sample_data_1d)
        initial_data = app.data.copy()
        app.smoothing_select.value = "None"
        app._on_apply(None)  # Simulate button click
        assert app.output.equals(initial_data)  # Should be unchanged

    @pytest.mark.skip
    def test_on_apply_gaussian_filter(self, sample_data_1d, mock_arpes_modules):
        app = SmoothingApp(sample_data_1d)
        app.smoothing_select.value = "Gaussian"
        # Set some values for the sliders
        for widget in app.param_widgets_box.objects:
            if "Sigma energy" in widget.name:
                widget.value = 0.2
            if "Iteration" in widget.name:
                widget.value = 3

        app._on_apply(None)
        mock_arpes_modules[0].assert_called_once()
        args, kwargs = mock_arpes_modules[0].call_args
        assert kwargs["arr"].equals(sample_data_1d)
        assert kwargs["sigma"] == {"energy": 0.2}
        assert kwargs["iteration_n"] == 3
        assert not app.output.equals(sample_data_1d)  # Should be modified

    @pytest.mark.skip
    def test_on_apply_savitzky_golay_filter(self, sample_data_1d, mock_arpes_modules):
        app = SmoothingApp(sample_data_1d)
        app.smoothing_select.value = "Savitzky-Golay"
        # Set some values for the sliders
        for widget in app.param_widgets_box.objects:
            if "Window Length energy" in widget.name:
                widget.value = 7
            if "Polyorder energy" in widget.name:
                widget.value = 3

        app._on_apply(None)
        mock_arpes_modules[1].assert_called_once()
        args, kwargs = mock_arpes_modules[1].call_args
        assert kwargs["arr"].equals(sample_data_1d)
        assert kwargs["window_length"] == {"energy": 7}
        assert kwargs["polyorder"] == {"energy": 3}
        assert not app.output.equals(sample_data_1d)

    @pytest.mark.skip
    def test_on_apply_boxcar_filter(self, sample_data_2d, mock_arpes_modules):
        app = SmoothingApp(sample_data_2d)
        app.smoothing_select.value = "Boxcar"
        # Set some values for the sliders
        for widget in app.param_widgets_box.objects:
            if "Kernel Size energy" in widget.name:
                widget.value = 0.3
            if "Kernel Size angle" in widget.name:
                widget.value = 0.4
            if "Iteration" in widget.name:
                widget.value = 2

        app._on_apply(None)
        mock_arpes_modules[2].assert_called_once()
        args, kwargs = mock_arpes_modules[2].call_args
        assert kwargs["arr"].equals(sample_data_2d)
        assert kwargs["size"] == {"energy": 0.3, "angle": 0.4}
        assert kwargs["iteration_n"] == 2
        assert not app.output.equals(sample_data_2d)

    def test_on_apply_with_output_name(self, sample_data_1d):
        app = SmoothingApp(sample_data_1d)
        app.output_name.value = "my_smoothed_data"
        app._on_apply(None)
        assert "my_smoothed_data" in app.named_output
        assert app.named_output["my_smoothed_data"].equals(app.output)

    def test_update_plot_1d(self, sample_data_1d):
        app = SmoothingApp(sample_data_1d)
        app._update_plot()
        assert isinstance(app.output_pane.object, hv.Curve)
        assert app.output_pane.object.kdims[0].name == "x"

    @pytest.mark.skip
    def test_update_plot_2d(self, sample_data_2d, mock_holoviews_regrid):
        app = SmoothingApp(sample_data_2d)
        app._update_plot()
        assert isinstance(app.output_pane.object, hv.Image)
        mock_holoviews_regrid.assert_called_once()
        assert app.output_pane.object.kdims[0].name == "angle"
        assert app.output_pane.object.kdims[1].name == "energy"

    def test_panel_method_returns_layout(self, sample_data_1d):
        app = SmoothingApp(sample_data_1d)
        assert isinstance(app.panel(), pn.layout.Panel)


@pytest.mark.skip
class TestSmoothingHelperFunctions:
    def test_iteration_slider(self):
        sliders = _iteration_slider()
        assert "iteration" in sliders
        assert isinstance(sliders["iteration"], pn.widgets.IntSlider)
        assert sliders["iteration"].name == "Iteration"
        assert sliders["iteration"].value == 1

    def test_gaussian_slider(self, sample_data_1d, sample_data_2d):
        sliders_1d = _gaussian_slider(sample_data_1d)
        assert "iteration" in sliders_1d
        assert "energy" in sliders_1d
        assert isinstance(sliders_1d["energy"], pn.widgets.FloatSlider)
        assert sliders_1d["energy"].name == "Sigma energy"

        sliders_2d = _gaussian_slider(sample_data_2d)
        assert "iteration" in sliders_2d
        assert "energy" in sliders_2d
        assert "angle" in sliders_2d
        assert sliders_2d["energy"].name == "Sigma energy"
        assert sliders_2d["angle"].name == "Sigma angle"

    def test_boxcar_slider(self, sample_data_1d, sample_data_2d):
        sliders_1d = _boxcar_slider(sample_data_1d)
        assert "iteration" in sliders_1d
        assert "energy" in sliders_1d
        assert isinstance(sliders_1d["energy"], pn.widgets.FloatSlider)
        assert sliders_1d["energy"].name == "Kernel Size energy"

        sliders_2d = _boxcar_slider(sample_data_2d)
        assert "iteration" in sliders_2d
        assert "energy" in sliders_2d
        assert "angle" in sliders_2d
        assert sliders_2d["energy"].name == "Kernel Size energy"
        assert sliders_2d["angle"].name == "Kernel Size angle"

    def test_savgol_slider(self, sample_data_1d, sample_data_2d):
        sliders_1d = _savgol_slider(sample_data_1d)
        assert "window_length_energy" in sliders_1d
        assert "polyorder_energy" in sliders_1d
        assert isinstance(sliders_1d["window_length_energy"], pn.widgets.IntSlider)
        assert sliders_1d["window_length_energy"].name == "Window Length energy"
        assert isinstance(sliders_1d["polyorder_energy"], pn.widgets.IntSlider)
        assert sliders_1d["polyorder_energy"].name == "Polyorder energy"

        sliders_2d = _savgol_slider(sample_data_2d)
        assert "window_length_energy" in sliders_2d
        assert "polyorder_energy" in sliders_2d
        assert "window_length_angle" in sliders_2d
        assert "polyorder_angle" in sliders_2d


def test_smoothing_app_construction(sample_data_2d):
    app = SmoothingApp(sample_data_2d)
    assert isinstance(app.layout, pn.layout.Panel)
    assert app.smoothing_select.name == "Smoothing Function"
    assert app.output_pane.object is not None


@pytest.mark.parametrize("method", ["Gaussian", "Boxcar"])
def test_smoothing_methods(sample_data_2d, method):
    app = SmoothingApp(sample_data_2d)
    app.smoothing_select.value = method
    for w in app.param_widgets_box.objects:
        if isinstance(w, pn.widgets.IntSlider):
            w.value = 3
        elif isinstance(w, pn.widgets.FloatSlider):
            w.value = 4
    app.output_name.value = f"smoothed_{method}"
    app._on_apply(None)
    assert app.output.shape == sample_data_2d.shape
    assert f"smoothed_{method}" in app.named_output


def test_smoothing_app_panel_output(sample_data_2d):
    app = SmoothingApp(sample_data_2d)
    panel = app.panel()
    assert isinstance(panel, pn.layout.Panel)


def test_differentiate_app_with_derivative(sample_data_1d):
    app = DifferentiateApp(sample_data_1d)
    app.smoothing_select.value = "None"
    app.derivation_select.value = "Derivative"
    for w in app.derivative_param_widgets_box.objects:
        if isinstance(w, pn.widgets.Select):
            w.value = sample_data_1d.dims[0]
        elif isinstance(w, pn.widgets.IntSlider):
            w.value = 1
    app.output_name.value = "diff_derivative"
    app._on_apply(None)
    assert app.output.shape == sample_data_1d.shape
    assert "diff_derivative" in app.named_output


@pytest.mark.parametrize(
    "method",
    ["Maximum curvature (1D)", "Maximum curvature (2D)", "Minimum Gradient"],
)
def test_differentiate_app_methods(sample_data_2d, method):
    app = DifferentiateApp(sample_data_2d)
    app.smoothing_select.value = "None"
    app.derivation_select.value = method

    for w in app.derivative_param_widgets_box.objects:
        if isinstance(w, pn.widgets.Select):
            w.value = sample_data_2d.dims[0]
        elif isinstance(w, pn.widgets.IntSlider):
            w.value = 1
        elif isinstance(w, pn.widgets.FloatSlider):
            w.value = 0.1

    app.output_name.value = f"diff_{method}"
    app._on_apply(None)
    assert app.output.shape == sample_data_2d.shape
    assert f"diff_{method}" in app.named_output


def test_slider_functions(sample_data_2d):
    for func in [
        _derivative_slider,
        _gaussian_slider,
        _boxcar_slider,
        _savgol_slider,
        _max_curvature_1d_slider,
    ]:
        sliders = func(sample_data_2d)
        assert isinstance(sliders, dict)
        assert all(isinstance(w, pn.widgets.Widget) for w in sliders.values())

    sliders_2d = _max_curvature_2d_slider()
    assert isinstance(sliders_2d, dict)
    assert all(isinstance(w, pn.widgets.Widget) for w in sliders_2d.values())


class DummySmoothingApp(SmoothingApp):
    def __init__(self):
        data = xr.DataArray(np.arange(10), dims=("x",))
        super().__init__(data)
        self.logged_messages = []

    def log_message(self, msg: str):
        self.logged_messages.append(msg)


def test_savitzky_golay_normal(monkeypatch):
    app = DummySmoothingApp()
    data = app.data

    called = {}

    def fake_savgol_filter_multi(d, axis_params):
        called["args"] = (d, axis_params)
        return d + 1

    monkeypatch.setattr(smoothing, "savgol_filter_multi", fake_savgol_filter_multi)

    result = app._savitzky_golay_smoothing(data, window_length_x=5, polyorder_x=2)
    assert isinstance(result, xr.DataArray)
    assert np.all(result == data + 1)
    assert called["args"][1] == {"x": (5, 2)}


def test_savitzky_golay_unknown_param():
    app = DummySmoothingApp()
    data = app.data
    with pytest.raises(ValueError) as e:
        app._savitzky_golay_smoothing(data, foo_x=5)
    assert "Unknown parameter" in str(e.value)


def test_savitzky_golay_even_window(monkeypatch):
    app = DummySmoothingApp()
    data = app.data

    result = app._savitzky_golay_smoothing(data, window_length_x=4, polyorder_x=2)
    assert result is data
    assert any("Window length must be odd" in m for m in app.logged_messages)


def test_savitzky_golay_polyorder_too_large(monkeypatch):
    app = DummySmoothingApp()
    data = app.data

    result = app._savitzky_golay_smoothing(data, window_length_x=3, polyorder_x=5)
    assert result is data
    assert any("Polyorder must be less than" in m for m in app.logged_messages)


def test_savitzky_golay_multiple_axes(monkeypatch):
    app = DummySmoothingApp()
    data = xr.DataArray(np.arange(9).reshape(3, 3), dims=("x", "y"))

    called = {}

    def fake_savgol_filter_multi(d, axis_params):
        called["args"] = (d, axis_params)
        return d

    monkeypatch.setattr(smoothing, "savgol_filter_multi", fake_savgol_filter_multi)

    app._savitzky_golay_smoothing(
        data, window_length_x=5, polyorder_x=2, window_length_y=3, polyorder_y=1
    )
    assert called["args"][1] == {"x": (5, 2), "y": (3, 1)}
