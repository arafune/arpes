from unittest.mock import MagicMock, patch

import numpy as np
import panel as pn
import pytest
import xarray as xr
from holoviews.element import QuadMesh

from arpes.plotting.ui.combine import TailorApp

pn.extension()


@pytest.fixture
def sample_data():
    x = np.linspace(0, 10, 11)
    x2 = np.linspace(5, 15, 11)
    y = np.linspace(0, 5, 6)
    z1 = np.random.rand(6, 11)
    z2 = np.random.rand(6, 11)
    z3 = np.random.rand(6, 11)
    data_a = xr.DataArray(z1, coords={"eV": y, "phi": x}, dims=("eV", "phi"), attrs={"id": 1})
    data_c = xr.DataArray(z3, coords={"eV": y, "phi": x2}, dims=("eV", "phi"), attrs={"id": 3})
    data_b = xr.DataArray(z2, coords={"eV": y, "phi": x2}, dims=("eV", "phi"), attrs={"id": 2})
    return data_a, data_b, data_c


@pytest.fixture
def app(sample_data):
    data_a, data_b, data_c = sample_data
    return TailorApp(data_a, data_b, pane_kwargs={"log": False, "cmap": "viridis"})


def test_init_and_build(app):
    assert isinstance(app.output_button, pn.widgets.Button)
    assert isinstance(app.output_name, pn.widgets.TextInput)
    assert isinstance(app.ratio_slider, pn.widgets.FloatSlider)
    assert app.layout is not None


def test_on_apply(app):
    app.output = app.data.copy()
    app.output_name.value = "test_output"
    app._on_apply(None)
    assert "test_output" in app.named_output
    assert "Output stored: 'test_output'" in app.message_pane.object


def test_toggle_ratio_slider(app):
    app.ratio_slider.disabled = False
    event = type("Event", (), {"new": True})
    app._toggle_ratio_slider(event)
    assert app.ratio_slider.disabled is True


@patch("arpes.plotting.ui.combine.concat_along_phi")
@patch("arpes.plotting.ui.combine.get_image_options", return_value={})
def test_update_plot_normal(mock_opts, mock_concat, app):
    dummy = app.data.copy()
    mock_concat.return_value = dummy
    app.toggle_laminate_mode.value = False
    app._update_plot()
    assert isinstance(app.output_pane.object, QuadMesh)


@patch("arpes.plotting.ui.combine.concat_along_phi")
@patch("arpes.plotting.ui.combine.get_image_options", return_value={})
def test_update_plot_laminate(mock_opts, mock_concat, app):
    dummy = app.data.copy()
    mock_concat.return_value = dummy
    app.toggle_laminate_mode.value = True
    app._update_plot()
    assert isinstance(app.output_pane.object, QuadMesh)


def test_on_slider_change_triggers_update(app):
    app._update_plot = MagicMock()
    app._on_slider_change(None)
    app._update_plot.assert_called_once()


@pytest.mark.skip  # ptytest-mock is required.
def test_on_slider_change_triggers_update0(app, mocker):
    spy = mocker.spy(app, "_update_plot")
    app._on_slider_change(None)
    spy.assert_called_once()
