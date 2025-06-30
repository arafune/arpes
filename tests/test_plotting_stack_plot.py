"""This module contains unit tests for the waterfall_dispersion functions."""

from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from pandas._config.config import is_instance_factory
import pytest
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from arpes.plotting.stack_plot import _get_colors, waterfall_dispersion


@pytest.fixture
def test_data():
    eV = np.linspace(0.0, 5.0, 100)
    phi = np.linspace(-30, 30, 5)

    # 2D データ作成 (phi, eV)
    zz = np.empty((len(phi), len(eV)))
    for i, p in enumerate(phi):
        zz[i] = np.exp(-((eV - 2.5 - 0.01 * p) ** 2) / 0.1)

    return xr.DataArray(
        zz,
        coords={"phi": phi, "eV": eV},
        dims=("phi", "eV"),
        name="Photoemission Intensity",
    )


def test_get_colors():
    colors = _get_colors(plt.colormaps["viridis"], 5)
    assert len(colors) == 5
    assert isinstance(colors[0][0], np.float64)
    assert isinstance(colors[0][1], np.float64)
    assert isinstance(colors[0][2], np.float64)
    assert isinstance(colors[0][3], np.float64)


def test_basic_output_structure(test_data):
    fig, ax, ax_right = waterfall_dispersion(test_data)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    assert isinstance(ax_right, Axes)


def test_returns_two_axes_when_scale_zero(test_data):
    fig, ax = waterfall_dispersion(test_data, scale_factor=0)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_negative_scale_factor_raises(test_data):
    with pytest.raises(AssertionError, match="scale factor should be positive"):
        waterfall_dispersion(test_data, scale_factor=-1)


def test_right_axis_label(test_data):
    fig, ax, ax_right = waterfall_dispersion(test_data)
    assert ax_right.get_ylabel() == "phi"


def test_plot_modes(test_data):
    for mode in ["line", "fill_between", "hide_line"]:
        fig, ax, ax_right = waterfall_dispersion(test_data, mode=mode)
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)
        assert isinstance(ax_right, Axes)


def test_plot_with_reverse_falsel(test_data):
    fig, ax, ax_right = waterfall_dispersion(test_data, reverse=False)
    assert ax_right.yaxis.get_inverted() == np.False_


def test_custom_cmap_string(test_data):
    fig, ax, ax_right = waterfall_dispersion(test_data, cmap="viridis")
    assert isinstance(fig, Figure)


def test_custom_cmap_color_string(test_data):
    fig, ax, ax_right = waterfall_dispersion(test_data, cmap="blue")
    assert isinstance(fig, Figure)


def test_returns_correct_stack_axis(test_data):
    fig, ax, ax_right = waterfall_dispersion(test_data)
    right_label = ax_right.get_ylabel()
    assert right_label == "phi"


def test_plot_called(test_data):
    with patch("matplotlib.pyplot.Axes.plot", autospec=True) as mock_plot:
        _, ax, _ = waterfall_dispersion(test_data)
        assert mock_plot.call_count >= len(test_data.coords["phi"])


def test_fill_between_called_in_fill_mode(test_data):
    with patch("matplotlib.axes.Axes.fill_between", autospec=True) as mock_fill:
        _, ax, _ = waterfall_dispersion(test_data, mode="fill_between")
        assert mock_fill.call_count >= len(test_data.coords["phi"])


def test_fill_between_called_in_hide_lines(test_data):
    with patch("matplotlib.axes.Axes.fill_between", autospec=True) as mock_fill:
        _, ax, _ = waterfall_dispersion(test_data, mode="hide_lines")
        assert mock_fill.call_count >= len(test_data.coords["phi"])
