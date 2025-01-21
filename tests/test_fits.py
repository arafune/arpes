"""Unit test for plot_fit function in plotting/plot_fit."""

from unittest.mock import MagicMock

import lmfit as lf
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from arpes.plotting.fits import plot_fit, plot_fits


@pytest.fixture
def mock_model_results() -> list[lf.model.ModelResult]:
    results = []
    for _ in range(4):
        mock_result = MagicMock()
        x = np.linspace(0, 10, 100)
        data = np.sin(x)
        best_fit = np.sin(x)
        residual = data - best_fit

        mock_result.userkws = {"x": x}
        mock_result.model.independent_vars = ["x"]
        mock_result.data = data
        mock_result.best_fit = best_fit
        mock_result.residual = residual

        results.append(mock_result)
    return results


@pytest.fixture
def mock_model_result():
    mock_result = MagicMock()
    x = np.linspace(0, 10, 100)
    data = np.sin(x)
    best_fit = np.sin(x)
    residual = data - best_fit
    mock_result.userkws = {"x": x}
    mock_result.model.independent_vars = ["x"]
    mock_result.data = data
    mock_result.best_fit = best_fit
    mock_result.residual = residual

    return mock_result


def test_plot_fit_creates_axes(mock_model_result: list) -> None:
    ax = plot_fit(mock_model_result)
    assert isinstance(ax, Axes)


def test_plot_fit_uses_provided_axes(mock_model_result: list) -> None:
    fig, ax = plt.subplots()
    result_ax = plot_fit(mock_model_result, ax)
    assert result_ax is ax


def test_plot_fits_creates_axes(mock_model_results: list):
    plot_fits(mock_model_results)  # axs is None, simple_ax_grid should be called
    plt.draw()  # Ensure all figures are drawn for testing purposes
    fig = plt.gcf()
    assert len(fig.axes) == len(mock_model_results) * 2
