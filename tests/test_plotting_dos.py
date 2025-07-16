"""Unit test for plotting.dos."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from arpes.plotting.dos import plot_core_levels, plot_dos
import arpes.xarray_extensions


def test_plot_dos_returns_figure():
    arr = xr.DataArray(
        np.random.default_rng().random((10, 10)),
        dims=("eV", "kp"),
        coords={"eV": np.linspace(-1, 1, 10), "kp": np.linspace(-1, 1, 10)},
    )
    fig, axes = plot_dos(arr)
    assert isinstance(fig, Figure)
    assert len(axes) == 2


def test_plot_dos_saves_file(tmp_path: Path):
    arr = xr.DataArray(
        np.random.default_rng().random((10, 10)),
        dims=("eV", "k"),
        coords={"eV": np.linspace(-1, 1, 10), "k": np.linspace(-1, 1, 10)},
    )
    out_file = tmp_path / "test_plot.png"
    result = plot_dos(arr, out=str(out_file))
    assert isinstance(result, Path)
    assert result.exists()


def test_plot_dos_orientation():
    arr = xr.DataArray(
        np.random.default_rng().random((10, 10)),
        dims=("eV", "k"),
        coords={"eV": np.linspace(-1, 1, 10), "k": np.linspace(-1, 1, 10)},
    )
    fig_h, _ = plot_dos(arr, orientation="horizontal")
    fig_v, _ = plot_dos(arr, orientation="vertical")
    assert isinstance(fig_h, Figure)
    assert isinstance(fig_v, Figure)


def test_plot_with_provided_core_levels(xps_map: xr.Dataset):
    fig, ax = plt.subplots()
    xps_spectrum = xps_map.spectrum.sum(["x", "y"], keep_attrs=True)
    with patch.object(ax, "axvline") as mock_axvline:
        result = plot_core_levels(
            data=xps_spectrum,
            core_levels=[-32.0, -40.0, -33],
            ax=ax,
            out="",
            alpha=0.6,
        )
        assert isinstance(result, tuple)
        assert result[0] is None
        assert isinstance(result[1], Axes)
        assert mock_axvline.call_count == 3


def test_plot_without_ax(xps_map: xr.Dataset):
    xps_spectrum = xps_map.spectrum.sum(["x", "y"], keep_attrs=True)
    with (
        patch("arpes.plotting.dos.approximate_core_levels", return_value=[-32.0]),
        patch("matplotlib.pyplot.subplots") as mock_subplots,
    ):
        mock_ax = MagicMock(spec=Axes)
        mock_fig = MagicMock(spec=Figure)
        mock_subplots.return_value = (mock_fig, mock_ax)

        result = plot_core_levels(data=xps_spectrum, ax=None)
        mock_ax.axvline.assert_called_once_with(-32.0, ymin=0.1, ymax=0.25, color="red", ls="-")
        assert result == (mock_fig, mock_ax)


def test_plot_core_levels_with_file_output(xps_map: xr.Dataset):
    xps_spectrum = xps_map.spectrum.sum(["x", "y"], keep_attrs=True)
    with (
        patch("arpes.plotting.dos.approximate_core_levels", return_value=[42.0]),
        patch("arpes.plotting.dos.savefig") as mock_savefig,
        patch(
            "arpes.plotting.dos.path_for_plot",
            return_value=Path("test_output.png"),
        ) as mock_path,
    ):
        result = plot_core_levels(data=xps_spectrum, out="test_output.png", ax=None)

        mock_savefig.assert_called_once_with("test_output.png", dpi=400)
        mock_path.assert_called_once_with("test_output.png")
        assert isinstance(result, Path)
