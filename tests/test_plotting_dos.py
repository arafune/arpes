"""Unit test for plotting.dos."""

from pathlib import Path

import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from arpes.plotting.dos import plot_dos


def test_plot_dos_returns_figure():
    arr = xr.DataArray(
        np.random.default_rng().random((10, 10)),
        dims=("eV", "kp"),
        coords={"eV": np.linspace(-1, 1, 10), "kp": np.linspace(-1, 1, 10)},
    )
    fig, axes = plot_dos(arr)
    assert isinstance(fig, Figure)
    assert len(axes) == 2


def test_plot_dos_saves_file(tmp_path):
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
