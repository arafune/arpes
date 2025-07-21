"""Unit test for analysis.general.py."""

import numpy as np
import pytest
import xarray as xr

from arpes.analysis.general import (
    _bin,
    condense,
    fit_fermi_edge,
    normalize_by_fermi_distribution,
    rebin,
    symmetrize_axis,
)
from arpes.constants import K_BOLTZMANN_EV_KELVIN


def test_normalize_by_fermi_distribution(dataarray_map: xr.DataArray) -> None:
    cut = dataarray_map.sum("theta", keep_attrs=True).sel(
        eV=slice(-0.2, 0.2),
        phi=slice(-0.25, 0.3),
    )
    cut_at_0 = cut.sel(phi=0, method="nearest")
    np.testing.assert_allclose(
        normalize_by_fermi_distribution(cut_at_0)[:12],
        np.array(
            [
                207347.85665894,
                207305.51187134,
                205025.95785522,
                203731.65188599,
                204150.95864868,
                205776.95861816,
                203399.02005005,
                202427.34777832,
                203042.97930908,
                203049.06216431,
                200666.57147217,
                197436.81713867,
            ],
        ),
    )


def test_normalize_by_fermi_distribution_total_broadening(dataarray_map: xr.DataArray) -> None:
    cut = dataarray_map.sum("theta", keep_attrs=True).sel(
        eV=slice(-0.2, 0.2),
        phi=slice(-0.25, 0.3),
    )
    cut_at_0 = cut.sel(phi=0, method="nearest")
    np.testing.assert_allclose(
        normalize_by_fermi_distribution(cut_at_0, total_broadening=0.03)[:12],
        np.array(
            [
                207707.21226,
                207826.179214,
                205772.205478,
                204806.277626,
                205711.498189,
                208056.485503,
                206664.307747,
                207136.756698,
                209888.552459,
                212969.879364,
                214875.002101,
                217696.089503,
            ],
        ),
    )


@pytest.fixture
def dummy_data() -> xr.DataArray:
    energy = np.linspace(-0.1, 0.1, 200)
    intensity = 1 / (np.exp(energy / (0.01 * K_BOLTZMANN_EV_KELVIN)) + 1)
    return xr.DataArray(intensity, coords=[("eV", energy)])


@pytest.mark.skip
def test_fit_fermi_edge(dummy_data: xr.DataArray) -> None:
    result = fit_fermi_edge(dummy_data)
    assert "center" in result
    assert "gamma" in result


@pytest.mark.skip
def test_symmetrize_axis():
    energy = np.linspace(-1, 1, 201)
    data = np.random.default_rng().random(201)
    da = xr.DataArray(data, coords=[("eV", energy)])
    result = symmetrize_axis(da, axis_name="eV", flip_axes=False)
    assert isinstance(result, xr.DataArray)
    assert result.shape == da.shape


def test_condense():
    data = xr.DataArray(np.random.default_rng().random((5, 10)), dims=["x", "y"])
    result = condense(data)
    assert isinstance(result, xr.DataArray)


def test_rebin():
    data = xr.DataArray(np.random.default_rng().random((10, 100)), dims=["x", "eV"])
    shape = {"eV": 50}
    result = rebin(data, shape, bin_width=None, method="mean")
    assert isinstance(result, xr.DataArray)
    assert result.sizes["eV"] == 50


def test__bin():
    data = xr.DataArray(np.arange(100), dims=["x"])
    bins = np.linspace(0, 100, 11)
    result = _bin(data, bin_axis="x", bins=bins, method="mean")
    assert isinstance(result, xr.DataArray)
    assert result.sizes["x"] == 10
