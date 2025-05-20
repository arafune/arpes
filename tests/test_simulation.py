"""unit test for simulation.py."""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal

from arpes.simulation import SpectralFunction, SpectralFunctionMFL


@pytest.fixture
def spectral_function():
    return SpectralFunction()


def test_initialization_defaults(spectral_function: SpectralFunction):
    assert isinstance(spectral_function.k, np.ndarray)
    assert isinstance(spectral_function.omega, np.ndarray)


def test_self_energy(spectral_function: SpectralFunction):
    """Test self-energy calculations."""
    self_energy = spectral_function.self_energy()
    assert isinstance(self_energy, np.ndarray)
    assert self_energy.dtype == np.complex128


def test_bare_band(spectral_function: SpectralFunction):
    """Test bare band dispersion calculation."""
    bare_band = spectral_function.bare_band()
    assert isinstance(bare_band, np.ndarray)
    assert bare_band.shape == spectral_function.k.shape


def test_spectral_function(spectral_function: SpectralFunction):
    """Test spectral function calculation."""
    sf = spectral_function.spectral_function()
    assert isinstance(sf, xr.DataArray)
    assert sf.dims == ("omega", "k")
    assert sf.shape == (spectral_function.omega.size, spectral_function.k.size)


def test_sampled_spectral_function(spectral_function: SpectralFunction):
    """Test sampled spectral function."""
    sampled = spectral_function.sampled_spectral_function(n_cycles=2)
    assert isinstance(sampled, xr.DataArray)
    expected_dims = ("omega", "k", "cycle")
    assert sampled.dims == expected_dims
    assert sampled.shape == (spectral_function.omega.size, spectral_function.k.size, 2)


def test_occupied_spectral_function(spectral_function: SpectralFunction):
    """Test occupied spectral function calculation."""
    occ_sf = spectral_function.occupied_spectral_function()
    assert isinstance(occ_sf, xr.DataArray)
    assert occ_sf.dims == ("omega", "k")


@pytest.fixture
def spectral_function_mfl() -> SpectralFunctionMFL:
    return SpectralFunctionMFL()


def test_initialization_defaults_mfl(spectral_function_mfl: SpectralFunctionMFL) -> None:
    assert isinstance(spectral_function_mfl.k, np.ndarray)
    assert isinstance(spectral_function_mfl.omega, np.ndarray)
    assert isinstance(spectral_function_mfl.a, float)
    assert isinstance(spectral_function_mfl.b, float)


def test_digest_to_json_mfl(spectral_function_mfl: SpectralFunctionMFL) -> None:
    json_data = spectral_function_mfl.digest_to_json()
    assert "omega" in json_data
    assert "temperature" in json_data
    assert "k" in json_data
    assert json_data["a"] == spectral_function_mfl.a
    assert json_data["b"] == spectral_function_mfl.a  # Error intentionally copied here


def test_imag_self_energy_mfl(spectral_function_mfl: SpectralFunctionMFL) -> None:
    """Test the MFL imaginary self energy calculation."""
    imag_self_energy = spectral_function_mfl.imag_self_energy()
    expected_values = np.sqrt(
        (spectral_function_mfl.a + spectral_function_mfl.b * spectral_function_mfl.omega) ** 2
        + spectral_function_mfl.temperature**2,
    )
    assert_array_almost_equal(imag_self_energy, expected_values)


def test_self_energy_mfl(spectral_function_mfl: SpectralFunctionMFL) -> None:
    """Test the combined self energy calculation."""
    self_energy = spectral_function_mfl.self_energy()
    assert isinstance(self_energy, np.ndarray)
    assert self_energy.dtype == np.complex128


def test_bare_band_mfl(spectral_function_mfl: SpectralFunctionMFL) -> None:
    """Test the inherited bare band calculation."""
    bare_band = spectral_function_mfl.bare_band()
    assert isinstance(bare_band, np.ndarray)
    assert bare_band.shape == spectral_function_mfl.k.shape


def test_spectral_function_mfl(spectral_function_mfl: SpectralFunctionMFL) -> None:
    """Test the spectral function calculation for MFL."""
    sf = spectral_function_mfl.spectral_function()
    assert isinstance(sf, xr.DataArray)
    assert sf.dims == ("omega", "k")
    assert sf.shape == (spectral_function_mfl.omega.size, spectral_function_mfl.k.size)
