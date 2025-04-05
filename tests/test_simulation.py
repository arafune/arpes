"""unit test for simulation.py."""

import pytest
import numpy as np
import xarray as xr
from numpy.testing import assert_array_almost_equal

from arpes.simulation import SpectralFunction
from arpes.constants import K_BOLTZMANN_MEV_KELVIN


@pytest.fixture
def spectral_function():
    return SpectralFunction()


def test_initialization_defaults(spectral_function: SpectralFunction):
    assert isinstance(spectral_function.k, np.ndarray)
    assert isinstance(spectral_function.omega, np.ndarray)


def test_fermi_dirac(spectral_function: SpectralFunction):
    """Test the Fermi-Dirac distribution calculation."""
    omegas = np.array([-0.1, 0.0, 0.1])
    fd = spectral_function.fermi_dirac(omegas)
    expected_values = 1 / (
        np.exp(omegas / (K_BOLTZMANN_MEV_KELVIN * spectral_function.temperature)) + 1
    )
    assert_array_almost_equal(fd, expected_values)


def test_self_energy(spectral_function):
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
