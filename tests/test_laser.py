"""Test for laser.py."""

import numpy as np

from arpes.laser import wavelength_to_energy


def test_wavelength_to_energy():
    np.testing.assert_allclose(1.239841974e3, wavelength_to_energy(1), rtol=1e-4)
    np.testing.assert_allclose(1.5498024804150035, wavelength_to_energy(800), rtol=1e-4)
