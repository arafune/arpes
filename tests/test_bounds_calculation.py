"""Test for bounds_calculations module."""

import numpy as np
from numpy.testing import assert_array_almost_equal

from arpes.utilities.conversion.bounds_calculations import (
    euler_to_kx,
    euler_to_ky,
    euler_to_kz,
    full_angles_to_k,
    spherical_to_kx,
    spherical_to_ky,
    spherical_to_kz,
)


def test_full_angles_to_k() -> None:
    """Test full_angles_to_k function."""
    kx, ky, kz = full_angles_to_k(
        kinetic_energy=100.0,
        phi=0.1,
        psi=0.2,
        alpha=0.3,
        beta=0.4,
        theta=0.5,
        chi=0.6,
        inner_potential=10.0,
    )
    assert_array_almost_equal(kx, np.array(-2.24792676))
    assert_array_almost_equal(ky, np.array(0.679376))
    assert_array_almost_equal(kz, np.array(4.295265))


def test_euler_to_kx() -> None:
    """Test euler_to_kx function."""
    kx = euler_to_kx(
        kinetic_energy=np.array([100.0]),
        phi=np.array([0.1]),
        beta=np.array([0.2]),
        theta=0.3,
        slit_is_vertical=False,
    )
    assert_array_almost_equal(kx, np.array(1.995055))


def test_euler_to_ky() -> None:
    """Test euler_to_ky function."""
    ky = euler_to_ky(
        kinetic_energy=np.array([100.0]),
        phi=np.array([0.1]),
        beta=np.array([0.2]),
        theta=0.3,
        slit_is_vertical=False,
    )
    assert_array_almost_equal(ky, np.array(0.937471))


def test_euler_to_kz() -> None:
    """Test euler_to_kz function."""
    kz = euler_to_kz(
        kinetic_energy=np.array([100.0]),
        phi=np.array([0.1]),
        beta=np.array([0.2]),
        theta=0.3,
        inner_potential=10.0,
        slit_is_vertical=False,
    )
    assert_array_almost_equal(kz, np.array(4.900248))


def test_spherical_to_kx() -> None:
    """Test spherical_to_kx function."""
    kx = spherical_to_kx(
        kinetic_energy=100.0,
        theta=0.1,
        phi=0.2,
    )
    assert_array_almost_equal(kx, np.array(0.501268))


def test_spherical_to_ky() -> None:
    """Test spherical_to_ky function."""
    ky = spherical_to_ky(
        kinetic_energy=100.0,
        theta=0.1,
        phi=0.2,
    )
    assert_array_almost_equal(ky, np.array(0.101612))


def test_spherical_to_kz() -> None:
    """Test spherical_to_kz function."""
    kz = spherical_to_kz(
        kinetic_energy=100.0,
        theta=0.1,
        inner_potential=10.0,
    )
    assert_array_almost_equal(kz, np.array(5.348825))
