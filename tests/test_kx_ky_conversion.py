import numpy as np
import pytest
import xarray as xr

from arpes.utilities.conversion.kx_ky_conversion import ConvertKp, ConvertKxKy


@pytest.fixture
def convert_kp(dataarray_cut: xr.DataArray) -> ConvertKp:
    return ConvertKp(dataarray_cut)


@pytest.fixture
def convert_kxky(dataarray_cut: xr.DataArray) -> ConvertKxKy:
    return ConvertKxKy(dataarray_cut)


def test_compute_k_tot_kp(convert_kp: ConvertKp) -> None:
    binding_energy = np.random.default_rng().random(10)
    convert_kp.compute_k_tot(binding_energy)
    assert convert_kp.k_tot is not None
    assert len(convert_kp.k_tot) == len(binding_energy)


@pytest.mark.skip
def test_compute_k_tot_kxky(convert_kxky: ConvertKxKy) -> None:
    binding_energy = np.random.default_rng().random(10)
    convert_kxky.compute_k_tot(binding_energy)
    assert convert_kxky.k_tot is not None
    assert len(convert_kxky.k_tot) == len(binding_energy)


def test_kspace_to_phi_kp(convert_kp: ConvertKp) -> None:
    binding_energy = np.random.default_rng().random(10)
    kp = np.random.default_rng().random(10)
    phi = convert_kp.kspace_to_phi(binding_energy, kp)
    assert phi is not None
    assert len(phi) == len(kp)


@pytest.mark.skip
def test_kspace_to_phi_kxky(convert_kxky: ConvertKxKy) -> None:
    binding_energy = np.random.default_rng().random(10)
    kx = np.random.default_rng().random(10)
    ky = np.random.default_rng().random(10)
    phi = convert_kxky.kspace_to_phi(binding_energy, kx, ky)
    assert phi is not None
    assert len(phi) == len(ky)


@pytest.mark.skip
def test_kspace_to_perp_angle(convert_kxky: ConvertKxKy) -> None:
    binding_energy = np.random.default_rng().random(10)
    kx = np.random.default_rng().random(10)
    ky = np.random.default_rng().random(10)
    perp_angle = convert_kxky.kspace_to_perp_angle(binding_energy, kx, ky)
    assert perp_angle is not None
    assert len(perp_angle) == len(kx)


def test_get_coordinates_kp(convert_kp: ConvertKp) -> None:
    coordinates = convert_kp.get_coordinates()
    assert "kp" in coordinates
    assert isinstance(coordinates["kp"], np.ndarray)


@pytest.mark.skip
def test_get_coordinates_kxky(convert_kxky: ConvertKxKy) -> None:
    coordinates = convert_kxky.get_coordinates()
    assert "kx" in coordinates
    assert "ky" in coordinates
    assert isinstance(coordinates["kx"], np.ndarray)
    assert isinstance(coordinates["ky"], np.ndarray)


@pytest.mark.skip
def test_rkx_rky(convert_kxky: ConvertKxKy) -> None:
    kx = np.random.default_rng().random(10)
    ky = np.random.default_rng().random(10)
    rkx, rky = convert_kxky.rkx_rky(kx, ky)
    assert rkx is not None
    assert rky is not None
    assert len(rkx) == len(kx)
    assert len(rky) == len(ky)


def test_conversion_for_kp(convert_kp: ConvertKp) -> None:
    func = convert_kp.conversion_for("phi")
    assert callable(func)


@pytest.mark.skip
def test_conversion_for_kxky(convert_kxky: ConvertKxKy) -> None:
    func = convert_kxky.conversion_for("phi")
    assert callable(func)
    assert len(phi) == len(kp)


@pytest.mark.skip
def test_kspace_to_phi_kxky(convert_kxky: ConvertKxKy) -> None:
    binding_energy = np.random.rand(10)
    kx = np.random.rand(10)
    ky = np.random.rand(10)
    phi = convert_kxky.kspace_to_phi(binding_energy, kx, ky)
    assert phi is not None
    assert len(phi) == len(ky)


@pytest.mark.skip
def test_kspace_to_perp_angle(convert_kxky: ConvertKxKy) -> None:
    binding_energy = np.random.rand(10)
    kx = np.random.rand(10)
    ky = np.random.rand(10)
    perp_angle = convert_kxky.kspace_to_perp_angle(binding_energy, kx, ky)
    assert perp_angle is not None
    assert len(perp_angle) == len(kx)


def test_get_coordinates_kp(convert_kp: ConvertKp) -> None:
    coordinates = convert_kp.get_coordinates()
    assert "kp" in coordinates
    assert isinstance(coordinates["kp"], np.ndarray)


@pytest.mark.skip
def test_get_coordinates_kxky(convert_kxky: ConvertKxKy) -> None:
    coordinates = convert_kxky.get_coordinates()
    assert "kx" in coordinates
    assert "ky" in coordinates
    assert isinstance(coordinates["kx"], np.ndarray)
    assert isinstance(coordinates["ky"], np.ndarray)
