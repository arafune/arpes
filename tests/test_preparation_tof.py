"""Unit test for preparation.tof module."""

import numpy as np
import pytest
import xarray as xr

from arpes.preparation.tof import (
    build_KE_coords_to_time_coords,
    build_KE_coords_to_time_pixel_coords,
    convert_to_kinetic_energy,
    process_DLD,
    process_SToF,
)


@pytest.fixture
def simple_tof_dataarray():
    time = np.linspace(1.0, 5.0, 10)
    angle = np.linspace(-1.0, 1.0, 3)

    data = np.ones((len(time), len(angle)))

    return xr.DataArray(
        data,
        coords={"time": time, "angle": angle},
        dims=("time", "angle"),
        name="t_test",
        attrs={
            "mstar": 1.0,
            "length": 1.0,
        },
    )


@pytest.fixture
def simple_dataset(simple_tof_dataarray):
    ds = xr.Dataset(
        {
            "t_up": simple_tof_dataarray,
            "t_down": simple_tof_dataarray * 0.8,
        }
    )
    ds = ds.assign_coords(time=simple_tof_dataarray.coords["time"])
    ds.attrs.update(
        {
            "mstar": 1.0,
            "length": 1.0,
            "laser_t0": 0.0,
            "timing_offset": 0.0,
            "E_max": 10.0,
            "dE": 0.5,
            "sherman": 0.2,
        }
    )
    return ds


def test_convert_to_kinetic_energy_basic(simple_tof_dataarray):
    ke_axis = np.linspace(0.5, 5.0, 20)

    out = convert_to_kinetic_energy(simple_tof_dataarray, ke_axis)

    assert isinstance(out, xr.DataArray)
    assert out.dims[0] == "eV"
    assert "eV" in out.coords
    assert out.shape[0] == len(ke_axis)
    assert out.attrs["mstar"] == 1.0


def test_convert_to_kinetic_energy_preserves_other_dims(simple_tof_dataarray):
    ke_axis = np.linspace(0.5, 5.0, 10)
    out = convert_to_kinetic_energy(simple_tof_dataarray, ke_axis)

    assert out.dims == ("eV", "angle")


#  build_KE_coords_to_time_pixel_coords(


def test_build_KE_coords_to_time_pixel_coords_callable(simple_dataset):
    ke_axis = np.linspace(0.5, 5.0, 10)
    func = build_KE_coords_to_time_pixel_coords(simple_dataset, ke_axis)

    assert callable(func)


def test_KE_coords_to_time_pixel_coords_execution(simple_dataset):
    ke_axis = np.linspace(0.5, 5.0, 10)
    func = build_KE_coords_to_time_pixel_coords(simple_dataset, ke_axis)

    # fake coordinate tuple (like ndimage gives)
    coord_energy = xr.DataArray(np.array([0, 1, 2]), dims=("points",))
    coord_other = xr.DataArray(np.array([5, 6, 7]), dims=("points",))

    coords = (coord_energy, coord_other)

    out = func(coords, axis=0)

    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[0], np.ndarray)
    assert out[0].shape == coord_energy.shape


# build_KE_coords_to_time_coords


def test_build_KE_coords_to_time_coords_callable(simple_dataset):
    ke_axis = np.linspace(0.5, 5.0, 10)
    func = build_KE_coords_to_time_coords(simple_dataset, ke_axis)

    assert callable(func)


def test_KE_coords_to_time_coords_execution(simple_dataset):
    ke_axis = np.linspace(0.5, 5.0, 10)
    func = build_KE_coords_to_time_coords(simple_dataset, ke_axis)

    coords = xr.DataArray(
        np.arange(5),
        dims=("eV",),
    )

    out = func(coords, axis=0)

    assert isinstance(out, tuple)
    assert isinstance(out[0], xr.DataArray)


# process_SToF


def test_process_SToF_creates_energy_axis(simple_dataset):
    out = process_SToF(simple_dataset)

    assert isinstance(out, xr.Dataset)
    assert "eV" in out.dims


def test_process_SToF_renames_channels(simple_dataset):
    out = process_SToF(simple_dataset)

    assert "up" in out.data_vars
    assert "down" in out.data_vars


def test_process_SToF_spin_correction_applied(simple_dataset):
    out = process_SToF(simple_dataset)

    assert not np.allclose(out["up"], out["down"])


def test_process_SToF_spin_correction_identity_when_sherman_zero(simple_dataset):
    simple_dataset = simple_dataset.copy()
    simple_dataset.attrs["sherman"] = 0.0

    with pytest.raises(ZeroDivisionError):
        process_SToF(simple_dataset)


# process_DLD


def test_process_DLD_runs(simple_dataset):
    out = process_DLD(simple_dataset)

    assert isinstance(out, xr.Dataset)
