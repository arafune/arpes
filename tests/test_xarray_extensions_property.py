"""Unit test for xarray_extensions/accessor/property.py."""

import pytest
import xarray as xr

import arpes.xarray_extensions  # noqa: F401
from arpes.xarray_extensions.accessor.spectrum_type import SpectrumType

# --- Unittest for DataArray ---


def test_dataarray_spectrum_type_enum():
    da = xr.DataArray([1, 2, 3], dims=("x",))
    da.attrs["spectrum_type"] = SpectrumType.CUT
    assert da.S.spectrum_type == SpectrumType.CUT


def test_dataarray_spectrum_type_str():
    da = xr.DataArray([1, 2, 3], dims=("x",))
    da.attrs["spectrum_type"] = "map"
    assert da.S.spectrum_type == SpectrumType.MAP


def test_dataarray_spectrum_type_invalid_str():
    da = xr.DataArray([1, 2, 3], dims=("x",))
    da.attrs["spectrum_type"] = "invalid"
    with pytest.raises(TypeError):
        _ = da.S.spectrum_type


# ---Unit test for  Dataset ---


def test_dataset_spectrum_type_enum():
    ds = xr.Dataset({"a": ("x", [1, 2, 3])})
    ds.attrs["spectrum_type"] = SpectrumType.HV_MAP
    assert ds.spectrum_type is SpectrumType.HV_MAP


def test_dataset_spectrum_type_invalid_str():
    ds = xr.Dataset({"a": ("x", [1, 2, 3])})
    ds.attrs["spectrum_type"] = "invalid"
    with pytest.raises(TypeError):
        _ = ds.S.spectrum_type


#  --- Unit test for invalid energy notation
def test_energy_notation_invalid():
    da = xr.DataArray([1, 2, 3], dims=("x",))
    da.attrs["energy_notation"] = "invalid"
    with pytest.raises(ValueError, match="Invalid energy notation found: 'invalid'"):
        _ = da.S.energy_notation
