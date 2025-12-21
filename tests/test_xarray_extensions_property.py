"""Unit test for xarray_extensions/accessor/property.py."""

import numpy as np
import pytest
import xarray as xr

import arpes.xarray_extensions  # noqa: F401
from arpes.xarray_extensions.accessor.spectrum_type import AngleUnit, SpectrumType

# --- Unittest for DataArray ---


def test_dataarray_spectrum_type_enum() -> None:
    da = xr.DataArray([1, 2, 3], dims=("x",))
    da.attrs["spectrum_type"] = SpectrumType.CUT
    assert da.S.spectrum_type == SpectrumType.CUT


def test_dataarray_spectrum_type_str() -> None:
    da = xr.DataArray([1, 2, 3], dims=("x",))
    da.attrs["spectrum_type"] = "map"
    assert da.S.spectrum_type == SpectrumType.MAP


def test_dataarray_spectrum_type_invalid_str() -> None:
    da = xr.DataArray([1, 2, 3], dims=("x",))
    da.attrs["spectrum_type"] = "invalid"
    with pytest.raises(TypeError):
        _ = da.S.spectrum_type


# ---Unit test for  Dataset ---


def test_dataset_spectrum_type_enum() -> None:
    ds = xr.Dataset({"a": ("x", [1, 2, 3])})
    ds.attrs["spectrum_type"] = SpectrumType.HV_MAP
    assert ds.spectrum_type is SpectrumType.HV_MAP


def test_dataset_spectrum_type_invalid_str() -> None:
    ds = xr.Dataset({"a": ("x", [1, 2, 3])})
    ds.attrs["spectrum_type"] = "invalid"
    with pytest.raises(TypeError):
        _ = ds.S.spectrum_type


#  --- Unit test for invalid energy notation
def test_energy_notation_invalid() -> None:
    da = xr.DataArray([1, 2, 3], dims=("x",))
    da.attrs["energy_notation"] = "invalid"
    with pytest.raises(ValueError, match="Invalid energy notation found: 'invalid'"):
        _ = da.S.energy_notation


def test_lookup_coord_for_invalid_coord(dataarray_cut: xr.DataArray) -> None:
    np.testing.assert_allclose(dataarray_cut.S.lookup_coord("invalid_coord"), np.nan)


def test_psi_chi_offset(dataarray_cut: xr.DataArray) -> None:
    np.testing.assert_allclose(dataarray_cut.S.psi_offset, 0)
    np.testing.assert_allclose(dataarray_cut.S.chi_offset, dataarray_cut.S.lookup_offset("chi"))


class TestAngleUnitforDataArray:
    """Test class for angle_unit for DataArray."""

    def test_angle_unit(self, dataarray_cut: xr.DataArray) -> None:
        """Test for angle unit property for DataArray."""
        assert dataarray_cut.S.angle_unit is AngleUnit.RAD

    def test_setter_of_angle_unit(self, dataarray_cut: xr.DataArray) -> None:
        """Test for angle_unit setter."""
        dataarray_cut.S.angle_unit = AngleUnit.DEG
        assert dataarray_cut.S.angle_unit.value == "Degrees"
        assert dataarray_cut.attrs["angle_unit"] == "Degrees"

    def test_switched_angle_unit(self, dataarray_cut: xr.DataArray) -> None:
        """Test for switched_angle_unit."""
        converted_data = dataarray_cut.S.switched_angle_unit()
        assert converted_data.S.angle_unit is AngleUnit.DEG
        np.testing.assert_allclose(
            converted_data.coords["phi"].values[0:6],
            [12.7, 12.8, 12.9, 13.0, 13.1, 13.2],
        )
        reverted_data = converted_data.S.switched_angle_unit()
        assert reverted_data.S.angle_unit is AngleUnit.RAD

    def test_switch_angle_unit(self, dataarray_cut: xr.DataArray) -> None:
        """Test for switch_angle_unit (DataArray version)."""
        original_phi_coords = dataarray_cut.coords["phi"].values
        # rad -> deg
        dataarray_cut.S.switch_angle_unit()
        phi_coords = dataarray_cut.coords["phi"].values
        np.testing.assert_allclose(phi_coords[0:6], [12.7, 12.8, 12.9, 13.0, 13.1, 13.2])
        assert (
            dataarray_cut.coords["chi"]
            == dataarray_cut.attrs["chi_offset"]
            == np.rad2deg(-0.10909301748228785)
        )
        assert dataarray_cut.attrs["chi_offset"] == np.rad2deg(-0.10909301748228785)
        assert dataarray_cut.S.angle_unit is AngleUnit.DEG
        # deg -> rad
        dataarray_cut.S.switch_angle_unit()

        np.testing.assert_allclose(
            dataarray_cut.coords["phi"].values[0:6],
            original_phi_coords[0:6],
        )
        assert dataarray_cut.S.angle_unit is AngleUnit.RAD

    def test_for_is_slit_vertical(self, dataarray_cut: xr.DataArray) -> None:
        """Test for is_slit_vertical (DataArray version)."""
        assert dataarray_cut.S.is_slit_vertical is False
        dataarray_cut.coords["alpha"] = np.pi / 2
        assert dataarray_cut.S.is_slit_vertical is True
        dataarray_cut.S.switch_angle_unit()
        assert dataarray_cut.S.is_slit_vertical is True


class TestAngleUnitForDataset:
    """Test class for angle_unit for DataSet."""

    def test_angle_unit(self, dataset_cut: xr.Dataset) -> None:
        """Test for angle unit property for Dataset."""
        assert dataset_cut.S.angle_unit is AngleUnit.RAD

    def test_setter_of_angle_unit(self, dataset_cut: xr.Dataset) -> None:
        """Test for angle_unit setter. (Dataset)."""
        dataset_cut.S.angle_unit = AngleUnit.DEG
        assert dataset_cut.S.angle_unit.value == "Degrees" == dataset_cut.attrs["angle_unit"]
        for spectrum in dataset_cut.S.spectra:
            assert spectrum.S.angle_unit is AngleUnit.DEG

    def test_switched_angle_unit(self, dataset_cut: xr.Dataset) -> None:
        """Test for switched_angle_unit."""
        converted_data = dataset_cut.S.switched_angle_unit()
        assert converted_data.S.angle_unit is AngleUnit.DEG
        np.testing.assert_allclose(
            converted_data.coords["phi"].values[0:6],
            [12.7, 12.8, 12.9, 13.0, 13.1, 13.2],
        )
        reverted_data = converted_data.S.switched_angle_unit()
        assert reverted_data.S.angle_unit is AngleUnit.RAD

    def test_switch_angle_unit_raise_type_error(self, dataarray_cut: xr.DataArray) -> None:
        dataarray_cut.attrs["angle_unit"] = "mil"
        with pytest.raises(ValueError, match="Invalid angle unit found: 'mil'"):
            dataarray_cut.S.switch_angle_unit()

    def test_switched_angle_unit_for_dataset(self, dataset_cut: xr.Dataset) -> None:
        converted_data = dataset_cut.S.switched_angle_unit()
        assert converted_data.S.angle_unit is AngleUnit.DEG
        np.testing.assert_allclose(
            converted_data.coords["phi"].values[0:6],
            [12.7, 12.8, 12.9, 13.0, 13.1, 13.2],
        )
        np.testing.assert_allclose(
            converted_data.S.spectrum.coords["phi"].values[0:6],
            [12.7, 12.8, 12.9, 13.0, 13.1, 13.2],
        )

    def test_switch_angle_untit_for_dataset(self, dataset_cut: xr.Dataset) -> None:
        dataset_cut.S.switch_angle_unit()
        assert dataset_cut.S.angle_unit is AngleUnit.DEG
        np.testing.assert_allclose(
            dataset_cut.coords["phi"].values[0:6],
            [12.7, 12.8, 12.9, 13.0, 13.1, 13.2],
        )
        np.testing.assert_allclose(
            dataset_cut.S.spectrum.coords["phi"].values[0:6],
            [12.7, 12.8, 12.9, 13.0, 13.1, 13.2],
        )

    def test_switch_angle_unit(self, dataset_cut: xr.Dataset) -> None:
        """Test for switch_angle_unit (Dataset version)."""
        # rad -> deg
        dataset_cut.S.switch_angle_unit()
        phi_coords = dataset_cut.coords["phi"].values
        np.testing.assert_allclose(phi_coords[0:6], [12.7, 12.8, 12.9, 13.0, 13.1, 13.2])
        assert (
            dataset_cut.coords["chi"]
            == dataset_cut.attrs["chi_offset"]
            == np.rad2deg(-0.10909301748228785)
        )
        assert dataset_cut.attrs["chi_offset"] == np.rad2deg(-0.10909301748228785)
        assert dataset_cut.S.angle_unit is AngleUnit.DEG

    def test_offset_from_symmetry_point(self, dataset_cut: xr.Dataset) -> None:
        dataset_cut.attrs["symmetry_points"] = {"G": {"phi": dataset_cut.S.phi_offset + 0.1}}
        np.testing.assert_allclose(dataset_cut.S.lookup_offset("phi"), 0.505)

        np.testing.assert_allclose(dataset_cut.S.lookup_offset("chi"), dataset_cut.S.chi_offset)

    def test_for_is_slit_vertical(self, dataset_cut: xr.Dataset) -> None:
        """Test for is_slit_vertical (Dataset version)."""
        assert dataset_cut.S.is_slit_vertical is False
        dataset_cut.coords["alpha"] = np.pi / 2
        for spectrum in dataset_cut.S.spectra:
            spectrum.coords["alpha"] = np.pi / 2
        assert dataset_cut.S.is_slit_vertical is True
        dataset_cut.S.switch_angle_unit()
        assert dataset_cut.S.is_slit_vertical is True
