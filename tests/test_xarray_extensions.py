"""Unit test for xarray_extensions.py."""

import numpy as np
import pytest
import xarray as xr

from arpes.io import example_data


@pytest.fixture()
def dataset_cut() -> xr.Dataset:
    """A fixture for loading Dataset."""
    return example_data.cut


@pytest.fixture()
def dataarray_cut() -> xr.DataArray:
    """A fixture for loading DataArray."""
    return example_data.cut.spectrum


@pytest.fixture()
def xps_map() -> xr.Dataset:
    """A fixture for loading example_data.xps."""
    return example_data.nano_xps


@pytest.fixture()
def hv_map() -> xr.Dataset:
    """A fixture for loading photonenergy dependence."""
    return example_data.photon_energy


class TestforProperties:
    """Test class for Array Dataset properties."""

    def test_hv_for_hv_map(self, hv_map: xr.Dataset) -> None:
        """Test for self.hv."""
        np.testing.assert_equal(hv_map.S.hv.values, np.linspace(50, 90, 21))

    def test_degrees_of_freedom_dims(self, xps_map: xr.Dataset) -> None:
        """Test for degrees_of_freedom."""
        assert xps_map.S.spectrum_degrees_of_freedom == {"eV"}
        assert xps_map.S.scan_degrees_of_freedom == {"x", "y"}

    def test_is_functions(self, xps_map: xr.Dataset) -> None:
        assert xps_map.S.is_spatial

    def test_find_spectrum_energy_edges(self, dataarray_cut: xr.DataArray) -> None:
        """Test for find_spectrum_energy_edges."""
        np.testing.assert_array_almost_equal(
            np.array([-0.3883721, -0.14883726, 0.00465109]),
            dataarray_cut.S.find_spectrum_energy_edges(),
        )
        np.testing.assert_array_equal(
            np.array([16, 119, 185]),
            dataarray_cut.S.find_spectrum_energy_edges(indices=True),
        )

    def test_transpose_front_back(self, dataarray_cut: xr.DataArray) -> None:
        """Test for transpose_to_front/back."""
        original_ndarray = dataarray_cut.values
        transpose_to_front_ndarray = dataarray_cut.S.transpose_to_front("eV").values
        transpose_to_back_ndarray = (
            dataarray_cut.S.transpose_to_front("eV").S.transpose_to_back("eV").values
        )
        np.testing.assert_array_equal(original_ndarray, transpose_to_front_ndarray.T)
        np.testing.assert_array_equal(original_ndarray, transpose_to_back_ndarray)

    def test_spectrometer_property(self, dataarray_cut: xr.DataArray) -> None:
        """Test for spectrometer property."""
        assert dataarray_cut.S.spectrometer == {
            "name": "MC",
            "rad_per_pixel": (1 / 10) * (np.pi / 180),
            "type": "hemisphere",
            "is_slit_vertical": False,
        }

    def test_property_for_degrees_of_freedom(
        self,
        dataset_cut: xr.Dataset,
    ) -> None:
        """Test for spectrum degrees of freedom."""
        assert dataset_cut.S.spectrum_degrees_of_freedom == {"phi", "eV"}

    def test_property_for_sample_pos(
        self,
        dataset_cut: xr.Dataset,
        dataarray_cut: xr.DataArray,
    ) -> None:
        """Test for spectrum degrees of freedom."""
        assert (
            dataarray_cut.S.sample_pos
            == dataset_cut.S.sample_pos
            == (-0.7704345, 34.74984, -3.400000000001e-05)
        )

    def test_property_spatial(self, dataarray_cut: xr.DataArray) -> None:
        """Test for spatial."""
        assert dataarray_cut.S.is_spatial is False

    def test_property_dshape(self, dataarray_cut: xr.DataArray, dataset_cut: xr.Dataset) -> None:
        """Test property for dshape."""
        assert dataset_cut.S.dshape == {"phi": 240, "eV": 240}
        assert dataarray_cut.S.dshape == {"phi": 240, "eV": 240}

    def test_property_sample_angles(self, dataarray_cut: xr.Dataset) -> None:
        """Test for sample_angles."""
        assert dataarray_cut.S.sample_angles[0] == 0
        assert dataarray_cut.S.sample_angles[1] == 0
        assert dataarray_cut.S.sample_angles[2] == -0.10909301748228785  # noqa: PLR2004
        np.testing.assert_almost_equal(
            dataarray_cut.S.sample_angles[3][0:3].values,
            np.array([0.2216568, 0.2234021, 0.2251475]),
        )
        assert dataarray_cut.S.sample_angles[4] == 0
        assert dataarray_cut.S.sample_angles[5] == 0

    def test_is_subtracted(self, dataarray_cut: xr.DataArray) -> None:
        """Test property is_subtracted."""
        assert dataarray_cut.S.is_subtracted is False

    def test_property_is_kspace(self, dataset_cut: xr.Dataset) -> None:
        """Test property is_kspace."""
        assert dataset_cut.S.is_kspace is False

    def test_spectrometer_setting(self, dataset_cut: xr.Dataset) -> None:
        """Test property for spectrometer_settings."""
        assert dataset_cut.S.spectrometer_settings == {}

    def test_beamline_settings_reference_settings(self, dataset_cut: xr.Dataset) -> None:
        """Test for beamline settings."""
        assert dataset_cut.S.beamline_settings == dataset_cut.S.reference_settings == {"hv": 5.93}

    def test_full_coords(self, dataset_cut: xr.Dataset) -> None:
        """Test for full coords."""
        assert list(dataset_cut.S.full_coords.keys()) == [
            "x",
            "y",
            "z",
            "beta",
            "theta",
            "chi",
            "phi",
            "psi",
            "alpha",
            "hv",
            "eV",
        ]

    def test_G_stride_and_range(self, dataarray_cut: xr.DataArray) -> None:
        """Test for G.range and G.stride."""
        generic_stride = dataarray_cut.G.stride(generic_dim_names=True)
        assert "x" in generic_stride
        assert "y" in generic_stride
        generic_stride = dataarray_cut.G.stride("eV", generic_dim_names=False)
        np.testing.assert_almost_equal(generic_stride, 0.0023255810)
        stride = dataarray_cut.G.stride(["eV"], generic_dim_names=False)
        np.testing.assert_almost_equal(stride, 0.0023255810)
        stride = dataarray_cut.G.stride(["eV", "phi"], generic_dim_names=False)
        np.testing.assert_array_almost_equal(stride, (0.0023255810, 0.001745))
        #
        range_ = dataarray_cut.G.range(generic_dim_names=False)
        np.testing.assert_array_almost_equal(range_["eV"], (-0.4255814, 0.13023245))
        np.testing.assert_array_almost_equal(
            range_["phi"],
            (0.22165681500327986, 0.6387905062299246),
        )
        range_ = dataarray_cut.G.range(generic_dim_names=True)
        np.testing.assert_array_almost_equal(range_["y"], (-0.4255814, 0.13023245))
        np.testing.assert_array_almost_equal(range_["x"], (0.22165681500327986, 0.6387905062299246))

    def test_experimental_conditions(self, dataset_cut: xr.Dataset) -> None:
        """Test for property experimenta_conditions."""
        assert dataset_cut.S.experimental_conditions == {
            "hv": 5.93,
            "polarization": np.nan,
            "temperature": np.nan,
        }

    def test_location_and_endstation(self, dataset_cut: xr.Dataset) -> None:
        """Test for property  endstation property."""
        assert dataset_cut.S.endstation == "ALG-MC"

    def test_history(self, dataarray_cut: xr.DataArray) -> None:
        """Test for S.history."""
        history = dataarray_cut.S.history
        assert history[0]["record"]["what"] == "Loaded MC dataset from FITS."

    def test_short_history(self, dataarray_cut: xr.DataArray) -> None:
        """Test for S.short_history."""
        history = dataarray_cut.S.short_history()
        assert history[0] == "load_MC"
        assert history[1] == "filesystem"


def test_select_around(dataarray_cut: xr.DataArray) -> None:
    """Test for select_around."""
    data_1 = dataarray_cut.S.select_around(points={"phi": 0.30}, radius={"phi": 0.05}).values
    data_2 = dataarray_cut.sel(phi=slice(0.25, 0.35)).sum("phi").values
    np.testing.assert_almost_equal(data_1, data_2)
    #
    data_1 = dataarray_cut.S.select_around(
        points={"phi": 0.30},
        radius={"phi": 0.05},
        mode="mean",
    ).values
    data_2 = dataarray_cut.sel(phi=slice(0.25, 0.35)).mean("phi").values
    np.testing.assert_almost_equal(data_1, data_2)
    #
    data_1 = dataarray_cut.S.select_around(points={"phi": 0.30}, radius={"phi": 0.000001}).values
    data_2 = dataarray_cut.sel(phi=0.3, method="nearest").values
    np.testing.assert_almost_equal(data_1, data_2)


def test_find(dataarray_cut: xr.DataArray) -> None:
    """Test for S.find."""
    assert sorted(dataarray_cut.S.find("offset")) == sorted(
        [
            "apply_offsets",
            "beta_offset",
            "chi_offset",
            "logical_offsets",
            "lookup_offset",
            "lookup_offset_coord",
            "offsets",
            "phi_offset",
            "psi_offset",
            "theta_offset",
            "with_rotation_offset",
        ],
    )


class TestEnergyNotation:
    """Test class for energy notation."""

    def test_energy_notation(self, dataarray_cut: xr.DataArray, dataset_cut: xr.Dataset) -> None:
        """Test for energy notation."""
        assert dataarray_cut.S.energy_notation == "Binding"
        assert dataset_cut.S.energy_notation == "Binding"

    def test_switch_energy_notation(
        self,
        dataarray_cut: xr.DataArray,
        dataset_cut: xr.Dataset,
        hv_map: xr.Dataset,
    ) -> None:
        """Test for switch energy notation."""
        # Test for DataArray
        dataarray_cut.S.switch_energy_notation()
        assert dataarray_cut.S.energy_notation == "Kinetic"
        dataarray_cut.S.switch_energy_notation()
        assert dataarray_cut.S.energy_notation == "Binding"

        # Test for Dataset
        dataset_cut.S.switch_energy_notation()
        assert dataset_cut.S.energy_notation == "Kinetic"
        dataset_cut.S.switch_energy_notation()
        assert dataset_cut.S.energy_notation == "Binding"

        with pytest.raises(RuntimeError) as e:
            hv_map.S.switch_energy_notation()
        assert str(e.value) == "Not impremented yet."

        with pytest.raises(RuntimeError) as e:
            hv_map.S.switch_energy_notation()
        assert str(e.value) == "Not impremented yet."

    def test_spectrum_type(self, dataarray_cut: xr.DataArray) -> None:
        """Test spectrum_type."""
        assert dataarray_cut.S.spectrum_type == "cut"
        del dataarray_cut.attrs["spectrum_type"]
        assert dataarray_cut.S.spectrum_type == "cut"

    def test_label(self, dataarray_cut: xr.DataArray) -> None:
        """Test scan_name."""
        assert dataarray_cut.S.label == "cut.fits"


class TestGeneralforDataArray:
    """Test class for "G"."""

    def test_G_stride(self, dataarray_cut: xr.DataArray) -> None:
        """Test for G.stride."""
        assert dataarray_cut.G.stride("x", "y") == [0.001745329251994332, 0.002325581000000021]

        assert dataarray_cut.G.stride(generic_dim_names=False) == {
            "phi": 0.001745329251994332,
            "eV": 0.002325581000000021,
        }


class TestAngleUnitforDataArray:
    """Test class for angle_unit for DataArray."""

    def test_angle_unit(self, dataarray_cut: xr.DataArray) -> None:
        """Test for angle unit property for DataArray."""
        assert dataarray_cut.S.angle_unit == "Radians"

    def test_setter_of_angle_unit(self, dataarray_cut: xr.DataArray) -> None:
        """Test for angle_unit setter."""
        dataarray_cut.S.angle_unit = "Degrees"
        assert dataarray_cut.S.angle_unit == "Degrees"
        assert dataarray_cut.attrs["angle_unit"] == "Degrees"

    def test_swap_angle_unit(self, dataarray_cut: xr.DataArray) -> None:
        """Test for swap_angle_unit (DataArray version)."""
        original_phi_coords = dataarray_cut.coords["phi"].values
        # rad -> deg
        dataarray_cut.S.swap_angle_unit()
        phi_coords = dataarray_cut.coords["phi"].values
        np.testing.assert_array_almost_equal(phi_coords[0:6], [12.7, 12.8, 12.9, 13.0, 13.1, 13.2])
        assert (
            dataarray_cut.coords["chi"]
            == dataarray_cut.attrs["chi_offset"]
            == np.rad2deg(-0.10909301748228785)
        )
        assert dataarray_cut.attrs["chi_offset"] == np.rad2deg(-0.10909301748228785)
        assert dataarray_cut.S.angle_unit == "Degrees"
        # deg -> rad
        dataarray_cut.S.swap_angle_unit()

        np.testing.assert_array_almost_equal(
            dataarray_cut.coords["phi"].values[0:6],
            original_phi_coords[0:6],
        )
        assert dataarray_cut.S.angle_unit == "Radians"
        #
        dataarray_cut.attrs["angle_unit"] = "Rad."
        with pytest.raises(TypeError):
            dataarray_cut.S.swap_angle_unit()

    def test_for_is_slit_vertical(self, dataarray_cut: xr.DataArray) -> None:
        """Test for is_slit_vertical (DataArray version)."""
        assert dataarray_cut.S.is_slit_vertical is False
        dataarray_cut.coords["alpha"] = np.pi / 2
        assert dataarray_cut.S.is_slit_vertical is True
        #
        dataarray_cut.S.swap_angle_unit()
        assert dataarray_cut.S.is_slit_vertical is True


class TestAngleUnitForDataset:
    """Test class for angle_unit for DataSet."""

    def test_angle_unit(self, dataset_cut: xr.Dataset) -> None:
        """Test for angle unit property for Dataset."""
        assert dataset_cut.S.angle_unit == "Radians"

    def test_setter_of_angle_unit(self, dataset_cut: xr.Dataset) -> None:
        """Test for angle_unit setter. (Dataset)."""
        dataset_cut.S.angle_unit = "Degrees"
        assert dataset_cut.S.angle_unit == "Degrees" == dataset_cut.attrs["angle_unit"]
        for spectrum in dataset_cut.S.spectra:
            assert spectrum.S.angle_unit == "Degrees"

    def test_swap_angle_unit(self, dataset_cut: xr.Dataset) -> None:
        """Test for swap_angle_unit (Dataset version)."""
        original_phi_coords = dataset_cut.coords["phi"].values
        # rad -> deg
        dataset_cut.S.swap_angle_unit()
        phi_coords = dataset_cut.coords["phi"].values
        np.testing.assert_array_almost_equal(phi_coords[0:6], [12.7, 12.8, 12.9, 13.0, 13.1, 13.2])
        assert (
            dataset_cut.coords["chi"]
            == dataset_cut.attrs["chi_offset"]
            == np.rad2deg(-0.10909301748228785)
        )
        assert dataset_cut.attrs["chi_offset"] == np.rad2deg(-0.10909301748228785)
        assert dataset_cut.S.angle_unit == "Degrees"
        for spectrum in dataset_cut.S.spectra:
            assert (
                spectrum.coords["chi"]
                == spectrum.attrs["chi_offset"]
                == np.rad2deg(-0.10909301748228785)
            )
            np.testing.assert_array_almost_equal(
                spectrum.coords["phi"][0:6],
                [12.7, 12.8, 12.9, 13.0, 13.1, 13.2],
            )

        # deg -> rad
        dataset_cut.S.swap_angle_unit()
        np.testing.assert_array_almost_equal(
            dataset_cut.coords["phi"].values[0:6],
            original_phi_coords[0:6],
        )
        assert dataset_cut.S.angle_unit == "Radians"

        for spectrum in dataset_cut.S.spectra:
            np.testing.assert_array_almost_equal(
                spectrum.coords["phi"].values[0:6],
                original_phi_coords[0:6],
            )

        # Exception test
        dataset_cut.attrs["angle_unit"] = "Rad."
        with pytest.raises(TypeError):
            dataset_cut.S.swap_angle_unit()

    def test_for_is_slit_vertical(self, dataset_cut: xr.Dataset) -> None:
        """Test for is_slit_vertical (Dataset version)."""
        assert dataset_cut.S.is_slit_vertical is False
        dataset_cut.coords["alpha"] = np.pi / 2
        for spectrum in dataset_cut.S.spectra:
            spectrum.coords["alpha"] = np.pi / 2
        assert dataset_cut.S.is_slit_vertical is True
        #
        dataset_cut.S.swap_angle_unit()
        assert dataset_cut.S.is_slit_vertical is True
