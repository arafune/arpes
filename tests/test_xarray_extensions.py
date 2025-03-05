"""Unit test for xarray_extensions.py."""

import numpy as np
import pytest
import xarray as xr

from arpes.fits.fit_models import (
    AffineBackgroundModel,
    AffineBroadenedFD,
    LorentzianModel,
    QuadraticModel,
)
from arpes.fits.utilities import broadcast_model


class TestforProperties:
    """Test class for Array Dataset properties."""

    def test_hv_for_hv_map(self, hv_map: xr.Dataset) -> None:
        """Test for self.hv."""
        np.testing.assert_equal(hv_map.S.hv.values, np.linspace(50, 90, 21))

    def test_degrees_of_freedom_dims(self, xps_map: xr.Dataset) -> None:
        """Test for degrees_of_freedom."""
        assert xps_map.S.spectrum_degrees_of_freedom == {"eV"}
        assert xps_map.S.scan_degrees_of_freedom == {"x", "y"}

    def test_is_spatial(self, xps_map: xr.Dataset) -> None:
        """Test for is_* function."""
        assert xps_map.S.is_spatial

    def test_workfunction(self, dataarray_cut: xr.DataArray) -> None:
        """Test for S.workfunction."""
        assert dataarray_cut.S.work_function == 4.3
        dataarray_cut.attrs["sample_workfunction"] = 4.8
        assert dataarray_cut.S.work_function == 4.8
        dataarray_cut.attrs["workfunction"] = 4.5
        assert dataarray_cut.S.analyzer_work_function == 4.5
        dataarray_cut.attrs["inner_potential"] = 9.8
        assert dataarray_cut.S.inner_potential == 9.8

    def test_sum_other(self, dataarray_cut: xr.DataArray) -> None:
        """Test S.sum_other / mean_other."""
        small_region = dataarray_cut.sel({"eV": slice(-0.001, 0.0), "phi": slice(0.40, 0.41)})
        np.testing.assert_allclose(
            small_region.S.sum_other(["phi"]),
            np.array(
                [467, 472, 464, 458, 438],
            ),
        )
        np.testing.assert_allclose(
            small_region.S.mean_other(["phi"]),
            np.array(
                [467, 472, 464, 458, 438],
            ),
        )

    def test_property_for_degrees_of_freedom(
        self,
        dataset_cut: xr.Dataset,
    ) -> None:
        """Test for spectrum degrees of freedom."""
        assert dataset_cut.S.spectrum_degrees_of_freedom == {"phi", "eV"}
        assert dataset_cut.S.degrees_of_freedom == {"phi", "eV"}
        assert dataset_cut.S.spectrum_type == "cut"

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

    def test_property_sample_angles(self, dataarray_cut: xr.Dataset) -> None:
        """Test for sample_angles."""
        assert dataarray_cut.S.sample_angles[0] == 0
        assert dataarray_cut.S.sample_angles[1] == 0
        assert dataarray_cut.S.sample_angles[2] == -0.10909301748228785
        np.testing.assert_allclose(
            dataarray_cut.S.sample_angles[3][0:3].values,
            np.array([0.2216568, 0.2234021, 0.2251475]),
            rtol=1e-5,
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
        assert dataset_cut.S.beamline_settings == {
            "entrance_slit": np.nan,
            "exit_slit": np.nan,
            "hv": np.nan,
            "grating": None,
        }
        assert dataset_cut.S.reference_settings == {"hv": 5.93}

    def test_full_coords(self, dataset_cut: xr.Dataset) -> None:
        """Test for full coords."""
        assert sorted(dataset_cut.S.full_coords.keys()) == [
            "alpha",
            "beta",
            "chi",
            "eV",
            "hv",
            "phi",
            "psi",
            "theta",
            "x",
            "y",
            "z",
        ]

    def test_G_stride_and_range(self, dataarray_cut: xr.DataArray) -> None:
        """Test for G.range and G.stride."""
        generic_stride = dataarray_cut.G.stride(generic_dim_names=True)
        assert "x" in generic_stride
        assert "y" in generic_stride
        generic_stride = dataarray_cut.G.stride("eV", generic_dim_names=False)
        np.testing.assert_allclose(generic_stride, 0.0023255810)
        stride = dataarray_cut.G.stride(["eV"], generic_dim_names=False)
        np.testing.assert_allclose(stride, 0.0023255810)
        stride = dataarray_cut.G.stride(["eV", "phi"], generic_dim_names=False)
        np.testing.assert_allclose(stride, (0.0023255810, 0.001745), rtol=1e-3)
        range_ = dataarray_cut.G.range(generic_dim_names=False)
        np.testing.assert_allclose(range_["eV"], (-0.4255814, 0.13023245))
        np.testing.assert_allclose(
            range_["phi"],
            (0.22165681500327986, 0.6387905062299246),
        )
        range_ = dataarray_cut.G.range(generic_dim_names=True)
        np.testing.assert_allclose(range_["y"], (-0.4255814, 0.13023245))
        np.testing.assert_allclose(range_["x"], (0.22165681500327986, 0.6387905062299246))

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

    def test_symmetry_points(self, dataarray_cut: xr.DataArray) -> None:
        """Test around symmetry_points."""
        dataarray_cut.attrs["symmetry_points"] = {"G": {"phi": dataarray_cut.attrs["phi_offset"]}}
        sym_points = dataarray_cut.S.iter_own_symmetry_points
        assert next(sym_points) == ("G", {"phi": 0.405})
        with pytest.raises(StopIteration):
            next(sym_points)
        dataarray_cut.attrs["symmetry_points"] = {"XX": {"phi": dataarray_cut.attrs["phi_offset"]}}
        with pytest.raises(RuntimeError):
            dataarray_cut.S.symmetry_points()


def test_select_around(dataarray_cut: xr.DataArray) -> None:
    """Test for select_around."""
    data_1 = dataarray_cut.S.select_around(point={"phi": 0.30}, radius={"phi": 0.05}).values
    data_2 = dataarray_cut.sel(phi=slice(0.25, 0.35)).sum("phi").values
    np.testing.assert_allclose(data_1, data_2)
    data_1 = dataarray_cut.S.select_around(
        point={"phi": 0.30},
        radius={"phi": 0.05},
        mode="mean",
    ).values
    data_2 = dataarray_cut.sel(phi=slice(0.25, 0.35)).mean("phi").values
    np.testing.assert_allclose(data_1, data_2)
    data_1 = dataarray_cut.S.select_around(point={"phi": 0.30}, radius={"phi": 0.000001}).values
    data_2 = dataarray_cut.sel(phi=0.3, method="nearest").values
    np.testing.assert_allclose(data_1, data_2)


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
        assert dataarray_cut.S.energy_notation == "Final"
        dataarray_cut.S.switch_energy_notation()
        assert dataarray_cut.S.energy_notation == "Binding"

        # Test for Dataset
        dataset_cut.S.switch_energy_notation()
        assert dataset_cut.S.energy_notation == "Final"
        dataset_cut.S.switch_energy_notation()
        assert dataset_cut.S.energy_notation == "Binding"

        with pytest.raises(RuntimeError) as e:
            hv_map.S.switch_energy_notation()
        assert str(e.value) == "Not implemented yet."

        with pytest.raises(RuntimeError) as e:
            hv_map.S.switch_energy_notation()
        assert str(e.value) == "Not implemented yet."
        with pytest.raises(RuntimeError) as e:
            hv_map.spectrum.S.switch_energy_notation()
        assert str(e.value) == "Not implemented yet."

    def test_spectrum_type(self, dataarray_cut: xr.DataArray) -> None:
        """Test spectrum_type."""
        assert dataarray_cut.S.spectrum_type == "cut"
        del dataarray_cut.attrs["spectrum_type"]
        assert dataarray_cut.S.spectrum_type == "cut"

    def test_label(self, dataarray_cut: xr.DataArray, dataarray_cut2: xr.DataArray) -> None:
        """Test scan_name."""
        assert dataarray_cut.S.label == "cut.fits"
        assert dataarray_cut2.S.scan_name == "ID: 2"


class TestGeneralforDataArray:
    """Test class for "G"."""

    def test_G_iter_coords(self, dataarray_cut: xr.DataArray) -> None:
        """Test for G.iter_coords."""
        eV_generator = dataarray_cut.G.iter_coords("eV")
        assert next(eV_generator) == {"eV": -0.4255814}

    def test_G_stride(self, dataarray_cut: xr.DataArray) -> None:
        """Test for G.stride."""
        assert dataarray_cut.G.stride("x", "y") == [0.001745329251994332, 0.002325581000000021]

        assert dataarray_cut.G.stride(generic_dim_names=False) == {
            "phi": 0.001745329251994332,
            "eV": 0.002325581000000021,
        }

    def test_enumerate_iter_coords(self, dataarray_map: xr.DataArray) -> None:
        """Test for G.test_enumerate_iter_coords."""
        enumerate_ = dataarray_map.G.enumerate_iter_coords()
        first_ = next(enumerate_)
        assert first_[0] == (0, 0, 0)
        assert first_[1] == {
            "theta": -0.20943951023931953,
            "eV": -1.3371349573135376,
            "phi": -0.2910254835234106,
        }

    @pytest.fixture
    def edge(self, dataarray_map: xr.DataArray) -> xr.DataArray:
        fmap = dataarray_map
        cut = fmap.sum("theta", keep_attrs=True).sel(eV=slice(-0.2, 0.1), phi=slice(-0.25, 0.3))
        fit_results = broadcast_model(AffineBroadenedFD, cut, "phi")
        return (
            QuadraticModel()
            .guess_fit(
                fit_results.results.F.p("center"),
            )
            .eval(x=fmap.phi)
        )

    def test_G_shift(
        self,
        dataarray_map: xr.DataArray,
        edge: xr.DataArray,
    ) -> None:
        """Test for G.shift_by."""
        fmap = dataarray_map
        np.testing.assert_allclose(
            actual=fmap.G.shift_by(
                edge,
                shift_axis="eV",
                by_axis="phi",
            ).sel(
                eV=0,
                method="nearest",
            )[:][0][:5],
            desired=np.array([5.625749, 566.8711542, 757.8334417, 637.2900199, 610.679927]),
            rtol=1e-2,
        )

    def test_G_shift_with_extend_coords(
        self,
        dataarray_map: xr.DataArray,
        edge: xr.DataArray,
    ) -> None:
        """Test for G.shift_by with extend_coords."""
        fmap = dataarray_map
        shifted_without_extension_coords = fmap.G.shift_by(
            edge,
            shift_axis="eV",
            by_axis="phi",
            extend_coords=False,
        )
        assert shifted_without_extension_coords.shape == (81, 150, 111)

        shifted_with_extension_coords = fmap.G.shift_by(
            edge,
            shift_axis="eV",
            by_axis="phi",
            extend_coords=True,
        )

        assert shifted_with_extension_coords.shape == (81, 154, 111)

        should_be_zero_array = (
            (
                shifted_with_extension_coords.isel(eV=slice(0, 150))
                - shifted_without_extension_coords
            )
            .fillna(0)
            .values
        )
        assert np.all(should_be_zero_array == 0)

    def test_G_meshgrid(self, dataarray_cut: xr.DataArray) -> None:
        """Test for G.meshgrid, G.scale_meshgrid, G.shift_meshgrid."""
        small_region = dataarray_cut.sel({"eV": slice(-0.01, 0.0), "phi": slice(0.40, 0.42)})
        meshgrid_results = small_region.G.meshgrid()
        np.testing.assert_allclose(
            meshgrid_results["phi"][0],
            np.array(
                [
                    0.40142573,
                    0.40317106,
                    0.40491639,
                    0.40666172,
                    0.40840704,
                    0.41015237,
                    0.4118977,
                    0.41364303,
                    0.41538836,
                    0.41713369,
                    0.41887902,
                ],
            ),
        )
        np.testing.assert_allclose(
            meshgrid_results["eV"][-1],
            np.array(
                [
                    -7.7e-08,
                    -7.7e-08,
                    -7.7e-08,
                    -7.7e-08,
                    -7.7e-08,
                    -7.7e-08,
                    -7.7e-08,
                    -7.7e-08,
                    -7.7e-08,
                    -7.7e-08,
                    -7.7e-08,
                ],
            ),
        )
        meshgrid_set = small_region.G.meshgrid(as_dataset=True)
        assert isinstance(meshgrid_set, xr.Dataset)

    def test_G_ravel(self, dataarray_cut: xr.DataArray) -> None:
        """Test for G.ravel."""
        small_region = dataarray_cut.sel({"eV": slice(-0.001, 0.0), "phi": slice(0.40, 0.41)})
        ravel_ = small_region.G.ravel()
        np.testing.assert_allclose(
            ravel_["phi"],
            np.array([0.40142573, 0.40317106, 0.40491639, 0.40666172, 0.40840704]),
        )
        np.testing.assert_allclose(
            ravel_["eV"],
            np.array([-7.7e-08, -7.7e-08, -7.7e-08, -7.7e-08, -7.7e-08]),
        )
        np.testing.assert_allclose(ravel_["data"], np.array([467, 472, 464, 458, 438]))


class TestGeneralforDataset:
    """Test class for GenericDatasetAccessor."""

    @pytest.fixture
    def near_ef(self, dataset_temperature_dependence: xr.Dataset) -> xr.DataArray:
        return (
            dataset_temperature_dependence.sel(
                eV=slice(-0.05, 0.05),
                phi=slice(-0.2, None),
            )
            .sum(dim="eV")
            .spectrum
        )

    @pytest.fixture
    def phi_values(self, near_ef: xr.DataArray) -> xr.DataArray:
        return broadcast_model(
            [AffineBackgroundModel, LorentzianModel],
            near_ef,
            "temperature",
        ).results.F.p("b_center")

    def test_select_around_data(
        self,
        dataset_temperature_dependence: xr.Dataset,
        phi_values: xr.DataArray,
    ) -> None:
        selected_data: xr.DataArray = dataset_temperature_dependence.spectrum.S.select_around_data(
            {"phi": phi_values},
            mode="mean",
            radius={"phi": 0.005},
        )
        assert selected_data.dims == ("eV", "temperature")
        np.testing.assert_allclose(
            selected_data.values[0][:5],
            np.array([442.52083333, 420.17708333, 402.65625, 434.79166667, 451.3515625]),
        )

    def test_select_around_data2(
        self,
        dataset_temperature_dependence: xr.Dataset,
        phi_values: xr.DataArray,
    ) -> None:
        selected_data: xr.DataArray = dataset_temperature_dependence.spectrum.S.select_around_data(
            points={"phi": phi_values},
            mode="sum",
            radius={"phi": 0.005},
        )
        assert selected_data.dims == ("eV", "temperature")
        np.testing.assert_allclose(
            selected_data.values[0][:5],
            np.array([1327.5625, 1260.53125, 1207.96875, 1304.375, 1805.40625]),
        )

    def test__radius(
        self,
        dataset_temperature_dependence: xr.Dataset,
        phi_values: xr.DataArray,
    ) -> None:
        selected_data: xr.DataArray = dataset_temperature_dependence.spectrum.S.select_around_data(
            points={"phi": phi_values},
            mode="sum",
            radius={"phi": 0.005},
        )

        should_same_as_above: xr.DataArray = (
            dataset_temperature_dependence.spectrum.S.select_around_data(
                points={"phi": phi_values},
                mode="sum",
                radius=0.005,
            )
        )

        np.testing.assert_allclose(selected_data.values, should_same_as_above.values)

    def test__if_radius_is_None(
        self,
        dataset_temperature_dependence: xr.Dataset,
        phi_values: xr.DataArray,
    ) -> None:
        radius = dataset_temperature_dependence.spectrum.S._radius(
            points={"phi": phi_values},
            radius=None,
        )
        assert radius == {"phi": 0.02}

    def test__if_radius_is_array(
        self,
        dataset_temperature_dependence: xr.Dataset,
        phi_values: xr.DataArray,
    ) -> None:
        with pytest.raises(TypeError, match="radius should be a float, dictionary or None"):
            dataset_temperature_dependence.spectrum.S._radius(
                points={"phi": phi_values},
                radius=[0.02],
            )

    def test_G_shift(self, near_ef: xr.DataArray, phi_values: xr.DataArray):
        #
        # Taken from custom-dot-t-function.ipynb
        #
        near_ef.G.shift_by(phi_values - phi_values.mean(), shift_axis="phi")
        np.testing.assert_allclose(
            near_ef.sel(phi=-0.12, method="nearest").values[:5],
            np.array(
                [4041.9375, 4023.5, 4068.875, 4011.21875, 4002.75],
            ),
        )

    def test_G_meshgrid_operation(self, dataarray_cut: xr.DataArray):
        """Test G.scale_meshgrid and G.shift_meshgrid, and transform_meshgrid."""
        small_region = dataarray_cut.sel({"eV": slice(-0.01, 0.0), "phi": slice(0.40, 0.42)})
        meshgrid_set = small_region.G.meshgrid(as_dataset=True)
        shifted_meshgrid = meshgrid_set.G.shift_meshgrid(("phi",), -0.2)
        np.testing.assert_allclose(
            shifted_meshgrid["phi"][1].values[:3],
            np.array(
                [0.20142573, 0.20317106, 0.20491639],
            ),
        )

        scaled_meshgrid = meshgrid_set.G.scale_meshgrid(("eV",), 1.5)
        np.testing.assert_allclose(
            scaled_meshgrid["eV"][-1].values[:3],
            np.array(
                [-1.155e-07, -1.155e-07, -1.155e-07],
            ),
        )


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
        assert dataarray_cut.S.angle_unit == "Degrees"
        # deg -> rad
        dataarray_cut.S.switch_angle_unit()

        np.testing.assert_allclose(
            dataarray_cut.coords["phi"].values[0:6],
            original_phi_coords[0:6],
        )
        assert dataarray_cut.S.angle_unit == "Radians"

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
        assert dataset_cut.S.angle_unit == "Radians"

    def test_setter_of_angle_unit(self, dataset_cut: xr.Dataset) -> None:
        """Test for angle_unit setter. (Dataset)."""
        dataset_cut.S.angle_unit = "Degrees"
        assert dataset_cut.S.angle_unit == "Degrees" == dataset_cut.attrs["angle_unit"]
        for spectrum in dataset_cut.S.spectra:
            assert spectrum.S.angle_unit == "Degrees"

    def test_switch_angle_unit(self, dataset_cut: xr.Dataset) -> None:
        """Test for switch_angle_unit (Dataset version)."""
        original_phi_coords = dataset_cut.coords["phi"].values
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
        assert dataset_cut.S.angle_unit == "Degrees"
        for spectrum in dataset_cut.S.spectra:
            assert (
                spectrum.coords["chi"]
                == spectrum.attrs["chi_offset"]
                == np.rad2deg(-0.10909301748228785)
            )
            np.testing.assert_allclose(
                spectrum.coords["phi"][0:6],
                [12.7, 12.8, 12.9, 13.0, 13.1, 13.2],
            )

        # deg -> rad
        dataset_cut.S.switch_angle_unit()
        np.testing.assert_allclose(
            dataset_cut.coords["phi"].values[0:6],
            original_phi_coords[0:6],
        )
        assert dataset_cut.S.angle_unit == "Radians"

        for spectrum in dataset_cut.S.spectra:
            np.testing.assert_allclose(
                spectrum.coords["phi"].values[0:6],
                original_phi_coords[0:6],
            )

    def test_for_is_slit_vertical(self, dataset_cut: xr.Dataset) -> None:
        """Test for is_slit_vertical (Dataset version)."""
        assert dataset_cut.S.is_slit_vertical is False
        dataset_cut.coords["alpha"] = np.pi / 2
        for spectrum in dataset_cut.S.spectra:
            spectrum.coords["alpha"] = np.pi / 2
        assert dataset_cut.S.is_slit_vertical is True
        dataset_cut.S.switch_angle_unit()
        assert dataset_cut.S.is_slit_vertical is True


class TestShiftCoords:
    """Test class for correction of coordinates of the XArray."""

    def test_corrected_coords_with_cut_by_phi_offset(self, dataarray_cut: xr.DataArray) -> None:
        """Test the corrected_coords method with the cut DataArray.

        Args:
            dataarray_cut (xr.DataArray): The input DataArray.
        """
        corrected_cut = dataarray_cut.S.corrected_coords("phi_offset")
        assert corrected_cut.attrs["phi_offset"] == 0
        np.testing.assert_array_almost_equal(
            corrected_cut.coords["phi"].values[:5],
            np.array([-0.18334318, -0.18159786, -0.17985253, -0.1781072, -0.17636187]),
        )

    def test_correct_coords_with_cut_by_phi_offset(self, dataarray_cut: xr.DataArray) -> None:
        """Test the correct_coords method with a cut DataArray.

        Args:
            dataarray_cut (xr.DataArray): The input DataArray.
        """
        correct_cut = dataarray_cut.S.correct_coords("phi_offset")
        assert correct_cut is None
        assert dataarray_cut.attrs["phi_offset"] == 0
        np.testing.assert_array_almost_equal(
            dataarray_cut.coords["phi"].values[:5],
            np.array([-0.18334318, -0.18159786, -0.17985253, -0.1781072, -0.17636187]),
        )

    def test_correct_coords_with_cut2_by_phi_offset_and_beta(
        self,
        dataarray_cut2: xr.DataArray,
    ) -> None:
        """Test the correct_coords method with the cut2 DataArray.

        Args:
            dataarray_cut2 (xr.DataArray): The input DataArray.
        """
        dataarray_cut2.S.correct_coords("phi_offset")
        assert dataarray_cut2.attrs["phi_offset"] == 0
        np.testing.assert_array_almost_equal(
            dataarray_cut2.coords["phi"].values[:5],
            np.array(
                [
                    -0.22325728,
                    -0.22253006,
                    -0.22180284,
                    -0.22107561,
                    -0.22034839,
                ],
            ),
        )

        dataarray_cut2.S.correct_coords("beta")
        assert dataarray_cut2.attrs["beta"] == 0
        np.testing.assert_array_almost_equal(
            dataarray_cut2.coords["phi"].values[:5],
            np.array(
                [
                    -0.27561716,
                    -0.27488994,
                    -0.27416271,
                    -0.27343549,
                    -0.27270827,
                ],
            ),
        )

    def test_corrected_coords_with_cut2_mulitiple_corrections(
        self,
        dataarray_cut2: xr.DataArray,
    ) -> None:
        """Test the corrected_coords method with the cut2 DataArray and multiple corrections.

        Args:
            dataarray_cut2 (xr.DataArray): The input DataArray.
        """
        corrected = dataarray_cut2.S.corrected_coords("phi_offset").S.corrected_coords("beta")
        dataarray_cut2.S.correct_coords("beta")
        dataarray_cut2.S.correct_coords("phi_offset")
        np.testing.assert_array_almost_equal(
            corrected.coords["phi"].values,
            dataarray_cut2.coords["phi"].values,
        )

    def test_corrected_coords_with_cut2_using_tuple_of_collections(
        self,
        dataarray_cut2: xr.DataArray,
    ) -> None:
        """Test the corrected_coords method with the cut2 DataArray using a tuple of corrections.

        Args:
            dataarray_cut2 (xr.DataArray): The input DataArray.
        """
        corrected1 = dataarray_cut2.S.corrected_coords("phi_offset").S.corrected_coords("beta")
        corrected2 = dataarray_cut2.S.corrected_coords(("phi_offset", "beta"))
        np.testing.assert_array_almost_equal(
            corrected1.coords["phi"].values,
            corrected2.coords["phi"].values,
        )


class TestFatSel:
    """Test class for S.fat_sel."""

    def test_fat_sel_with_sum(self, dataarray_map: xr.DataArray) -> None:
        fat1 = dataarray_map.S.fat_sel(eV=0, method="sum")
        expected = dataarray_map.sel({"eV": slice(-0.025, 0.025)}).sum("eV")
        np.testing.assert_array_almost_equal(fat1.values, expected.values)

    def test_fat_sel_with_mean(self, dataarray_map: xr.DataArray) -> None:
        fat1 = dataarray_map.S.fat_sel(widths={"eV": 0.05}, eV=-0.1)
        expected = dataarray_map.sel({"eV": slice(-0.125, -0.075)}).mean("eV")
        np.testing.assert_array_almost_equal(fat1.values, expected.values)

    def test_arg_handling(self, dataarray_map: xr.DataArray) -> None:
        """Test handling arguments in fat_sel."""
        fat1 = dataarray_map.S.fat_sel(eV=0)
        fat2 = dataarray_map.S.fat_sel(widths={"eV": 0.05}, eV=0)
        fat3 = dataarray_map.S.fat_sel(eV=0, eV_width=0.05)

        np.testing.assert_array_almost_equal(fat2.values, fat3.values)
        np.testing.assert_array_almost_equal(fat1.values, fat2.values)

    def test_fat_sel_raises_type_error(self, dataarray_map: xr.DataArray) -> None:
        with pytest.raises(TypeError, match="The slice center is not spcefied."):
            dataarray_map.S.fat_sel(widths={"eV": 0.05})
        with pytest.raises(TypeError, match="The slice center is not spcefied."):
            dataarray_map.S.fat_sel(eV_width=0.05)

    def test_fat_sel_raise_runtime_error(self, dataarray_map: xr.DataArray) -> None:
        with pytest.raises(RuntimeError, match="Method should be either 'mean' or 'sum'."):
            dataarray_map.S.fat_sel(eV=0.05, method="suum")
