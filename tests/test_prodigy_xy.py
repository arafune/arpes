"""Unit test for prodigy_xy.py."""

from pathlib import Path

import numpy as np
import pytest

from arpes.endstations.prodigy_xy import ProdigyXY

data_dir = Path(__file__).parent.parent / "src" / "arpes" / "example_data"


@pytest.fixture
def sample_xy() -> ProdigyXY:
    """Fixture that reads data from a .xy file and returns a ProdigyXY object.

    Reads the contents of the "BLGr_GK_map.xy" file located in the data directory,
    and initializes a ProdigyXY object with the read data.

    Returns:
        ProdigyXY: An instance of ProdigyXY initialized with the data from the .xy file.
    """
    with Path(data_dir / "BLGr_GK_map.xy").open(mode="r") as xy_file:
        xy_data: list[str] = xy_file.readlines()
    return ProdigyXY(xy_data)


class TestXY:
    """test Class for prodigy_xy.py."""

    def test_parameters(self, sample_xy: ProdigyXY) -> None:
        """Test for reading xy file.

        This test verifies the following properties of the `sample_xy` object:
        - The axis information for dimensions d1, d2, and d3.
        - The types of various parameters in the `params` dictionary.
        - The values of specific parameters and axis information using `np.testing.assert_allclose`.

        Args:
            sample_xy (ProdigyXY): The sample object to be tested.

        """
        assert sample_xy.axis_info["d1"][1] == "eV"
        assert sample_xy.axis_info["d2"][1] == "nonenergy"
        assert sample_xy.axis_info["d3"][1] == "polar"
        assert isinstance(sample_xy.params["detector_voltage"], float)
        assert isinstance(sample_xy.params["values_curve"], int)
        assert isinstance(sample_xy.params["eff_workfunction"], float)
        assert isinstance(sample_xy.params["excitation_energy"], float)
        np.testing.assert_allclose(sample_xy.params["eff_workfunction"], 4.32)
        np.testing.assert_allclose(sample_xy.params["excitation_energy"], 21.2182)
        np.testing.assert_allclose(sample_xy.axis_info["d1"][0][5], 19.782284)
        np.testing.assert_allclose(sample_xy.axis_info["d3"][0][0], -68.0)

    def test_integrated_intensity(self, sample_xy: ProdigyXY) -> None:
        """Test the `integrated_intensity` property of the `ProdigyXY` class.

        This test verifies that the `integrated_intensity` property of the
        `sample_xy` instance of `ProdigyXY` returns the expected value.

        Parameters:
        sample_xy (ProdigyXY): An instance of the `ProdigyXY` class to be tested.

        """
        np.testing.assert_allclose(sample_xy.integrated_intensity, 1.01248214e+08)

    def test_convert_to_data_array(self, sample_xy: ProdigyXY) -> None:
        """Test the conversion of ProdigyXY object to an xarray.DataArray.

        This test verifies that the resulting DataArray has the correct dimensions,
        shape, and coordinate values.

        Args:
            sample_xy (ProdigyXY): An instance of ProdigyXY to be converted.

        """
        sample_data_array = sample_xy.to_data_array()
        assert sample_data_array.dims == ("eV", "nonenergy", "polar")
        assert sample_data_array.shape == (137, 82, 116)
        np.testing.assert_allclose(sample_data_array.coords["polar"][0], -68.0)
