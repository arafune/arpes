"""Unit test for prodigy_xy.py."""

from pathlib import Path

import numpy as np
import pytest
from arpes.endstations.prodigy_xy import ProdigyXY

data_dir = Path(__file__).parent.parent / "src" / "arpes" / "example_data"


@pytest.fixture()
def sample_xy() -> ProdigyXY:
    """Fixture."""
    with Path(data_dir / "example_xy_data_bulk_si.xy").open(mode="r") as xy_file:
        xy_data: list[str] = xy_file.readlines()
    return ProdigyXY(xy_data)


class TestXY:
    """test Class for prodigy_xy.py."""

    def test_parameters(self, sample_xy: ProdigyXY) -> None:
        """Test for reading xy file."""
        assert sample_xy.axis_info["d1"][1] == "eV"
        assert sample_xy.axis_info["d2"][1] == "nonenergy"
        assert isinstance(sample_xy.params["detector_voltage"], float)
        assert isinstance(sample_xy.params["values_curve"], int)
        np.testing.assert_almost_equal(sample_xy.params["eff_workfunction"], 4.3)
        np.testing.assert_almost_equal(sample_xy.params["excitation_energy"], 21.2182)
        np.testing.assert_almost_equal(sample_xy.axis_info["d1"][0][5], 16.798332279411763)

    def test_integrated_intensity(self, sample_xy: ProdigyXY) -> None:
        """Test for integrated_intensity property."""
        np.testing.assert_almost_equal(sample_xy.integrated_intensity, 1131362.930806)

    def test_convert_to_data_array(self, sample_xy: ProdigyXY) -> None:
        """Test for convert to xr.DataArray."""
        sample_data_array = sample_xy.to_data_array()
        assert sample_data_array.dims == ("eV", "nonenergy")
        assert sample_data_array.shape == (137, 82)
