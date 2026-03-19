"""Unit test for prodigy_xy.py."""

from pathlib import Path

import numpy as np
import pytest

from arpes.endstations.prodigy_xy import ProdigyXY

data_dir = Path(__file__).parent.parent / "src" / "arpes" / "example_data"


@pytest.fixture
def sample_xy() -> ProdigyXY:
    """Fixture."""
    with Path(data_dir / "BLGr_GK_map.xy").open(mode="r") as xy_file:
        xy_data: list[str] = xy_file.readlines()
    return ProdigyXY(xy_data)


class TestXY:
    """test Class for prodigy_xy.py."""

    def test_parameters(self, sample_xy: ProdigyXY) -> None:
        """Test parsing of metadata and axes."""
        # --- axes ---
        assert sample_xy.axes[0].name == "eV"
        assert sample_xy.axes[1].name == "nonenergy"
        assert sample_xy.axes[2].name == "polar"

        # --- params ---
        assert isinstance(sample_xy.params["detector_voltage"], float)
        assert isinstance(sample_xy.params["values_curve"], int)

        np.testing.assert_allclose(float(sample_xy.params["eff_workfunction"]), 4.32)
        np.testing.assert_allclose(float(sample_xy.params["excitation_energy"]), 21.2182)

        # --- axis values ---
        np.testing.assert_allclose(sample_xy.axes[0].values[5], 19.782284)
        np.testing.assert_allclose(sample_xy.axes[2].values[0], -68.0)

    def test_integrated_intensity(self, sample_xy: ProdigyXY) -> None:
        """Test integrated intensity property."""
        np.testing.assert_allclose(sample_xy.integrated_intensity, 1.01248214e08)

    def test_convert_to_data_array(self, sample_xy: ProdigyXY) -> None:
        """Test conversion to xr.DataArray."""
        data_array = sample_xy.to_data_array()

        assert data_array.dims == ("eV", "nonenergy", "polar")
        assert data_array.shape == (137, 82, 116)

        np.testing.assert_allclose(data_array.coords["polar"][0], -68.0)

    def test_axes_match_data_shape(self, sample_xy: ProdigyXY) -> None:
        """Ensure axes sizes match intensity shape."""
        sizes = [len(ax.values) for ax in sample_xy.axes]
        assert tuple(sizes) == sample_xy.intensity.shape
