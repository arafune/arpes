"""Unit test for prodigy_xy.py."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from arpes.endstations.prodigy_xy import load_xy

data_dir = Path(__file__).parent.parent / "src" / "arpes" / "example_data"


@pytest.fixture
def sample_xy() -> xr.DataArray:
    """Fixture: load xy file into PyARPES compatible xr.DataArray."""
    return load_xy(data_dir / "BLGr_GK_map.xy")


class TestXYLoader:
    """Unit tests for load_xy."""

    def test_parameters(self, sample_xy: xr.DataArray):
        """Test parsing of metadata and axes."""
        # --- axes ---
        assert sample_xy.dims[0] == "eV"
        assert sample_xy.dims[1] == "nonenergy"
        assert "polar" in sample_xy.dims  # third dim name detected dynamically

        # --- params ---
        attrs = sample_xy.attrs
        assert isinstance(attrs["detector_voltage"], float)
        assert isinstance(attrs["values_curve"], int)
        assert isinstance(attrs["scan_mode"], str)
        assert attrs["scan_mode"] == "SnapshotFAT"

        np.testing.assert_allclose(attrs["values_curve"], 1)
        np.testing.assert_allclose(attrs["eff_workfunction"], 4.32)
        np.testing.assert_allclose(attrs["excitation_energy"], 21.2182)

        # --- axis values ---
        np.testing.assert_allclose(sample_xy.coords["eV"][5], 19.782284)
        np.testing.assert_allclose(sample_xy.coords["polar"][0], -68.0)

    def test_integrated_intensity(self, sample_xy: xr.DataArray):
        """Test integrated intensity through simple sum."""
        total = float(sample_xy.sum())
        np.testing.assert_allclose(total, 1.01248214e08)

    def test_data_array_shape(self, sample_xy: xr.DataArray):
        """Test DataArray shape and dim order."""
        assert sample_xy.dims == ("eV", "nonenergy", "polar")
        assert sample_xy.shape == (137, 82, 116)

        np.testing.assert_allclose(sample_xy.coords["polar"][0], -68.0)

    def test_axes_match_data_shape(self, sample_xy: xr.DataArray):
        """Ensure axes sizes match data shape."""
        sizes = [sample_xy.coords[d].size for d in sample_xy.dims]
        assert tuple(sizes) == sample_xy.shape
