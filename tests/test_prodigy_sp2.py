"""Unit test for prodigy_itx.py."""

from pathlib import Path

import pytest
import xarray as xr

from arpes.endstations.prodigy_sp2 import load_sp2

data_dir = Path(__file__).parent.parent / "src" / "arpes" / "example_data"


@pytest.fixture
def sample_sp2() -> xr.DataArray:
    """Fixture: produce xr.DataArray."""
    return load_sp2(data_dir / "GrIr_111_20230410_1.sp2")


class TestSp2:
    """Test class for load_sp2 function."""

    def test_parameters(self, sample_sp2: xr.DataArray) -> None:
        """Test sp2 file parameter.

        [TODO:description]

        Args:
            sample_sp2: [TODO:description]

        """
        assert sample_sp2.dims == ("phi", "eV")
