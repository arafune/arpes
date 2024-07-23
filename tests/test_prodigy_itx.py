"""Unit test for prodigy_itx.py."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from arpes.endstations.prodigy_itx import ProdigyItx, load_sp2

data_dir = Path(__file__).parent.parent / "src" / "arpes" / "example_data"


@pytest.fixture()
def sample_itx() -> ProdigyItx:
    """Fixture."""
    with Path(data_dir / "example_itx_data.itx").open(mode="r") as itx_file:
        itx_data: list[str] = itx_file.readlines()
    return ProdigyItx(itx_data)


@pytest.fixture()
def sample_sp2() -> xr.DataArray:
    """Fixture: produce xr.DataArray."""
    return load_sp2(data_dir / "GrIr_111_20230410_1.sp2")


class TestItx:
    """test Class for prodigy_itx.py."""

    def test_parameters(self, sample_itx: ProdigyItx) -> None:
        """Test for reading itx file."""
        workfunction_analyzer = 4.401
        assert sample_itx.params["WorkFunction"] == workfunction_analyzer
        assert sample_itx.pixels == (600, 501)
        assert sample_itx.axis_info["x"] == ("I", -12.4792, 12.4792, "deg (theta_y)")

        """Test for integrated_intensity property."""

    def test_integrated_intensity(self, sample_itx: ProdigyItx) -> None:
        """Test for integrated_intensity property."""
        np.testing.assert_almost_equal(sample_itx.integrated_intensity, 666371.3147352)

    def test_convert_to_dataarray(self, sample_itx: ProdigyItx) -> None:
        """Test for convert to xr.DataArray."""
        sample_dataarray = sample_itx.to_data_array()
        assert sample_dataarray.dims == ("phi", "eV")

    def test_convert_itx_fiile(self, sample_itx: ProdigyItx) -> None:
        """Test itx file.

        [TODO:description]

        Args:
            sample_itx: [TODO:description]

        Returns:
            [TODO:description]
        """


class TestSp2:
    """Test class for load_sp2 function."""

    def test_parameters(self, sample_sp2: xr.DataArray) -> None:
        """Test sp2 file parameter.

        [TODO:description]

        Args:
            sample_sp2: [TODO:description]

        """
        assert sample_sp2.dims == ("phi", "eV")
