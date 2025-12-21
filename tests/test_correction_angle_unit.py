"""Unit test for correction.angle_unit module."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import xarray as xr

from arpes.xarray_extensions.accessor.spectrum_type import AngleUnit
from arpes.correction.angle_unit import (
    degree_to_radian,
    radian_to_degree,
    switch_angle_unit,
    switched_angle_unit,
)
from arpes.xarray_extensions.accessor.spectroscopy import (
    ARPESDataArrayAccessor,
    ARPESDatasetAccessor,
)


@pytest.fixture
def mock_data_deg():
    return xr.DataArray(
        np.zeros((3, 3)),
        coords={"phi": [0, 45, 90], "theta": [0, 10, 20]},
        dims=("phi", "theta"),
        attrs={
            "angle_unit": "Degrees",
            "phi": 45,
            "phi_offset": 5,
            "other_attr": 100,
        },
    )


@pytest.fixture
def mock_data_rad():
    return xr.DataArray(
        np.zeros((3, 3)),
        coords={"phi": [0, np.pi / 4, np.pi / 2], "theta": [0, 0.1, 0.2]},
        dims=("phi", "theta"),
        attrs={
            "angle_unit": "Radians",
            "phi": np.pi / 4,
            "phi_offset": 0.1,
        },
    )


class TestAngleConversion:
    def test_degree_to_radian(self, mock_data_deg):
        mock_data_deg.S.angle_unit = AngleUnit.DEG

        result = degree_to_radian(mock_data_deg)

        assert result.attrs["angle_unit"] == "Radians"
        assert result.coords["phi"].values == pytest.approx(np.deg2rad([0, 45, 90]))
        assert result.attrs["phi"] == pytest.approx(np.deg2rad(45))
        assert result.attrs["phi_offset"] == pytest.approx(np.deg2rad(5))
        assert result.attrs["other_attr"] == 100  # 角度以外は維持

    def test_degree_to_radian_wo_change(self, mock_data_rad):
        result = degree_to_radian(mock_data_rad)

        assert result is mock_data_rad

    def test_radian_to_degree_wo_change(self, mock_data_deg):
        result = radian_to_degree(mock_data_deg)

        assert result is mock_data_deg

    def test_radian_to_degree(self, mock_data_rad):
        mock_data_rad.S.angle_unit = AngleUnit.RAD

        result = radian_to_degree(mock_data_rad)

        assert result.attrs["angle_unit"] == "Degrees"
        assert result.coords["phi"].values == pytest.approx([0, 45, 90])
        assert result.attrs["phi"] == pytest.approx(45)

    def test_switched_angle_unit_returns_copy(self, mock_data_deg):
        from arpes.xarray_extensions.accessor.spectrum_type import AngleUnit

        mock_data_deg.S.angle_unit = AngleUnit.DEG

        original_val = mock_data_deg.coords["phi"].values[1]
        result = switched_angle_unit(mock_data_deg)

        assert result.attrs["angle_unit"] == "Radians"
        assert mock_data_deg.attrs["angle_unit"] == "Degrees"
        assert mock_data_deg is not result

    def test_switch_angle_unit_inplace(self, mock_data_deg):
        mock_data_deg.S.angle_unit = AngleUnit.DEG

        switch_angle_unit(mock_data_deg)

        assert mock_data_deg.attrs["angle_unit"] == "Radians"
        assert mock_data_deg.coords["phi"].values[1] == pytest.approx(np.deg2rad(45))
