"""Unit test for tarpes.py."""

import xarray as xr

from arpes.analysis import tarpes


def test_find_t_for_max_intensity(mock_tarpes: list[xr.DataArray]) -> None:
    """Test for find_t_for_max_intensity."""
    tarpes_dataarray = tarpes.build_crosscorrelation(
        mock_tarpes,
        delayline_dim="position",
        delayline_origin=100.31,
    )
    assert tarpes.find_t_for_max_intensity(tarpes_dataarray) == 1021.2881894590657
    tarpes_dataarray = tarpes.build_crosscorrelation(
        mock_tarpes,
        delayline_dim="position",
        delayline_origin=100.31,
        convert_position_to_time=False,
    )
    assert tarpes.find_t_for_max_intensity(tarpes_dataarray) == 0.15308724832215148
