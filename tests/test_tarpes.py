"""Unit test for tarpes.py."""

import numpy as np
import xarray as xr
from IPython.display import HTML

from arpes.analysis import tarpes


def test_find_t_for_max_intensity(mock_tarpes: list[xr.DataArray]) -> None:
    """Test for find_t_for_max_intensity."""
    tarpes_dataarray = tarpes.build_crosscorrelation(
        mock_tarpes,
        delayline_dim="position",
        delayline_origin=100.31,
    )
    np.testing.assert_allclose(
        tarpes.find_t_for_max_intensity(tarpes_dataarray),
        1021.2881894590657,
        rtol=1e-5,
    )
    tarpes_dataarray = tarpes.build_crosscorrelation(
        mock_tarpes,
        delayline_dim="position",
        delayline_origin=100.31,
        convert_position_to_time=None,
    )
    assert tarpes.find_t_for_max_intensity(tarpes_dataarray) == 0.15308724832215148

    tarpes_dataarray = tarpes.build_crosscorrelation(
        mock_tarpes,
        delayline_dim="position",
        delayline_origin=0.0,
        convert_position_to_time=None,
    )
    assert tarpes.find_t_for_max_intensity(tarpes_dataarray) == 100.46308724832215


def test_as_movie(mock_tarpes: list[xr.DataArray]) -> None:
    """Test xarray.G.as_movie."""
    tarpes_dataarray = tarpes.build_crosscorrelation(
        mock_tarpes,
        delayline_dim="position",
        delayline_origin=100.31,
    )
    anim = tarpes_dataarray.G.as_movie()
    assert type(anim) is HTML
    anim_out = tarpes_dataarray.G.as_movie(out="test.mp4")
    assert "test.mp4" in str(anim_out)
