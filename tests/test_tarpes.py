"""Unit test for tarpes.py."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from IPython.display import HTML

from arpes.analysis import tarpes
from src.arpes.plotting.movie import (
    _replace_after_col,
    _replace_after_row,
    plot_movie_and_evolution,
)


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


def test_replace_after_col():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    col_num = 2
    expected_output = np.array([[1, 2, np.nan], [4, 5, np.nan], [7, 8, np.nan]], dtype=np.float64)
    output = _replace_after_col(array, col_num)
    np.testing.assert_array_equal(output, expected_output)


def test_replace_after_col_no_replacement():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    col_num = 3
    expected_output = array
    output = _replace_after_col(array, col_num)
    np.testing.assert_array_equal(output, expected_output)


def test_replace_after_col_all_replacement():
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    col_num = 0
    expected_output = np.full(array.shape, np.nan)
    output = _replace_after_col(array, col_num)
    np.testing.assert_array_equal(output, expected_output)


# Assuming _replace_after_row is defined somewhere above or imported
# from your_module import _replace_after_row


def test_replace_after_row() -> None:
    # Test case 1: Normal case
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    expected = np.array([[1, 2, 3], [4, 5, 6], [np.nan, np.nan, np.nan]], dtype=np.float64)
    result = _replace_after_row(array, 2)
    np.testing.assert_array_equal(result, expected)

    # Test case 2: Edge case where row_num is 0
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    expected = np.array(
        [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
        dtype=np.float64,
    )
    result = _replace_after_row(array, 0)
    np.testing.assert_array_equal(result, expected)

    # Test case 3: Edge case where row_num is the last row
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    expected = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    result = _replace_after_row(array, 3)
    np.testing.assert_array_equal(result, expected)

    # Test case 4: Single row array
    array = np.array([[1, 2, 3]], dtype=np.float64)
    expected = np.array([[1, 2, 3]], dtype=np.float64)
    result = _replace_after_row(array, 1)
    np.testing.assert_array_equal(result, expected)

    # Test case 5: Single column array
    array = np.array([[1], [2], [3]], dtype=np.float64)
    expected = np.array([[1], [2], [np.nan]], dtype=np.float64)
    result = _replace_after_row(array, 2)
    np.testing.assert_array_equal(result, expected)


@pytest.fixture
def sample_data(mock_tarpes: list[xr.DataArray]):
    return tarpes.build_crosscorrelation(
        mock_tarpes,
        delayline_dim="position",
        delayline_origin=100.31,
    )


def test_plot_movie_and_evolution_html_output(sample_data: xr.DataArray):
    result = plot_movie_and_evolution(sample_data, out="")
    assert isinstance(result, HTML)


def test_plot_movie_and_evolution_path_output(sample_data: xr.DataArray, tmp_path: Path):
    output_path = tmp_path / "test_movie.mp4"
    result = plot_movie_and_evolution(sample_data, out=output_path)
    assert isinstance(result, Path)
    assert result.exists()


def test_plot_movie_and_evolution_figsize(sample_data: xr.DataArray):
    result = plot_movie_and_evolution(sample_data, figsize=(12, 8), out="")
    assert isinstance(result, HTML)


def test_plot_movie_and_evolution_dark_bg(sample_data: xr.DataArray):
    result = plot_movie_and_evolution(sample_data, dark_bg=True, out="")
    assert isinstance(result, HTML)


def test_plot_movie_and_evolution_evolution_at(sample_data: xr.DataArray):
    result = plot_movie_and_evolution(sample_data, evolution_at=("phi", 0.0), out="")
    assert isinstance(result, HTML)
