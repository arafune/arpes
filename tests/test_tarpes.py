"""Unit test for tarpes.py."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from matplotlib.figure import Figure

from arpes.analysis import tarpes
from arpes.plotting.movie import (
    _replace_after_col,
    _replace_after_row,
    plot_movie,
    plot_movie_and_evolution,
)


@pytest.fixture
def sample_data(mock_tarpes: list[xr.DataArray]):
    return tarpes.build_crosscorrelation(
        mock_tarpes,
        delayline_dim="position",
        delayline_origin=100.31,
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


def test_as_movie(sample_data: xr.DataArray) -> None:
    """Test xarray.G.as_movie."""
    anim = sample_data.G.as_movie()
    assert type(anim) is HTML
    anim_out = sample_data.G.as_movie(out="test.mp4")
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


def test_plot_movie_and_evolution_html_output(sample_data: xr.DataArray):
    result = plot_movie_and_evolution(sample_data, out=None)
    assert isinstance(result, HTML)


def test_plot_movie_and_evolution_path_output(sample_data: xr.DataArray, tmp_path: Path):
    output_path = tmp_path / "test_movie.mp4"
    result = plot_movie_and_evolution(sample_data, out=output_path)
    assert isinstance(result, Path)
    assert result.exists()


def test_plot_movie_and_evolution_figsize(sample_data: xr.DataArray):
    result = plot_movie_and_evolution(sample_data, figsize=(12, 8), out=None)
    assert isinstance(result, HTML)


def test_plot_movie_and_evolution_dark_bg(sample_data: xr.DataArray):
    result = plot_movie_and_evolution(sample_data, dark_bg=True, out=None)
    assert isinstance(result, HTML)


def test_plot_movie_and_evolution_evolution_at(sample_data: xr.DataArray):
    result = plot_movie_and_evolution(sample_data, evolution_at=("phi", 0.0), out=None)
    assert isinstance(result, HTML)


def test_plot_movie_and_evolution_snapshot(sample_data: xr.DataArray):
    result = plot_movie_and_evolution(sample_data, evolution_at=("phi", (0.0, 0.05)), out=0.1)
    assert isinstance(result, Figure)


def test_plot_movie_and_evolution_FuncAnimation(sample_data: xr.DataArray):
    result = plot_movie_and_evolution(sample_data, evolution_at=("phi", 0.0), out=...)
    assert isinstance(result, FuncAnimation)


@pytest.fixture
def another_sample_data():
    """Fixture to create sample ARPES data."""
    time = np.linspace(0, 10, 5)  # 5 time points
    x = np.linspace(-5, 5, 50)  # 50 x points
    y = np.linspace(-5, 5, 50)  # 50 y points
    data = np.random.random((50, 50, 5))  # Random 3D data
    return xr.DataArray(
        data,
        dims=["eV", "phi", "delay"],
        coords={"eV": y, "phi": x, "delay": time},
        attrs={"subtracted": False},
    )


def test_plot_movie_and_evolution_html(another_sample_data: xr.DataArray):
    """Test that the function returns an HTML object."""
    result = plot_movie_and_evolution(
        data=another_sample_data,
        out=None,  # Return HTML
    )
    from IPython.core.display import HTML

    assert isinstance(result, HTML)


def test_plot_movie_and_evolution_funcanimation(another_sample_data: xr.DataArray):
    """Test that the function returns a FuncAnimation object."""
    result = plot_movie_and_evolution(
        data=another_sample_data,
        out=...,  # Return FuncAnimation
    )
    assert isinstance(result, FuncAnimation)


def test_plot_movie_and_evolution_figure(another_sample_data: xr.DataArray):
    """Test that the function returns a Figure object."""
    result = plot_movie_and_evolution(
        data=another_sample_data,
        out=0.0,  # Return snapshot as Figure
    )
    assert isinstance(result, Figure)


def test_plot_movie_and_evolution_save(tmp_path: Path, another_sample_data: xr.DataArray):
    """Test that the function saves the animation to a file."""
    output_path = tmp_path / "test_animation.mp4"
    result = plot_movie_and_evolution(
        data=another_sample_data,
        out=output_path,  # Save animation
    )
    assert isinstance(result, Path)
    assert result.exists()


def test_plot_movie_and_evolution_missing_attrs(another_sample_data: xr.DataArray):
    """Test that the function handles missing attributes gracefully."""
    del another_sample_data.attrs["subtracted"]
    result = plot_movie_and_evolution(data=another_sample_data, out=None)
    from IPython.core.display import HTML

    assert isinstance(result, HTML)


def test_plot_movie_and_evolution_empty_data():
    """Test that the function raises an error for empty data."""
    empty_data = xr.DataArray(
        np.empty((0, 0, 0)),
        dims=["eV", "phi", "delay"],
        coords={"eV": [], "phi": [], "delay": []},
    )
    with pytest.raises(KeyError, match="\"not all values found in index 'phi'\""):
        plot_movie_and_evolution(data=empty_data, out=None)


def test_plot_movie_and_evolution_custom_fig_ax(another_sample_data: xr.DataArray):
    """Test that the function works with custom figure and axes."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    result = plot_movie_and_evolution(data=another_sample_data, fig_ax=(fig, ax), out=None)
    from IPython.core.display import HTML

    assert isinstance(result, HTML)


def test_plot_movie_and_evolution_with_labels(another_sample_data: xr.DataArray):
    """Test that the function correctly applies custom labels."""
    labels = ("X-axis Label", "Time-axis Label", "Y-axis Label")
    result = plot_movie_and_evolution(
        data=another_sample_data,
        labels=labels,
        out=None,  # Return HTML
    )
    from IPython.core.display import HTML

    assert isinstance(result, HTML)

    # Verify that the labels are correctly applied
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) == 3
    assert ax[0].get_xlabel() == labels[0]
    assert ax[0].get_ylabel() == labels[2]
    assert ax[1].get_xlabel() == labels[1]


def test_plot_movie_and_evolution_is_subtracted_true(another_sample_data: xr.DataArray):
    """Test when data.S.is_subtracted is True."""
    another_sample_data.attrs["subtracted"] = True
    result = plot_movie_and_evolution(
        data=another_sample_data,
        out=None,  # Return HTML
    )
    from IPython.core.display import HTML

    assert isinstance(result, HTML)

    # Verify that the colormap and vmin/vmax are set correctly
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) == 3
    for artist in ax[0].collections + ax[1].collections:
        assert artist.get_cmap().name == "RdBu_r"
        assert artist.get_clim()[0] < 0  # vmin
        assert artist.get_clim()[1] > 0  # vmax


def test_plot_movie_and_evolution_is_subtracted_true_with_labels(another_sample_data: xr.DataArray):
    """Test when data.S.is_subtracted is True and labels are provided."""
    another_sample_data.attrs["subtracted"] = True
    labels = ("Custom X-axis", "Custom Y-axis")
    result = plot_movie(
        data=another_sample_data,
        labels=labels,
        out=None,  # Return HTML
    )
    from IPython.core.display import HTML

    assert isinstance(result, HTML)

    # Verify that the colormap and vmin/vmax are set correctly
    fig = plt.gcf()
    ax = fig.axes
    assert len(ax) == 2

    for artist in ax[0].collections:
        assert artist.get_cmap().name == "RdBu_r"
        clim = artist.get_clim()
        assert clim[0] < 0  # vmin
        assert clim[1] > 0  # vmax

    # Verify that the labels are correctly applied
    assert ax[0].get_xlabel() == labels[0]
    assert ax[0].get_ylabel() == labels[1]
