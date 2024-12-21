"""This module contains unit tests for the plotting utility functions."""

import datetime
import warnings
from pathlib import Path
from unittest.mock import patch

import pytest
import arpes
from arpes.plotting.utils import path_for_plot


@pytest.fixture
def mock_datetime():
    """Fixture to mock the current datetime."""
    with patch("arpes.plotting.utils.datetime") as mock_datetime:
        mock_datetime.datetime.now.return_value = datetime.datetime(
            2023,
            10,
            10,
            tzinfo=datetime.UTC,
        )
        yield mock_datetime


@pytest.mark.skip
def test_path_for_plot_with_workspace(mock_datetime):
    """Test path_for_plot function when a workspace is configured."""
    with (
        patch(
            "arpes.config.CONFIG",
            {"WORKSPACE": {"path": "/mock/workspace", "name": "test_workspace"}},
        ),
        patch("arpes.plotting.utils.FIGURE_PATH", None),
        patch("arpes.config.DATASET_PATH", Path("/mock/dataset")),
    ):
        mock_datetime.datetime.now.return_value = datetime.datetime(
            2023,
            10,
            10,
            tzinfo=datetime.UTC,
        )
        expected_path = Path("/mock/workspace/figures/test_workspace/2023-10-10/test_plot.png")
        result = path_for_plot("test_plot.png")
        assert result == expected_path


def test_path_for_plot_without_workspace():
    """Test path_for_plot function when no workspace is configured."""
    with (
        patch("arpes.plotting.utils.CONFIG", {"WORKSPACE": {}}),
        patch("arpes.plotting.utils.Path.cwd", return_value=Path("/current/directory")),
    ):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = path_for_plot("test_plot.png")
            assert result == Path("/current/directory/test_plot.png")
            assert any(item.category == UserWarning for item in w)


def test_path_for_plot_exception():
    """Test path_for_plot function when an exception occurs."""
    with (
        patch("arpes.plotting.utils.CONFIG", {"WORKSPACE": {"path": "/mock/workspace"}}),
        patch("arpes.plotting.utils.FIGURE_PATH", None),
        patch("arpes.plotting.utils.Path.cwd", return_value=Path("/current/directory")),
        patch("arpes.plotting.utils.Path.mkdir", side_effect=Exception),
    ):
        result = path_for_plot("test_plot.png")
        assert result == Path("/current/directory/test_plot.png")
