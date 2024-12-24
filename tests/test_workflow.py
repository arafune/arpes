"""Test suite for arpes workflow module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from arpes.workflow import (
    DataProvider,
    consume_data,
    go_to_cwd,
    go_to_figures,
    go_to_workspace,
    publish_data,
    read_data,
    summarize_data,
)


@pytest.mark.skip
@patch("arpes.workflow._open_path")
def test_go_to_figures(mock_open_path: MagicMock) -> None:
    """Test that go_to_figures opens the correct path."""
    with patch("arpes.workflow.path_for_plot", return_value="figures"):
        go_to_figures()
        mock_open_path.assert_called_once_with(Path("figures"))


@patch("arpes.workflow._open_path")
def test_go_to_workspace(mock_open_path: MagicMock) -> None:
    """Test that go_to_workspace opens the workspace path from the config."""
    with patch("arpes.workflow.CONFIG", {"WORKSPACE": {"path": "workspace_path"}}):
        go_to_workspace()
        mock_open_path.assert_called_once_with(Path("workspace_path"))


@patch("arpes.workflow._open_path")
def test_go_to_cwd(mock_open_path: MagicMock) -> None:
    """Test that go_to_cwd opens the current working directory."""
    go_to_cwd()
    mock_open_path.assert_called_once_with(Path.cwd())


@patch("arpes.workflow.DataProvider.publish")
def test_publish_data(mock_publish: MagicMock) -> None:
    """Test that publish_data calls the publish method with the correct arguments."""
    with patch("arpes.workflow.CONFIG", {"WORKSPACE": {"path": "workspace_path"}}):
        publish_data("key", "data")
        mock_publish.assert_called_once_with("key", "data")


@patch("arpes.workflow.DataProvider.consume")
def test_read_data(mock_consume: MagicMock) -> None:
    """Test that read_data calls the consume method with subscribe set to False."""
    with patch("arpes.workflow.CONFIG", {"WORKSPACE": {"path": "workspace_path"}}):
        read_data("key")
        mock_consume.assert_called_once_with("key", subscribe=False)


@patch("arpes.workflow.DataProvider.consume")
def test_consume_data(mock_consume: MagicMock) -> None:
    """Test that consume_data calls the consume method with subscribe set to True."""
    with patch("arpes.workflow.CONFIG", {"WORKSPACE": {"path": "workspace_path"}}):
        consume_data("key")
        mock_consume.assert_called_once_with("key", subscribe=True)


@patch("arpes.workflow.DataProvider.summarize_clients")
def test_summarize_data(mock_summarize_clients: MagicMock) -> None:
    """Test that summarize_data calls the summarize_clients method with the correct key."""
    with patch("arpes.workflow.CONFIG", {"WORKSPACE": {"path": "workspace_path"}}):
        summarize_data("key")
        mock_summarize_clients.assert_called_once_with(key="key")


def test_data_provider_init() -> None:
    """Test the initialization of DataProvider."""
    with patch("arpes.workflow.Path.mkdir") as mock_mkdir:
        provider = DataProvider(Path("test_path"), "workspace_name")
        assert provider.path == Path("test_path/data_provider")
        assert provider.workspace_name == "workspace_name"
        mock_mkdir.assert_called()


@pytest.mark.skip
@patch("arpes.workflow.dill.load")
@patch("arpes.workflow.Path.open")
def test_data_provider_read_pickled(mock_dill_load: MagicMock) -> None:
    """Test that _read_pickled correctly loads pickled data."""
    provider = DataProvider(Path("test_path"))
    mock_dill_load.return_value = {"key": "value"}
    result = provider._read_pickled("test")
    assert result == {"key": "value"}


@pytest.mark.skip
@patch("arpes.workflow.dill.dump")
@patch("arpes.workflow.Path.open")
def test_data_provider_write_pickled(mock_open: MagicMock, mock_dill_dump: MagicMock) -> None:
    """Test that _write_pickled correctly dumps data to a pickle."""
    provider = DataProvider(Path("test_path"))
    provider._write_pickled("test", {"key": "value"})
    mock_dill_dump.assert_called_once_with({"key": "value"}, mock_open().__enter__())
