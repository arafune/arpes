import pytest
from unittest.mock import patch
from arpes.utilities.jupyter import get_full_notebook_information
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


# data for test
TEST_NOTEBOOKS = [
    {"id": "123", "title": "Sample Notebook", "content": "This is a test notebook."},
    {"id": "456", "title": "Another Notebook", "content": "This is another test notebook."},
]


def get_full_notebook_information_test():
    return TEST_NOTEBOOKS


def test_get_full_notebook_information():
    result = get_full_notebook_information_test()
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["id"] == "123"
    assert result[0]["title"] == "Sample Notebook"
    assert result[0]["content"] == "This is a test notebook."
    assert result[1]["id"] == "456"
    assert result[1]["title"] == "Another Notebook"
    assert result[1]["content"] == "This is another test notebook."


# Mock CONFIG dictionary
CONFIG = {
    "WORKSPACE": {"path": "", "name": ""},
    "CURRENT_CONTEXT": None,
    "ENABLE_LOGGING": True,
    "LOGGING_STARTED": False,
    "LOGGING_FILE": None,
}


def test_update_configuration(monkeypatch):
    from arpes.config import update_configuration

    # Mock HAS_LOADED and paths
    monkeypatch.setattr("arpes.config.HAS_LOADED", False)
    mock_path = MagicMock()
    monkeypatch.setattr("arpes.config.Path", mock_path)

    # Call the function
    update_configuration("user_path")

    # Assert that paths are set correctly
    mock_path.assert_called_with("user_path")
    assert mock_path.return_value.__truediv__.call_count == 2


def test_workspace_matches(monkeypatch):
    from arpes.config import workspace_matches

    # Mock Path.iterdir
    mock_iterdir = MagicMock(return_value=[Path("data"), Path("other")])
    monkeypatch.setattr("pathlib.Path.iterdir", mock_iterdir)

    # Test with a valid workspace
    assert workspace_matches("some_path") is True

    # Test with an invalid workspace
    mock_iterdir.return_value = [Path("other")]
    assert workspace_matches("some_path") is False


def test_attempt_determine_workspace(monkeypatch):
    from arpes.config import attempt_determine_workspace

    # Mock Path.cwd to return a specific path
    mock_cwd = MagicMock(return_value=Path("/mock/path"))
    monkeypatch.setattr("pathlib.Path.cwd", mock_cwd)

    # Mock workspace_matches to return True for the mocked path
    monkeypatch.setattr("arpes.config.workspace_matches", lambda x: x == Path("/mock/path"))

    # Mock CONFIG to ensure isolation
    monkeypatch.setattr("arpes.config.CONFIG", CONFIG)

    # Call the function
    attempt_determine_workspace()

    # Assert that CONFIG is updated correctly
    assert CONFIG["WORKSPACE"]["path"] == Path("/mock/path")
    assert CONFIG["WORKSPACE"]["name"] == "path"


def test_load_json_configuration(monkeypatch):
    from arpes.config import load_json_configuration

    # Mock Path.open to simulate reading a JSON file
    mock_open = MagicMock()
    mock_open.return_value.__enter__.return_value.read.return_value = '{"key": "value"}'
    monkeypatch.setattr("pathlib.Path.open", mock_open)

    # Mock json.load to return a specific dictionary
    monkeypatch.setattr("json.load", lambda x: {"key": "value"})

    # Mock CONFIG to ensure isolation
    monkeypatch.setattr("arpes.config.CONFIG", CONFIG)

    # Call the function
    load_json_configuration("config.json")

    # Assert that CONFIG is updated correctly
    assert CONFIG["key"] == "value"
