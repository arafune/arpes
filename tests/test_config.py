from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib as mpl
import pytest

import arpes
import arpes.config
from arpes.config import UseTex, setup_logging, use_tex
from arpes.setting import CONFIG

# Mock CONFIG dictionary
CONFIG = {
    "WORKSPACE": {"path": "", "name": ""},
    "CURRENT_CONTEXT": None,
    "ENABLE_LOGGING": True,
    "LOGGING_STARTED": False,
    "LOGGING_FILE": None,
}


@pytest.fixture
def mock_cwd():
    with patch("pathlib.Path.cwd") as mock:
        yield mock


@pytest.fixture
def mock_workspace_matches():
    with patch("arpes.config.workspace_matches") as mock:
        yield mock


@pytest.fixture
def mock_logging_exception():
    with patch("logging.exception") as mock:
        yield mock


def test_attempt_determine_workspace_found(mock_cwd, mock_workspace_matches):
    mock_cwd.return_value = Path("/mock/workspace")
    mock_workspace_matches.side_effect = [False, False, True]

    arpes.config.attempt_determine_workspace()

    assert arpes.config.CONFIG["WORKSPACE"]["path"] == Path("/")
    assert arpes.config.CONFIG["WORKSPACE"]["name"] == ""


def test_attempt_determine_workspace_not_found(mock_cwd, mock_workspace_matches):
    mock_cwd.return_value = Path("/mock/workspace")
    mock_workspace_matches.side_effect = [False, False, False]

    arpes.config.DATASET_PATH = Path("/mock/dataset")
    arpes.config.attempt_determine_workspace()

    assert arpes.config.CONFIG["WORKSPACE"]["path"] == Path("/mock/dataset")
    assert arpes.config.CONFIG["WORKSPACE"]["name"] == "dataset"


def test_workspace_matches(monkeypatch: pytest.MonkeyPatch):
    from arpes.config import workspace_matches

    # Mock Path.iterdir
    mock_iterdir = MagicMock(return_value=[Path("data"), Path("other")])
    monkeypatch.setattr("pathlib.Path.iterdir", mock_iterdir)

    # Test with a valid workspace
    assert workspace_matches("some_path") is True

    # Test with an invalid workspace
    mock_iterdir.return_value = [Path("other")]
    assert workspace_matches("some_path") is False


def test_attempt_determine_workspace(monkeypatch: pytest.MonkeyPatch):
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
    assert arpes.config.CONFIG["WORKSPACE"]["path"] == Path("/mock/path")
    assert arpes.config.CONFIG["WORKSPACE"]["name"] == "path"


def test_load_json_configuration(monkeypatch: pytest.MonkeyPatch):
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
    assert arpes.config.CONFIG["key"] == "value"


def test_has_loaded_short_circuit():
    setup_logging()
    assert arpes.config.CONFIG["LOGGING_STARTED"] is False


def test_import_error():
    with patch("IPython.core.getipython.get_ipython", side_effect=ImportError):
        setup_logging()
        assert arpes.config.CONFIG["LOGGING_STARTED"] is False


def test_get_ipython_none():
    with patch("IPython.core.getipython.get_ipython", return_value=None):
        setup_logging()
        assert arpes.config.CONFIG["LOGGING_STARTED"] is False


@pytest.fixture
def original_rc_params():
    # Save original rcParams and restore after test
    original = mpl.rcParams["text.usetex"]
    yield original
    mpl.rcParams["text.usetex"] = original


@pytest.fixture
def mock_settings():
    with patch("arpes.config.SETTINGS", new_callable=dict) as mock_settings:
        mock_settings["use_tex"] = False
        yield mock_settings


def test_use_tex_function(original_rc_params, mock_settings):
    # Test enabling TeX
    use_tex(rc_text_should_use=True)
    assert mpl.rcParams["text.usetex"] is True
    assert mock_settings["use_tex"] is True

    # Test disabling TeX
    use_tex(rc_text_should_use=False)
    assert mpl.rcParams["text.usetex"] is False
    assert mock_settings["use_tex"] is False


def test_use_tex_context_manager(original_rc_params, mock_settings):
    mock_settings["use_tex"] = False
    with UseTex(use_tex=True):
        assert mpl.rcParams["text.usetex"] is True
        assert mock_settings["use_tex"] is True
    # Ensure settings are restored after exiting the context
    assert mpl.rcParams["text.usetex"] == original_rc_params
    assert mock_settings["use_tex"] is False
