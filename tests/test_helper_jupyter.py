"""Unit test for helper.jupyter module."""

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch
from urllib.error import HTTPError

import pytest
from ipykernel.zmqshell import ZMQInteractiveShell
from IPython.core.interactiveshell import InteractiveShell

# Import functions from your provided module, assuming it's arpes.helper.jupyter
# Adjust the import path if your file structure is different
from arpes.helper import jupyter
from arpes.helper.jupyter import generate_logfile_path


# Mock the setup_logger to prevent actual logging file creation during tests
@pytest.fixture(autouse=True)
def mock_setup_logger():
    with patch("arpes.helper.jupyter.setup_logger") as mock_logger:
        mock_logger.return_value = MagicMock()
        yield


def test_get_tqdm_notebook_environment():
    """Test get_tqdm when in a Jupyter notebook environment."""
    with patch("arpes.helper.jupyter.get_ipython") as mock_get_ipython:
        mock_get_ipython.return_value = ZMQInteractiveShell()
        tqdm_func = jupyter.get_tqdm()
        from tqdm.notebook import tqdm as notebook_tqdm

        assert tqdm_func == notebook_tqdm


def test_get_tqdm_cli_environment():
    """Test get_tqdm when not in a Jupyter notebook environment."""
    with patch("arpes.helper.jupyter.get_ipython") as mock_get_ipython:
        mock_get_ipython.return_value = None
        tqdm_func = jupyter.get_tqdm()
        from tqdm import tqdm as cli_tqdm

        assert tqdm_func == cli_tqdm


@patch("arpes.helper.jupyter.ipykernel.get_connection_file")
@patch("arpes.helper.jupyter.serverapp.list_running_servers")
@patch("arpes.helper.jupyter.urllib.request.urlopen")
@patch("arpes.helper.jupyter.json.load")
def test_get_full_notebook_information_success(
    mock_json_load,
    mock_urlopen,
    mock_list_running_servers,
    mock_get_connection_file,
):
    """Test successful retrieval of notebook information."""
    mock_get_connection_file.return_value = "/tmp/kernel-12345.json"
    mock_list_running_servers.return_value = [
        {
            "url": "http://localhost:8888/",
            "token": "test-token",
            "password": False,
            "pid": 123,
            "port": 8888,
            "root_dir": "/tmp",
            "secure": False,
            "sock": "",
            "version": "1.0",
            "base_url": "/",
        },
    ]
    mock_urlopen.return_value.__enter__.return_value = MagicMock()
    mock_json_load.return_value = [
        {
            "id": "session-id-1",
            "path": "path/to/notebook.ipynb",
            "name": "notebook.ipynb",
            "type": "notebook",
            "kernel": {"id": "12345", "name": "python3"},
            "notebook": {"path": "path/to/notebook.ipynb", "name": "notebook.ipynb"},
        },
    ]

    info = jupyter.get_full_notebook_information()
    assert info is not None
    assert info["server"]["url"] == "http://localhost:8888/"
    assert info["session"]["kernel"]["id"] == "12345"


@patch("arpes.helper.jupyter.ipykernel.get_connection_file")
def test_get_full_notebook_information_no_connection_file(mock_get_connection_file):
    """Test get_full_notebook_information when ipykernel.get_connection_file fails."""
    mock_get_connection_file.side_effect = RuntimeError("No connection file")
    info = jupyter.get_full_notebook_information()
    assert info is None


@patch("arpes.helper.jupyter.ipykernel.get_connection_file")
@patch("arpes.helper.jupyter.serverapp.list_running_servers")
def test_get_full_notebook_information_no_servers(
    mock_list_running_servers,
    mock_get_connection_file,
):
    """Test get_full_notebook_information when no servers are running."""
    mock_get_connection_file.return_value = "/tmp/kernel-12345.json"
    mock_list_running_servers.return_value = []
    info = jupyter.get_full_notebook_information()
    assert info is None


@patch("arpes.helper.jupyter.ipykernel.get_connection_file")
@patch("arpes.helper.jupyter.serverapp.list_running_servers")
@patch("arpes.helper.jupyter.urllib.request.urlopen")
def test_get_full_notebook_information_http_error(
    mock_urlopen,
    mock_list_running_servers,
    mock_get_connection_file,
):
    """Test get_full_notebook_information when HTTPError occurs."""
    mock_get_connection_file.return_value = "/tmp/kernel-12345.json"
    mock_list_running_servers.return_value = [
        {"url": "http://localhost:8888/", "token": "test-token", "password": False},
    ]
    mock_urlopen.side_effect = HTTPError("url", 404, "Not Found", {}, None)
    info = jupyter.get_full_notebook_information()
    assert info is None


@patch("arpes.helper.jupyter.ipykernel.get_connection_file")
@patch("arpes.helper.jupyter.serverapp.list_running_servers")
@patch("arpes.helper.jupyter.urllib.request.urlopen")
@patch("arpes.helper.jupyter.json.load")
def test_get_full_notebook_information_no_matching_kernel_id(
    mock_json_load,
    mock_urlopen,
    mock_list_running_servers,
    mock_get_connection_file,
):
    """Test get_full_notebook_information when kernel ID does not match."""
    mock_get_connection_file.return_value = "/tmp/kernel-nomatch.json"
    mock_list_running_servers.return_value = [
        {"url": "http://localhost:8888/", "token": "test-token", "password": False},
    ]
    mock_urlopen.return_value.__enter__.return_value = MagicMock()
    mock_json_load.return_value = [
        {
            "id": "session-id-1",
            "path": "path/to/notebook.ipynb",
            "name": "notebook.ipynb",
            "type": "notebook",
            "kernel": {"id": "12345", "name": "python3"},
            "notebook": {"path": "path/to/notebook.ipynb", "name": "notebook.ipynb"},
        },
    ]
    info = jupyter.get_full_notebook_information()
    assert info is None


@patch("arpes.helper.jupyter.get_full_notebook_information")
def test_get_notebook_name_found(mock_get_full_notebook_information):
    """Test get_notebook_name when notebook information is available."""
    mock_get_full_notebook_information.return_value = {
        "session": {"notebook": {"name": "MyTestNotebook.ipynb"}},
    }
    name = jupyter.get_notebook_name()
    assert name == "MyTestNotebook"


@patch("arpes.helper.jupyter.get_full_notebook_information")
def test_get_notebook_name_not_found(mock_get_full_notebook_information):
    """Test get_notebook_name when no notebook information is available."""
    mock_get_full_notebook_information.return_value = None
    name = jupyter.get_notebook_name()
    assert name == ""


@patch("arpes.helper.jupyter.get_notebook_name", return_value="TestNotebook")
@patch("datetime.datetime")
def test_generate_logfile_path_named_notebook(mock_datetime, mock_get_notebook_name):
    """Test generate_logfile_path with a named notebook."""
    mock_now = datetime(2025, 7, 7, 20, 0, 0, tzinfo=UTC)
    mock_datetime.now.return_value = mock_now
    mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

    path = jupyter.generate_logfile_path()
    expected_path = Path("logs/TestNotebook_2025-07-07_20-00-00.log")
    assert path == expected_path


@patch("arpes.helper.jupyter.get_notebook_name", return_value="")
@patch("datetime.datetime")
def test_generate_logfile_path_unnamed_notebook(mock_datetime, mock_get_notebook_name):
    """Test generate_logfile_path with an unnamed notebook."""
    mock_now = datetime(2025, 7, 7, 20, 0, 0, tzinfo=UTC)
    mock_datetime.now.return_value = mock_now
    mock_datetime.side_effect = lambda *args, **kw: datetime.datetime(*args, **kw)

    path = jupyter.generate_logfile_path()
    expected_path = Path("logs/unnamed_2025-07-07_20-00-00.log")
    assert path == expected_path


@patch("arpes.helper.jupyter.get_ipython")
def test_get_recent_history_success(mock_get_ipython):
    """Test get_recent_history with a successful history retrieval."""
    mock_ipython = MagicMock(spec=InteractiveShell)
    mock_ipython.history_manager.get_tail.return_value = [
        (1, 1, "line 1"),
        (1, 2, "line 2"),
        (1, 3, "line 3"),
    ]
    mock_get_ipython.return_value = mock_ipython
    history = jupyter.get_recent_history(n_items=3)
    assert history == ["line 1", "line 2", "line 3"]
    mock_ipython.history_manager.get_tail.assert_called_once_with(n=3, include_latest=True)


@patch("arpes.helper.jupyter.get_ipython")
def test_get_recent_history_no_ipython(mock_get_ipython):
    """Test get_recent_history when get_ipython returns None."""
    mock_get_ipython.return_value = None
    history = jupyter.get_recent_history()
    assert history == ["No accessible history."]


@patch("arpes.helper.jupyter.get_ipython")
def test_get_recent_history_attribute_error(mock_get_ipython):
    """Test get_recent_history when an AttributeError occurs."""
    mock_ipython = MagicMock(spec=InteractiveShell)
    mock_ipython.history_manager.get_tail.side_effect = AttributeError("No history manager")
    mock_get_ipython.return_value = mock_ipython
    history = jupyter.get_recent_history()
    assert history == ["No accessible history."]


@patch("arpes.helper.jupyter.get_ipython")
@patch("arpes.configuration.interface.get_logging_started", return_value=True)
@patch("arpes.configuration.interface.get_logging_file", return_value="test_log.log")
@patch("builtins.open", new_callable=mock_open, read_data=b"log line 1\nlog line 2\nlog line 3\n")
@patch("pathlib.Path.open")
def test_get_recent_logs_success(
    mock_path_open,
    mock_open_builtins,
    mock_get_logging_file,
    mock_get_logging_started,
    mock_get_ipython,
):
    """Test get_recent_logs with successful log retrieval."""
    mock_ipython = MagicMock(spec=InteractiveShell)
    mock_ipython.history_manager.get_tail.return_value = [(1, 1, "final cell output")]
    mock_get_ipython.return_value = mock_ipython

    # Mock file.seek to allow it to be called
    mock_file = MagicMock()
    mock_file.readlines.return_value = [b"log line 1\n", b"log line 2\n", b"log line 3\n"]
    mock_open_builtins.return_value.__enter__.return_value = mock_file
    mock_path_open.return_value.__enter__.return_value = mock_file

    logs = jupyter.get_recent_logs(n_bytes=100)
    assert logs == ["log line 1\n", "log line 2\n", "log line 3\n", "final cell output"]
    mock_file.seek.assert_called_with(-100, 2)


@patch("arpes.helper.jupyter.get_ipython")
@patch("arpes.configuration.interface.get_logging_started", return_value=False)
@patch("arpes.configuration.interface.get_logging_file")
def test_get_recent_logs_logging_not_started(
    mock_get_logging_file,
    mock_get_logging_started,
    mock_get_ipython,
):
    """Test get_recent_logs when logging is not started."""
    mock_get_ipython.return_value = MagicMock(spec=InteractiveShell)
    logs = jupyter.get_recent_logs()
    assert logs == ["No logging available. Logging is only available inside Jupyter."]
    mock_get_logging_file.assert_not_called()


@patch("arpes.helper.jupyter.get_ipython", return_value=None)
@patch("arpes.configuration.interface.get_logging_started", return_value=True)
def test_get_recent_logs_no_ipython(mock_get_logging_started, mock_get_ipython):
    """Test get_recent_logs when get_ipython returns None."""
    logs = jupyter.get_recent_logs()
    assert logs == ["No logging available. Logging is only available inside Jupyter."]


@patch("arpes.helper.jupyter.get_ipython")
@patch("arpes.configuration.interface.get_logging_started", return_value=True)
@patch(
    "arpes.configuration.interface.get_logging_file",
    return_value=None,
)  # Simulate get_logging_file returning None or invalid type
def test_get_recent_logs_invalid_logging_file(
    mock_get_logging_file,
    mock_get_logging_started,
    mock_get_ipython,
):
    """Test get_recent_logs when logging file is invalid."""
    mock_get_ipython.return_value = MagicMock(spec=InteractiveShell)
    logs = jupyter.get_recent_logs()
    assert logs == ["No logging available. Logging is only available inside Jupyter."]


def test_generate_logfile_path_with_name():
    mock_now = datetime(2023, 12, 31, 23, 59, 59, tzinfo=UTC)

    with patch("arpes.helper.jupyter.get_notebook_name", return_value="analysis"):
        with patch("arpes.helper.jupyter.datetime") as mock_datetime:
            mock_datetime.datetime.now.return_value = mock_now
            mock_datetime.UTC = UTC
            result = generate_logfile_path()
            assert result == Path("logs/analysis_2023-12-31_23-59-59.log")


def test_generate_logfile_path_unnamed():
    # Check for fallback to 'unnamed'
    with patch("arpes.helper.jupyter.get_notebook_name", return_value=None):
        with patch("arpes.helper.jupyter.datetime") as mock_datetime:
            mock_now = datetime(2024, 1, 1, 1, 2, 3, tzinfo=UTC)
            mock_datetime.datetime.now.return_value = mock_now
            mock_datetime.UTC = UTC
            path = generate_logfile_path()
            assert path.name.startswith("unnamed_2024-01-01_01-02-03")
            assert path.parent.name == "logs"
