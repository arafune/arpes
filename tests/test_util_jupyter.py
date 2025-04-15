from datetime import UTC, date, datetime, time
from pathlib import Path
from unittest.mock import patch

from arpes.utilities.jupyter import generate_logfile_path

# Mock CONFIG dictionary
CONFIG = {
    "WORKSPACE": {"path": "", "name": ""},
    "CURRENT_CONTEXT": None,
    "ENABLE_LOGGING": True,
    "LOGGING_STARTED": False,
    "LOGGING_FILE": None,
}


def test_generate_logfile_path_with_name():
    fixed_date = date(2023, 12, 31)
    fixed_time = time(23, 59, 59)
    mock_now = datetime(2023, 12, 31, 23, 59, 59, tzinfo=UTC)

    with patch("arpes.utilities.jupyter.get_notebook_name", return_value="analysis"):
        with patch("arpes.utilities.jupyter.datetime") as mock_datetime:
            mock_datetime.datetime.now.return_value = mock_now
            mock_datetime.UTC = UTC
            result = generate_logfile_path()
            assert result == Path("logs/analysis_2023-12-31_23-59-59.log")


def test_generate_logfile_path_unnamed():
    # Check for fallback to 'unnamed'
    with patch("arpes.utilities.jupyter.get_notebook_name", return_value=None):
        with patch("arpes.utilities.jupyter.datetime") as mock_datetime:
            mock_now = datetime(2024, 1, 1, 1, 2, 3, tzinfo=UTC)
            mock_datetime.datetime.now.return_value = mock_now
            mock_datetime.UTC = UTC
            path = generate_logfile_path()
            assert path.name.startswith("unnamed_2024-01-01_01-02-03")
            assert path.parent.name == "logs"
