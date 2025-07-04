from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from arpes.helper.jupyter import generate_logfile_path


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
