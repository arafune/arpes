from arpes.utilities.jupyter import get_full_notebook_information
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
