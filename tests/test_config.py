from pathlib import Path
from unittest.mock import patch

import pytest

import arpes.config


@pytest.fixture
def mock_cwd():
    with patch("arpes.config.Path.cwd") as mock:
        yield mock


@pytest.fixture
def mock_workspace_matches():
    with patch("arpes.config.workspace_matches") as mock:
        yield mock


@pytest.fixture
def mock_logging_exception():
    with patch("arpes.config.logging.exception") as mock:
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


def test_attempt_determine_workspace_exception(
    mock_cwd, mock_workspace_matches, mock_logging_exception,
):
    mock_cwd.side_effect = Exception("Test Exception")

    arpes.config.DATASET_PATH = Path("/mock/dataset")
    arpes.config.attempt_determine_workspace()

    mock_logging_exception.assert_called_once_with("Exception occurs")
    assert arpes.config.CONFIG["WORKSPACE"]["path"] == Path("/mock/dataset")
    assert arpes.config.CONFIG["WORKSPACE"]["name"] == "dataset"