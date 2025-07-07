# test_configuration_interface.
# Copyright (c) 2025 R. Arafune, All Rights Reserved.
#

from pathlib import Path

from arpes.configuration import interface


def test_get_workspace_path():
    """Return the absolute path to the current workspace directory."""
    assert interface.get_workspace_path() == Path()


def test_get_workspace_name():
    """Return the absolute path to the current workspace directory."""
    assert interface.get_workspace_name() == ""


def test_get_data_path():
    """Return the path to the data directory under the workspace."""
    assert interface.get_data_path() is None


def test_get_dataset_path():
    """Return the path to the data directory under the workspace."""
    assert interface.get_dataset_path() is None


def test_get_figure_path():
    """Return the path to the data directory under the workspace."""
    assert interface.get_dataset_path() is None


def test_loggin_file():
    """Return the path to the data directory under the workspace."""
    assert interface.get_logging_file() is None


def test_get_logging_started():
    """Return the path to the data directory under the workspace."""
    assert interface.get_logging_started() is False
