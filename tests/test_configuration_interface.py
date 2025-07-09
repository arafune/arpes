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


def test_loggin_file():
    """Return the path to the data directory under the workspace."""
    assert interface.get_logging_file() is None


def test_get_logging_started():
    """Return the path to the data directory under the workspace."""
    assert interface.get_logging_started() is False


def test_get_figure_path():
    """Return the path to the figure path under the workspace."""
    assert interface.get_figure_path() is None


def test_get_configu_value():
    """Return the path to the figure path under the workspace."""
    work_space = interface.get_config_value("WORKSPACE")
    assert isinstance(work_space, dict)


def test_get_setting_value():
    """Return the path to the figure path under the workspace."""
    interactive = interface.get_settings_value("interactive")
    assert isinstance(interactive, dict)


def test_get_full_config():
    """Return the full configuration dictionary."""
    config = interface.get_full_config()
    assert isinstance(config, dict)
    assert "WORKSPACE" in config
    assert isinstance(config["WORKSPACE"], dict)


def test_get_full_setting():
    """Return the full settings dictionary."""
    settings = interface.get_full_settings()
    assert isinstance(settings, dict)
    assert "interactive" in settings
    assert isinstance(settings["interactive"], dict)
