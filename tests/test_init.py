"""Tests for top-level initialization in arpes/__init__.py."""

import importlib
import sys
from unittest.mock import patch


def test_manual_initialize_calls_get_config_manager():
    """Test that `arpes.initialize()` calls `get_config_manager()`."""
    if "arpes" in sys.modules:
        del sys.modules["arpes"]

    import arpes

    with patch("arpes.__init__.get_config_manager") as mock_get_config_manager:
        importlib.reload(arpes)
        arpes.initialize()


def test_auto_initialize_triggers_in_interactive_env(monkeypatch):
    """Test that automatic initialization triggers in Jupyter/marimo."""
    if "arpes" in sys.modules:
        del sys.modules["arpes"]

    with (
        patch("arpes.configuration.should_initialize_automatically", return_value=True),
        patch("arpes.configuration.get_config_manager") as mock_get_config_manager,
    ):
        import arpes

        importlib.reload(arpes)
        mock_get_config_manager.assert_called_once()


def test_auto_initialize_skipped_in_script_env(monkeypatch):
    """Test that automatic initialization is skipped in non-interactive environments."""
    if "arpes" in sys.modules:
        del sys.modules["arpes"]

    with (
        patch("arpes.configuration.should_initialize_automatically", return_value=False),
        patch("arpes.configuration.get_config_manager") as mock_get_config_manager,
    ):
        import arpes

        importlib.reload(arpes)
        mock_get_config_manager.assert_not_called()


def test_version_exposed():
    """Test that `arpes.__version__` is available."""
    import arpes

    assert isinstance(arpes.__version__, str)
