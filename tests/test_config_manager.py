"""Unit tests for the ConfigManager and related runtime configuration utilities in PyARPES.

This test suite covers the complete functionality of the `ConfigManager` class,
which serves as the centralized configuration, logging, and workspace management system
within the PyARPES framework.

Coverage goals:
    - 100% branch and line coverage of `arpes.configuration.manager`
    - Explicit validation of config updates, workspace management,
      logging setup, plugin loading, and local configuration overrides
    - Environment-sensitive behavior such as Jupyter/marimo detection

Test categories:
    - LaTeX rendering control via `use_tex()` and `is_using_tex()`
    - Config loading from JSON files
    - Workspace detection and switching (`detect_workspace`, `enter_workspace`, etc.)
    - Custom workspace path setting and derived path verification
    - Local configuration overrides via `load_local_config()`
    - Plugin loading (mocked)
    - Safe reinitialization logic with `initialize(force=False)`
    - Environment detection: `should_initialize_automatically()`

Tools & Techniques:
    - `pytest` framework
    - Built-in fixtures: `tmp_path`, `monkeypatch`
    - Mocking: `importlib`, `warnings`, method injection
    - No actual filesystem side effects; all file operations occur in temp paths

Note:
    - Plugin loading and logging are mocked/stubbed to ensure isolation.
    - The tests are suitable for CI pipelines and are designed to be deterministic.

"""

import json
import warnings
from pathlib import Path
from types import ModuleType

import matplotlib as mpl
import pytest

from arpes.configuration.manager import ConfigManager, should_initialize_automatically


def test_use_tex():
    """Test enabling and disabling TeX rendering."""
    cm = ConfigManager()
    cm.use_tex(enable=True)
    assert cm.settings["use_tex"] is True
    assert mpl.rcParams["text.usetex"] is True

    cm.use_tex(enable=False)
    assert not cm.settings["use_tex"]
    assert not mpl.rcParams["text.usetex"]


def test_is_using_tex():
    """Test the is_using_tex method."""
    cm = ConfigManager()
    cm.use_tex(enable=True)
    assert cm.is_using_tex() is True
    cm.use_tex(enable=False)
    assert cm.is_using_tex() is False


def test_update_config_from_json(tmp_path):
    """Test updating config via JSON file."""
    cm = ConfigManager()
    json_path = tmp_path / "config.json"
    json_path.write_text(json.dumps({"LOGGING_STARTED": True}))
    cm.update_config_from_json(str(json_path))
    assert cm.config["LOGGING_STARTED"] is True


def test_workspace_detection(monkeypatch, tmp_path):
    """Test auto-detection of a workspace."""
    workspace = tmp_path / "myspace"
    (workspace / "data").mkdir(parents=True)
    monkeypatch.chdir(workspace)
    cm = ConfigManager()
    cm.detect_workspace()
    assert cm.workspace_path == workspace
    assert cm.workspace_name == "myspace"


def test_workspace_not_found(tmp_path, monkeypatch):
    """Test behavior when trying to enter non-existent workspace."""
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)
    cm = ConfigManager()
    cm.detect_workspace()
    with pytest.raises(ValueError):
        cm.enter_workspace("does_not_exist")


def test_workspace_properties(tmp_path, monkeypatch):
    """Test workspace_path and workspace_name properties."""
    (tmp_path / "Data").mkdir()
    monkeypatch.chdir(tmp_path)
    cm = ConfigManager()
    cm.detect_workspace()
    assert isinstance(cm.workspace_path, Path)
    assert isinstance(cm.workspace_name, str)


def test_set_workspace_paths(tmp_path):
    """Test explicitly setting workspace and derived paths."""
    base = tmp_path / "workspace"
    base.mkdir()
    cm = ConfigManager()
    cm.set_workspace(base)
    assert cm.workspace_path == base
    assert cm.data_path == base / "data"
    assert cm.dataset_path == base / "datasets"
    assert cm.figure_path == base / "figures"


def test_exit_workspace_calls_detect(monkeypatch):
    """Test that exit_workspace calls detect_workspace."""
    cm = ConfigManager()
    called = {"detected": False}
    monkeypatch.setattr(cm, "detect_workspace", lambda: called.update({"detected": True}))
    cm.exit_workspace()
    assert called["detected"]


def test_is_workspace(tmp_path):
    """Test _is_workspace logic."""
    cm = ConfigManager()
    path = tmp_path / "wspace"
    path.mkdir()
    assert not cm._is_workspace(path)
    (path / "data").mkdir()
    assert cm._is_workspace(path)


def test_local_config_not_found(monkeypatch):
    """Test local config loading with mock module."""
    cm = ConfigManager()
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with warnings.catch_warnings(record=True) as w:
        cm.load_local_config("nonexistent_config_module")
        assert any("could not find" in str(wi.message).lower() for wi in w)


def test_load_local_config_not_found(monkeypatch):
    """Test local config loading when module does not exist."""
    cm = ConfigManager()
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with warnings.catch_warnings(record=True) as w:
        cm.load_local_config("nonexistent_config_module")
        assert any("could not find" in str(wi.message).lower() for wi in w)


def test_load_local_config_found(monkeypatch):
    """Test local config loading with mock module."""
    cm = ConfigManager()

    class DummyLoader:
        def exec_module(self, mod):
            mod.CONFIG_OVERRIDES = {"LOGGING_STARTED": True}
            mod.SETTINGS_OVERRIDES = {"use_tex": True}
            mod.custom_init = lambda cm: cm.config.update({"custom": True})

    dummy_spec = type("Spec", (), {"loader": DummyLoader()})()
    monkeypatch.setattr("importlib.util.find_spec", lambda name: dummy_spec)
    monkeypatch.setattr("importlib.util.module_from_spec", lambda spec: ModuleType("dummy_mod"))

    cm.load_local_config("mock_config")
    assert cm.config["custom"]
    assert cm.config["LOGGING_STARTED"]
    assert cm.settings["use_tex"]


def test_initialize_sets_initialized(monkeypatch):
    """Test initialize runs only once unless forced."""
    cm = ConfigManager()

    monkeypatch.setitem(cm.config, "_initialized", False)
    monkeypatch.setattr(cm, "setup_logging", lambda: cm.config.update({"log_called": True}))
    monkeypatch.setattr(cm, "load_plugins", lambda: cm.config.update({"plugins_loaded": True}))
    monkeypatch.setattr(cm, "load_local_config", lambda: cm.config.update({"local_loaded": True}))
    monkeypatch.setattr(
        "importlib.import_module", lambda name: cm.config.update({"xarray_loaded": True}),
    )

    cm.initialize()
    assert cm.config["_initialized"]
    assert cm.config["log_called"]
    assert cm.config["plugins_loaded"]
    assert cm.config["local_loaded"]
    assert cm.config["xarray_loaded"]


def test_initialize_skips_when_already_initialized():
    """Test that initialize exits early when already initialized."""
    cm = ConfigManager()
    cm.config["_initialized"] = True
    cm.initialize()  # should do nothing
    # just make sure it doesn't crash


def test_should_initialize_automatically(monkeypatch):
    """Test auto-initialization flag depending on Jupyter or marimo."""
    monkeypatch.setattr("arpes.configuration.manager.is_jupyter", lambda: True)
    monkeypatch.setattr("arpes.configuration.manager.is_marimo", lambda: False)
    assert should_initialize_automatically()

    monkeypatch.setattr("arpes.configuration.manager.is_jupyter", lambda: False)
    monkeypatch.setattr("arpes.configuration.manager.is_marimo", lambda: True)
    assert should_initialize_automatically()

    monkeypatch.setattr("arpes.configuration.manager.is_marimo", lambda: False)
    assert not should_initialize_automatically()


def test_workspace_properties(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)
    cm = ConfigManager()
    cm.detect_workspace()
    assert isinstance(cm.workspace_path, Path)
    assert isinstance(cm.workspace_name, str)


def test_exit_workspace(monkeypatch, tmp_path):
    """Test resetting workspace using exit_workspace()."""
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)
    cm = ConfigManager()
    cm.exit_workspace()
    assert cm.workspace_path == tmp_path


def test_is_workspace_variants(tmp_path):
    """Test workspace detection with 'Data' instead of 'data'."""
    (tmp_path / "Data").mkdir()
    cm = ConfigManager()
    assert cm._is_workspace(tmp_path) is True


def test_is_using_tex():
    """Test is_using_tex() reflects matplotlib state."""
    cm = ConfigManager()
    cm.use_tex(enable=True)
    assert cm.is_using_tex() is True
    cm.use_tex(enable=False)
    assert cm.is_using_tex() is False


def test_update_config_from_json_full(tmp_path):
    """Test update_config_from_json with actual Path.open usage."""
    cm = ConfigManager()
    config_dict = {"LOGGING_STARTED": True}
    json_path = tmp_path / "cfg.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(config_dict, f)
    cm.update_config_from_json(str(json_path))
    assert cm.config["LOGGING_STARTED"] is True


def test_set_workspace(tmp_path):
    """Test manually setting workspace updates all paths correctly."""
    base = tmp_path / "newspace"
    base.mkdir()
    cm = ConfigManager()
    cm.set_workspace(base)
    assert cm.workspace_path == base
    assert cm.data_path == base / "data"
    assert cm.dataset_path == base / "datasets"
    assert cm.figure_path == base / "figures"


def test_enter_workspace_success(monkeypatch, tmp_path):
    """Test entering an existing workspace after detecting it."""
    base = tmp_path / "mainspace"
    base.mkdir()
    (base / "data").mkdir()
    monkeypatch.chdir(base)
    cm = ConfigManager()
    cm.detect_workspace()  # set WORKSPACE
    sibling = base.parent / "otherspace"
    sibling.mkdir()
    (sibling / "data").mkdir()
    cm.enter_workspace("otherspace")
    assert cm.workspace_name == "otherspace"
    assert cm.workspace_path == sibling


def test_detect_workspace_fallback(tmp_path, monkeypatch):
    """Test detect_workspace fallback when no valid workspace found."""
    dummy_root = tmp_path / "nonspace"
    dummy_root.mkdir()
    monkeypatch.chdir(dummy_root)
    cm = ConfigManager()
    cm.detect_workspace()
    # fallback: should return cwd as workspace
    assert cm.workspace_path == dummy_root
    assert cm.workspace_name == "nonspace"
