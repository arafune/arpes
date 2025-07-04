# tests/test_config_manager.py

import json
import warnings
from pathlib import Path

import matplotlib as mpl
import pytest

from arpes.configuration.manager import ConfigManager


def test_use_tex():
    cm = ConfigManager()
    cm.use_tex(enable=True)
    assert cm.settings["use_tex"] is True
    assert mpl.rcParams["text.usetex"] is True

    cm.use_tex(enable=False)
    assert not cm.settings["use_tex"]
    assert not mpl.rcParams["text.usetex"]


def test_update_config_from_json(tmp_path):
    cm = ConfigManager()
    json_path = tmp_path / "config.json"
    json_path.write_text(json.dumps({"LOGGING_STARTED": True}))
    cm.update_config_from_json(str(json_path))
    assert cm.config["LOGGING_STARTED"] is True


def test_workspace_detection(monkeypatch, tmp_path):
    workspace = tmp_path / "myspace"
    (workspace / "data").mkdir(parents=True)
    monkeypatch.chdir(workspace)
    cm = ConfigManager()
    cm.detect_workspace()
    assert cm.workspace_path == workspace
    assert cm.workspace_name == "myspace"


def test_workspace_not_found(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)
    cm = ConfigManager()
    cm.detect_workspace()
    with pytest.raises(ValueError):
        cm.enter_workspace("does_not_exist")


def test_local_config_not_found(monkeypatch):
    cm = ConfigManager()
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with warnings.catch_warnings(record=True) as w:
        cm.load_local_config("nonexistent_config_module")
        assert any("could not find" in str(wi.message).lower() for wi in w)


def test_workspace_properties(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    monkeypatch.chdir(tmp_path)
    cm = ConfigManager()
    cm.detect_workspace()
    assert isinstance(cm.workspace_path, Path)
    assert isinstance(cm.workspace_name, str)
