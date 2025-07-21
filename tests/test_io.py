"""Unit test of io module in aryspes."""

from pathlib import Path

import pytest
import xarray as xr

from arpes import io
from arpes.io import load_example_data


def test_load_example_data_all():
    for key in io.DATA_EXAMPLES:
        data = io.load_example_data(key)
        assert isinstance(data, xr.Dataset)


def test_example_data_object():
    ed = io.example_data
    assert isinstance(ed.cut, xr.Dataset)
    assert isinstance(ed.map, xr.Dataset)
    assert isinstance(ed.photon_energy, xr.Dataset)
    assert isinstance(ed.nano_xps, xr.Dataset)
    assert isinstance(ed.temperature_dependence, xr.Dataset)
    assert isinstance(ed.cut2, xr.Dataset)
    assert isinstance(ed.cut3, xr.Dataset)
    assert isinstance(ed.map2, xr.Dataset)
    assert isinstance(ed.t_arpes, list)
    assert isinstance(ed.t_arpes[0], xr.DataArray)


def test_easy_pickle_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(io, "get_workspace_path", lambda: tmp_path)
    data = {"foo": 123}
    io.easy_pickle(data, "test")
    loaded = io.easy_pickle("test")
    assert loaded == data


def test_list_pickles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(io, "get_workspace_path", lambda: tmp_path)
    data = {"bar": 456}
    io.easy_pickle(data, "sample_pickle")
    pickles = io.list_pickles()
    assert "sample_pickle" in pickles


def test_file_for_pickle_creates_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(io, "get_workspace_path", lambda: tmp_path)
    path = io.file_for_pickle("test_pickle")
    assert "test_pickle.pickle" in path
    assert Path(path).parent.exists()


def test_load_and_save_pickle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(io, "get_workspace_path", lambda: tmp_path)
    name = "abc"
    obj = {"x": 1}
    io.save_pickle(obj, name)
    result = io.load_pickle(name)
    assert result == obj


@pytest.mark.skip
def test_stitch_invalid_input():
    with pytest.raises(AssertionError):
        io._df_or_list_to_files("not a list")


@pytest.mark.skip
def test_stitch_empty_list():
    with pytest.raises(ValueError):
        io.stitch([], "temp")


@pytest.mark.skip
def test_load_data_warns_without_location(tmp_path: Path):
    file = Path(__file__).parent / "../src/arpes/example_data/cut.fits"
    assert file.exists()
    with pytest.warns(UserWarning, match="You should provide a location"):
        ds = io.load_data(file=file)
        assert isinstance(ds, xr.Dataset)


def test_load_data_deprecated_path_integer_warning():
    with pytest.warns(DeprecationWarning):
        with pytest.raises(Exception):
            # This will not resolve unless a plugin uses int paths
            io.load_data(123456789)


def test_load_example_data_invalid_key():
    with pytest.raises(KeyError):
        io.load_example_data("invalid_name")


def test_load_scan_plugin_called(monkeypatch: pytest.MonkeyPatch):
    class DummyEndstation:
        @staticmethod
        def find_first_file(number):
            return f"mocked_path_{number}"

        def load(self, scan_desc, **kwargs):
            return xr.Dataset({"mock": xr.DataArray([1, 2, 3])})

    monkeypatch.setattr(io, "resolve_endstation", lambda **kwargs: DummyEndstation)
    scan_desc = {"file": "123"}
    result = io.load_scan(scan_desc)
    assert isinstance(result, xr.Dataset)


def test_load_scan_with_attr(monkeypatch: pytest.MonkeyPatch):
    class DummyEndstation:
        def load(self, scan_desc, **kwargs):
            return xr.Dataset().assign_attrs(test_attr="ok")

        @staticmethod
        def find_first_file(n):
            return f"dummy{n}"

    monkeypatch.setattr(io, "resolve_endstation", lambda **kwargs: DummyEndstation)
    scan_desc = {"file": "123", "note": {"key": "value"}}
    result = io.load_scan(scan_desc)
    assert isinstance(result, xr.Dataset)


@pytest.mark.skip
def test_stitch_with_attrs(monkeypatch: pytest.MonkeyPatch):
    class DummyEndstation:
        def load(self, scan_desc, **kwargs):
            return xr.Dataset().assign_attrs(t=scan_desc["file"])

    monkeypatch.setattr(io, "load_data", lambda file: DummyEndstation().load({"file": file}))
    result = io.stitch(["a", "b", "c"], attr_or_axis=["x", "y", "z"], built_axis_name="axis")
    assert isinstance(result, xr.Dataset)
    assert "axis" in result.coords


def test_load_example_raises_kye_error() -> None:
    msg = "Could not find requested example_name: cut0.*"
    with pytest.raises(KeyError, match=msg):
        load_example_data("cut0")
