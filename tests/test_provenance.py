"""Unit tests for the arpes.provenance module."""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from arpes import provenance as prov


def make_dataarray(*, with_id: bool = True):
    da = xr.DataArray(np.random.rand(5, 5), dims=["x", "y"])
    if with_id:
        da.attrs["id"] = "test-id"
    return da


def test_attach_id():
    da = make_dataarray(with_id=False)
    prov.attach_id(da)
    assert "id" in da.attrs
    assert isinstance(da.attrs["id"], str)


def test_provenance_from_file(monkeypatch: pytest.MonkeyPatch) -> None:
    da = make_dataarray(with_id=False)

    monkeypatch.setattr(prov, "get_recent_history", lambda _: ["code1", "code2"])
    prov.provenance_from_file(da, "file.dat", {"what": "loaded"})
    assert "provenance" in da.attrs
    assert da.attrs["provenance"]["file"] == "file.dat"


def test_update_provenance_decorator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prov, "get_recent_history", lambda _: ["code3"])

    @prov.update_provenance("smoothing")
    def myfunc(data):
        return data + 1

    da = make_dataarray()
    out = myfunc(da)
    assert "provenance" in out.attrs
    assert out.attrs["provenance"]["record"]["what"] == "smoothing"


def test_update_provenance_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prov, "get_recent_history", lambda _: ["code3"])

    @prov.update_provenance("noop")
    def myfunc(data):
        return data

    da = make_dataarray()
    out = myfunc(da)
    assert out is da
    assert "provenance" not in out.attrs or "record" not in out.attrs.get("provenance", {})


def test_save_plot_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(prov, "get_recent_history", lambda _: ["ctx1"])
    monkeypatch.setattr(prov, "get_workspace_name", lambda: str(tmp_path.name))

    file_path = tmp_path / "plot.png"

    @prov.save_plot_provenance
    def plot_fn(data):
        file_path.write_text("dummy plot")
        return str(file_path)

    da = make_dataarray()
    da.attrs["provenance"] = {"what": "plotted"}
    result = plot_fn(da)
    assert Path(result).exists()
    provenance_path = str(result) + ".provenance.json"
    with Path(provenance_path).open() as f:
        prov_data = json.load(f)
    assert prov_data["args"][0]["what"] == "plotted"


def test_provenance_single_parent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(prov, "get_recent_history", lambda _: ["ctx"])
    child = make_dataarray(with_id=False)
    parent = make_dataarray()
    prov.provenance(child, parent, {"what": "filtered"})
    assert "provenance" in child.attrs
    assert child.attrs["provenance"]["parent_id"] == parent.attrs["id"]


def test_provenance_duplicate_warning(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(prov, "get_recent_history", lambda _: ["ctx"])
    parent = make_dataarray()
    child = parent.copy(deep=True)
    child.attrs["id"] = parent.attrs["id"]
    with pytest.warns(UserWarning, match="Duplicate id"):
        prov.provenance(child, parent, {"what": "copy"})


def test_provenance_multiple_parents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prov, "get_recent_history", lambda _: ["ctx"])
    parent1 = make_dataarray()
    parent2 = make_dataarray()
    parent1.attrs["provenance"] = {"what": "p1"}
    parent2.attrs["provenance"] = {"what": "p2"}
    child = make_dataarray(with_id=False)
    child.attrs["id"] = "unique-child-id"
    prov.provenance_multiple_parents(child, [parent1, parent2], {"what": "mix"})
    prov_info = child.attrs["provenance"]
    assert "parents_provenance" in prov_info
    assert len(prov_info["parent_id"]) == 2


def test_provenance_multiple_parents_single(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(prov, "get_recent_history", lambda _: ["ctx"])
    parent = make_dataarray()
    parent.attrs["provenance"] = {"what": "only"}
    child = make_dataarray(with_id=False)
    prov.provenance_multiple_parents(child, parent, {"what": "single-parent-wrap"})
    assert isinstance(child.attrs["provenance"]["parent_id"], list)
