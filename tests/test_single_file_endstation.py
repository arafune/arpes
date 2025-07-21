from pathlib import Path

import pytest

from arpes.endstations import single_file_endstation
from arpes.endstations.single_file_endstation import SingleFileEndstation


class DummyEndstation(SingleFileEndstation):
    pass


@pytest.fixture
def scan_desc_with_path(tmp_path: Path):
    file_path = tmp_path / "datafile.dat"
    file_path.write_text("dummy data")
    return {"path": str(file_path)}


@pytest.fixture
def scan_desc_with_file(tmp_path: Path):
    file_path = tmp_path / "datafile2.dat"
    file_path.write_text("dummy data")
    return {"file": str(file_path)}


def test_resolve_frame_locations_with_existing_path(scan_desc_with_path: dict) -> None:
    endstation = DummyEndstation()
    result = endstation.resolve_frame_locations(scan_desc_with_path)
    assert len(result) == 1
    assert Path(scan_desc_with_path["path"]) in result


def test_resolve_frame_locations_with_existing_file(scan_desc_with_file: dict) -> None:
    endstation = DummyEndstation()
    result = endstation.resolve_frame_locations(scan_desc_with_file)
    assert len(result) == 1
    assert Path(scan_desc_with_file["file"]) in result


def test_resolve_frame_locations_file_not_found(monkeypatch):
    monkeypatch.setattr(single_file_endstation, "get_data_path", lambda: None)

    endstation = DummyEndstation()
    scan_desc = {"path": "nonexistent.dat"}

    with pytest.raises(RuntimeError):
        endstation.resolve_frame_locations(scan_desc)


def test_resolve_frame_locations_none() -> None:
    endstation = DummyEndstation()
    with pytest.raises(ValueError):
        endstation.resolve_frame_locations(None)
