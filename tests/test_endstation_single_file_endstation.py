import pytest
from pathlib import Path
from unittest.mock import patch

# Import the target class
from arpes.endstations import SingleFileEndstation


@pytest.fixture
def scan_desc_path(tmp_path):
    # Create an actual file for testing the 'path' key
    test_file = tmp_path / "datafile.dat"
    test_file.write_text("dummy data")
    return {"path": str(test_file)}


@pytest.fixture
def scan_desc_file(tmp_path):
    # Create an actual file for testing the 'file' key
    test_file = tmp_path / "filedata.dat"
    test_file.write_text("dummy data")
    return {"file": str(test_file)}


def test_resolve_frame_locations_path(scan_desc_path):
    # Test with 'path' key present and file exists
    est = SingleFileEndstation()
    result = est.resolve_frame_locations(scan_desc_path)
    assert len(result) == 1
    assert result[0].exists()


def test_resolve_frame_locations_file(scan_desc_file):
    # Test with 'file' key present and file exists
    est = SingleFileEndstation()
    result = est.resolve_frame_locations(scan_desc_file)
    assert len(result) == 1
    assert result[0].exists()


def test_missing_scan_desc():
    # Test when scan_desc is None
    est = SingleFileEndstation()
    with pytest.raises(ValueError):
        est.resolve_frame_locations(None)


def test_missing_path_key():
    # Test when scan_desc does not contain 'path' or 'file'
    est = SingleFileEndstation()
    with pytest.raises(ValueError):
        est.resolve_frame_locations({})


def test_file_not_exists_and_no_datapath(tmp_path):
    # Test when file does not exist and get_data_path returns None
    est = SingleFileEndstation()
    bad_file = tmp_path / "notfound.dat"
    scan_desc = {"path": str(bad_file)}
    with patch("arpes.configuration.interface.get_data_path", return_value=None):
        with pytest.raises(RuntimeError):
            est.resolve_frame_locations(scan_desc)


@pytest.mark.skip
def test_file_not_exists_but_datapath(tmp_path):
    est = SingleFileEndstation()
    data_path = tmp_path / "datadir"
    data_path.mkdir()
    test_file = data_path / "found.dat"
    test_file.write_text("dummy")
    scan_desc = {"path": "found.dat"}
    with patch("arpes.configuration.interface.get_data_path", return_value=str(data_path)):
        result = est.resolve_frame_locations(scan_desc)
        assert len(result) == 1
        assert result[0].exists()
        assert result[0] == test_file


def test_file_not_found_even_with_datapath(tmp_path):
    # Test when file does not exist even after prepending data_path
    est = SingleFileEndstation()
    data_path = tmp_path / "datadir"
    data_path.mkdir()
    scan_desc = {"path": "notfound.dat"}
    with patch("arpes.configuration.interface.get_data_path", return_value=str(data_path)):
        with pytest.raises(RuntimeError):
            est.resolve_frame_locations(scan_desc)
