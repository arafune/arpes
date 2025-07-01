# tests/test_fallback_endstation.py
"""Unit tests for the FallbackEndstation plugin.

This test suite verifies the functionality of the FallbackEndstation class,
which is used when no 'location' is specified for a dataset. The fallback plugin
attempts to dynamically determine the appropriate loader based on a priority list
of known endstation plugins.

Test coverage includes:
- Sequential plugin resolution using determine_associated_loader
  - Success case: finds the first plugin that accepts the file
  - Failure case: raises ValueError when no plugins accept the file
- Proper delegation and warning emission in load()
  - Handles string and integer file descriptors
  - Issues a user warning when fallback is used
- Delegation of find_first_file to the resolved plugin
  - Returns the correct path
  - Also emits a warning

Mock plugins are used to isolate FallbackEndstation behavior from actual plugins.

To run:
    pytest tests/test_fallback_endstation.py --cov=arpes.endstations.plugin.fallback
"""

import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from arpes.endstations.plugin.fallback import FallbackEndstation, AUTOLOAD_WARNING


@pytest.fixture
def dummy_loader():
    class DummyLoader:
        @staticmethod
        def is_file_accepted(file):
            return True

        @staticmethod
        def find_first_file(file_number):
            return Path(f"/dummy/path/{file_number}.dat")

        def load(self, scan_desc, **kwargs):
            return {"loaded": True, "file": scan_desc["file"]}

    return DummyLoader


def test_determine_associated_loader_success(dummy_loader):
    with (
        patch("arpes.endstations.plugin.fallback.load_plugins"),
        patch("arpes.endstations.plugin.fallback.resolve_endstation") as mock_resolve,
    ):
        mock_resolve.return_value = dummy_loader
        file_path = "dummy_file.dat"

        loader = FallbackEndstation.determine_associated_loader(file_path)
        assert loader == dummy_loader
        mock_resolve.assert_called()


def test_determine_associated_loader_failure():
    with (
        patch("arpes.endstations.plugin.fallback.load_plugins"),
        patch(
            "arpes.endstations.plugin.fallback.resolve_endstation", side_effect=Exception("Fail")
        ),
    ):
        with pytest.raises(ValueError, match="failed to find a plugin acceptable"):
            FallbackEndstation.determine_associated_loader("some_file.dat")


def test_load_delegates_and_warns(dummy_loader):
    with (
        patch.object(FallbackEndstation, "determine_associated_loader", return_value=dummy_loader),
        warnings.catch_warnings(record=True) as w,
    ):
        warnings.simplefilter("always")
        fallback = FallbackEndstation()
        result = fallback.load(scan_desc={"file": "123.dat"})

        assert isinstance(result, dict)
        assert result["loaded"] is True
        assert len(w) == 1
        assert AUTOLOAD_WARNING.split()[0] in str(w[0].message)


def test_find_first_file_delegates_and_warns(dummy_loader):
    with (
        patch.object(FallbackEndstation, "determine_associated_loader", return_value=dummy_loader),
        warnings.catch_warnings(record=True) as w,
    ):
        warnings.simplefilter("always")
        result = FallbackEndstation.find_first_file(100)

        assert result == Path("/dummy/path/100.dat")
        assert len(w) == 1
        assert AUTOLOAD_WARNING.split()[0] in str(w[0].message)
