"""Unit tests for the arpes.endstations.registry module.

This test suite ensures complete coverage of the registry and lookup system
for ARPES endstation plugins. The key features tested include:

- Registering new endstation plugins via `add_endstation`.
- Resolving endstation classes using aliases via `endstation_from_alias`.
- Resolving principal plugin names via `endstation_name_from_alias`.
- Determining the correct plugin using `resolve_endstation`, including:
    - Resolution by direct class reference.
    - Resolution by alias.
    - Fallback mechanism when no endstation is provided.
    - Retry mechanism using `load_plugins` if resolution fails.

Mock classes (`DummyEndstation`, `DummyEndstation2`) are used to simulate real plugins,
and plugin loading behavior is patched where needed to simulate retry behavior.

"""

import pytest
import warnings
from unittest.mock import patch, MagicMock

from arpes.endstations import registry
from arpes.endstations.plugin.fallback import FallbackEndstation


class DummyEndstation:
    PRINCIPAL_NAME = "dummy"
    ALIASES = ["d", "dummy_alias"]


class DummyEndstation2:
    PRINCIPAL_NAME = "dummy2"
    ALIASES = ["d2"]


def test_add_endstation_and_lookup():
    registry._ENDSTATION_ALIASES.clear()

    registry.add_endstation(DummyEndstation)

    # Check all aliases are registered
    assert registry.endstation_from_alias("d") is DummyEndstation
    assert registry.endstation_from_alias("dummy_alias") is DummyEndstation
    assert registry.endstation_from_alias("dummy") is DummyEndstation

    # Check name resolution
    assert registry.endstation_name_from_alias("dummy_alias") == "dummy"


def test_duplicate_alias_ignored():
    registry._ENDSTATION_ALIASES.clear()

    registry.add_endstation(DummyEndstation)
    # Should not overwrite existing alias
    registry.add_endstation(DummyEndstation2)

    assert registry.endstation_from_alias("d") is DummyEndstation  # not DummyEndstation2
    assert registry.endstation_from_alias("dummy") is DummyEndstation
    assert registry.endstation_from_alias("d2") is DummyEndstation2


def test_resolve_endstation_direct_type():
    cls = registry.resolve_endstation(endstation=DummyEndstation)
    assert cls is DummyEndstation


def test_resolve_endstation_from_alias():
    registry._ENDSTATION_ALIASES.clear()
    registry.add_endstation(DummyEndstation)

    result = registry.resolve_endstation(location="d")
    assert result is DummyEndstation


def test_resolve_endstation_with_retry(monkeypatch):
    registry._ENDSTATION_ALIASES.clear()

    # Simulate plugin not found at first, but available after retry
    call_count = {"count": 0}

    def fake_load_plugins():
        call_count["count"] += 1
        registry.add_endstation(DummyEndstation)

    monkeypatch.setattr(registry, "load_plugins", fake_load_plugins)

    result = registry.resolve_endstation(location="d", retry=True)
    assert result is DummyEndstation
    assert call_count["count"] == 1


def test_resolve_endstation_fallback_warning():
    registry._ENDSTATION_ALIASES.clear()
    registry.add_endstation(FallbackEndstation)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = registry.resolve_endstation()
        assert result is FallbackEndstation
        assert len(w) == 1
        assert "Endstation not provided" in str(w[0].message)


def test_resolve_endstation_raises_error(monkeypatch):
    registry._ENDSTATION_ALIASES.clear()

    monkeypatch.setattr(registry, "load_plugins", lambda: None)  # skip plugin loading

    with pytest.raises(ValueError, match="Could not identify endstation"):
        registry.resolve_endstation(location="not_exist", retry=False)
