"""Mocks the analysis environment and provides data fixutres for tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import pytest

import arpes.config
import arpes.endstations
from arpes.io import example_data
from tests.utils import cache_loader

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import xarray as xr

    from arpes._typing import ScanInfo, WorkSpaceType


class Expected(TypedDict, total=False):
    """TypedDict for expected."""

    scan_info: ScanInfo


class Scenario(TypedDict, total=False):
    """TypedDict for SCENARIO."""

    file: str


@pytest.fixture
def dataset_cut() -> xr.Dataset:
    """A fixture for loading Dataset."""
    return example_data.cut


@pytest.fixture
def dataarray_cut() -> xr.DataArray:
    """A fixture for loading DataArray."""
    return example_data.cut.spectrum


@pytest.fixture
def dataset_map() -> xr.Dataset:
    """A fixture for loading Dataset."""
    return example_data.map


@pytest.fixture
def dataarray_map() -> xr.DataArray:
    """A fixture for loading DataArray."""
    return example_data.map.spectrum


@pytest.fixture
def xps_map() -> xr.Dataset:
    """A fixture for loading example_data.xps."""
    return example_data.nano_xps


@pytest.fixture
def hv_map() -> xr.Dataset:
    """A fixture for loading photonenergy dependence."""
    return example_data.photon_energy


@pytest.fixture
def dataset_cut2() -> xr.Dataset:
    """A fixture for loading Dataset."""
    return example_data.cut2


@pytest.fixture
def dataarray_cut2() -> xr.DataArray:
    """A fixture for loading Dataset."""
    return example_data.cut2.spectrum


@pytest.fixture
def dataset_temperature_dependence() -> xr.Dataset:
    """A fixture for loading Dataset (temperature_dependence)."""
    return example_data.temperature_dependence


@pytest.fixture
def mock_tarpes() -> list[xr.DataArray]:
    """A fixture for making a mock mimicking the tarpes measurements."""
    return example_data.t_arpes


@dataclass
class Sandbox:
    """Mocks some configuration calls which should not touch the FS during tests."""

    with_workspace: Callable
    load: Callable


SCAN_FIXTURE_LOCATIONS = {
    "basic/main_chamber_cut_0.fits": "ALG-MC",
    "basic/main_chamber_map_1.fits": "ALG-MC",
    "basic/main_chamber_PHONY_2.fits": "ALG-MC",
    "basic/main_chamber_PHONY_3.fits": "ALG-MC",
    "basic/SToF_PHONY_4.fits": "ALG-SToF",
    "basic/SToF_PHONY_5.fits": "ALG-SToF",
    "basic/SToF_PHONY_6.fits": "ALG-SToF",
    "basic/SToF_PHONY_7.fits": "ALG-SToF",
    "basic/MERLIN_8.pxt": "BL4",
    "basic/MERLIN_9.pxt": "BL4",
    "basic/MERLIN_10_S001.pxt": "BL4",
    "basic/MERLIN_11_S001.pxt": "BL4",
    "basic/MAESTRO_12.fits": "BL7",
    "basic/MAESTRO_13.fits": "BL7",
    "basic/MAESTRO_PHONY_14.fits": "BL7",
    "basic/MAESTRO_PHONY_15.fits": "BL7",
    "basic/MAESTRO_16.fits": "BL7",
    "basic/MAESTRO_nARPES_focus_17.fits": "BL7-nano",
    "basic/Uranos_cut.pxt": "Uranos",
    "basic/DSNP_UMCS_cut.xy": "DSNP_UMCS",
}


@pytest.fixture
def sandbox_configuration() -> Iterator[Sandbox]:
    """Generates a sandboxed configuration of the ARPES data analysis suite."""
    resources_dir = Path.cwd() / "tests" / "resources"

    def set_workspace(name: str) -> None:
        workspace: WorkSpaceType = {
            "path": resources_dir / "datasets" / name,
            "name": name,
        }
        arpes.config.CONFIG["WORKSPACE"] = workspace

    def load(path: str) -> xr.DataArray | xr.Dataset:
        assert path in SCAN_FIXTURE_LOCATIONS
        pieces = path.split("/")
        set_workspace(pieces[0])
        return cache_loader.load_test_scan(
            str(Path(path)),
            location=SCAN_FIXTURE_LOCATIONS[path],
        )

    arpes.config.update_configuration(user_path=resources_dir)
    sandbox = Sandbox(
        with_workspace=set_workspace,
        load=load,
    )
    arpes.config.load_plugins()
    yield sandbox
    arpes.config.CONFIG["WORKSPACE"] = None
    arpes.endstations._ENDSTATION_ALIASES = {}
