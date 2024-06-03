"""Unit test for conversion.forward."""

from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
from arpes.fits.fit_models import AffineBroadenedFD, QuadraticModel
from arpes.fits.utilities import broadcast_model
from arpes.utilities.conversion.forward import (
    convert_coordinate_forward,
    convert_through_angular_pair,
    convert_through_angular_point,
)

if TYPE_CHECKING:
    from collections.abc import Hashable


@pytest.fixture()
def energy_corrected(dataarray_map: xr.DataArray) -> xr.DataArray:
    """A fixture for loading DataArray."""
    fmap = dataarray_map
    cut = fmap.sum("theta", keep_attrs=True).sel(eV=slice(-0.2, 0.1), phi=slice(-0.25, 0.3))
    fit_results = broadcast_model(AffineBroadenedFD, cut, "phi")
    edge = QuadraticModel().guess_fit(fit_results.results.F.p("fd_center")).eval(x=fmap.phi)
    energy_corrected = fmap.G.shift_by(edge, shift_axis="eV", by_axis="phi")
    energy_corrected.attrs["energy_notation"] = "Binding"
    return energy_corrected


def test_convert_through_angular_point(energy_corrected: xr.DataArray) -> None:
    """Test for convert_through_anggular_pair.

    Taken from converting-to-kspace.ipynb
    """
    test_point: dict[Hashable, float] = {
        "phi": -0.13,
        "theta": -0.1,
        "eV": 0,
    }
    cut = convert_through_angular_point(
        energy_corrected,
        test_point,
        {"ky": np.linspace(-1, 1, 400)},
        {"kx": np.linspace(-0.02, 0.02, 10)},
    ).sel(eV=0, method="nearest")
    np.testing.assert_almost_equal(
        cut.values[-5:],
        np.array([2154.96241992, 2146.42062808, 2137.87654636, 2134.43809544, 2141.61500199]),
    )


def test_convert_through_angular_pair(energy_corrected: xr.DataArray) -> None:
    """Test for convert_through_anggular_pair.

    Taken from converting-to-kspace.
    """
    p1: dict[Hashable, float] = {
        "phi": 0.055,
        "theta": -0.013,
        "eV": 0,
    }
    p2: dict[Hashable, float] = {
        "phi": -0.09,
        "theta": -0.18,
        "eV": 0,
    }
    kp1 = convert_coordinate_forward(energy_corrected, p1)
    assert kp1 == {"kx": -0.027115300158778416, "ky": -0.022266815310293564}
    kp2 = convert_coordinate_forward(energy_corrected, p2)
    assert kp2 == {"kx": 0.9003614742745181, "ky": -0.7552690787473395}
    cut = convert_through_angular_pair(
        energy_corrected,
        p1,
        p2,
        {"kx": np.linspace(-0, 0, 400)},  # interpolate from p1 to p2 only
        {"ky": np.linspace(-0.02, 0.02, 10)},  # take 20 milli inv ang. perpendicular
    )
    np.testing.assert_almost_equal(
        cut.sel(eV=0.0, method="nearest").values[:5],
        np.array([2595.24789228, 2597.43947449, 2599.12997009, 2600.55552029, 2601.81521164]),
    )
