"""Unit test for trapezoid correction."""

import numpy as np
import xarray as xr
from numba import typed, types
from numpy.testing import assert_allclose

from arpes.correction.trapezoid import (
    _corners,
    _corners_typed_dict,
    _is_all_dicts,
    _is_all_floats,
    _phi_to_phi,
    _phi_to_phi_forward,
    trapezoid,
)

corners = [
    {"eV": 9, "phi": -0.2686923867293084},
    {"eV": 9, "phi": 0.17569017175537366},
    {"eV": 10, "phi": -0.2771474516235519},
    {"eV": 10, "phi": 0.14535341826841874},
]


def test__is_all_dicts() -> None:
    assert _is_all_dicts(corners)


def test__is_all_floats() -> None:
    corners = [
        -0.2686923867293084,
        0.17569017175537366,
        -0.2771474516235519,
        0.14535341826841874,
    ]
    assert _is_all_floats(corners)


def test__corners() -> None:
    corners_ = _corners(corners)
    assert len(corners_) == 4
    assert corners_["lower_left"] == {"eV": 9, "phi": -0.2686923867293084}


def test__corners_typed_dict() -> None:
    typed_dict_corners = _corners_typed_dict(corners)
    assert typed_dict_corners["lower_left"]["eV"] == 9


def test_trapezoid(dataarray_cut2: xr.DataArray) -> None:
    corrected = trapezoid(dataarray_cut2, corners, from_trapezoid=False)
    assert corrected["phi"].shape == (645,)
    assert corrected.coords["phi"].min().item() == -0.27785279531285406
    assert corrected.coords["phi"].max().item() == 0.1763885405977243


def test_phi_to_phi():
    """Tests the _phi_to_phi function with sample inputs."""
    energy = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    phi = np.array([10.0, 15.0, 20.0], dtype=np.float64)
    phi_out = np.zeros_like(phi)

    corner_type = types.DictType(keyty=types.unicode_type, valty=types.float64)
    corners = typed.Dict.empty(key_type=types.unicode_type, value_type=corner_type)
    upper_left = typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    upper_left["phi"] = 5.0
    upper_left["eV"] = 0.0

    lower_left = typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    lower_left["phi"] = 10.0
    lower_left["eV"] = 1.0

    upper_right = typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    upper_right["phi"] = 25.0
    upper_right["eV"] = 0.0

    lower_right = typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    lower_right["phi"] = 30.0
    lower_right["eV"] = 1.0

    corners["upper_left"] = upper_left
    corners["lower_left"] = lower_left
    corners["upper_right"] = upper_right
    corners["lower_right"] = lower_right

    rectangle_phis = [10.0, 20.0]

    _phi_to_phi(energy, phi, phi_out, corners, rectangle_phis)

    expected_phi_out = np.array([5.0, 17.5, 30.0], dtype=np.float64)

    assert_allclose(phi_out, expected_phi_out, atol=1e-6)


def test_phi_to_phi_forward():
    """Tests the _phi_to_phi_forward function with sample inputs."""
    energy = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    phi = np.array([5.0, 17.5, 30.0], dtype=np.float64)
    phi_out = np.zeros_like(phi)

    corner_type = types.DictType(keyty=types.unicode_type, valty=types.float64)
    corners = typed.Dict.empty(key_type=types.unicode_type, value_type=corner_type)

    upper_left = typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    upper_left["phi"] = 5.0
    upper_left["eV"] = 0.0

    lower_left = typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    lower_left["phi"] = 10.0
    lower_left["eV"] = 1.0

    upper_right = typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    upper_right["phi"] = 25.0
    upper_right["eV"] = 0.0

    lower_right = typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    lower_right["phi"] = 30.0
    lower_right["eV"] = 1.0

    corners["upper_left"] = upper_left
    corners["lower_left"] = lower_left
    corners["upper_right"] = upper_right
    corners["lower_right"] = lower_right

    rectangle_phis = [10.0, 20.0]

    _phi_to_phi_forward(energy, phi, phi_out, corners, rectangle_phis)

    expected_phi_out = np.array([10.0, 15.0, 20.0], dtype=np.float64)

    assert_allclose(phi_out, expected_phi_out, atol=1e-6)
