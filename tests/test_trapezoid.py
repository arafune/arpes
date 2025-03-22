"""Unit test for trapezoid correction."""


import xarray as xr

from arpes.correction.trapezoid import (
    _corners,
    _corners_typed_dict,
    _is_all_dicts,
    _is_all_floats,
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
