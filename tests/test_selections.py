"""Test suite for selections utilities."""

import pytest
import numpy as np
import xarray as xr

from arpes.utilities.selections import (
    ravel_from_mask,
    select_disk,
    select_disk_mask,
    unravel_from_mask,
    _radius,
)

rng = np.random.default_rng()


def test__if_radius_is_None(
    phi_values: xr.DataArray,
) -> None:
    radius = _radius(
        points={"phi": phi_values},
        radius=None,
    )
    assert radius == {"phi": 0.02}


def test__if_radius_is_array(phi_values: xr.DataArray) -> None:
    with pytest.raises(TypeError, match="radius should be a float, dictionary or None"):
        _radius(
            points={"phi": phi_values},
            radius=[0.02],
        )


def test_ravel_from_mask() -> None:
    """Test ravel_from_mask function."""
    data = xr.DataArray(rng.random((4, 4)), dims=["x", "y"])
    mask = xr.DataArray(
        [
            [True, False, True, False],
            [False, True, False, True],
            [True, False, True, False],
            [False, True, False, True],
        ],
        dims=["x", "y"],
    )
    result = ravel_from_mask(data, mask)
    assert result.shape == (8,)
    assert np.all(result.values == data.values[mask.values])


def test_unravel_from_mask() -> None:
    """Test unravel_from_mask function."""
    template = xr.DataArray(rng.random((4, 4)), dims=["x", "y"])
    mask = xr.DataArray(
        [
            [True, False, True, False],
            [False, True, False, True],
            [True, False, True, False],
            [False, True, False, True],
        ],
        dims=["x", "y"],
    )
    values = np.arange(8)
    result = unravel_from_mask(template, mask, values=values)
    assert result.shape == (4, 4)
    assert np.all(result.values[mask.values] == values)


def test_select_disk_mask() -> None:
    """Test select_disk_mask function."""
    data = xr.DataArray(rng.random((4, 4)), dims=["x", "y"])
    radius = 1.0
    mask = select_disk_mask(data, radius, around={"x": 2, "y": 2})
    assert mask.shape == (4, 4)
    assert np.sum(mask) > 0


def test_select_disk() -> None:
    """Test select_disk function."""
    data = xr.DataArray(rng.random((4, 4)), dims=["x", "y"])
    radius = 1.0
    coords, values, dist = select_disk(data, radius, around={"x": 2, "y": 2})
    assert len(values) > 0
    assert len(dist) == len(values)
    for d in dist:
        assert d <= radius
