"""Unit test for analysis.mask."""

import numpy as np
import pytest
import xarray as xr

from arpes.analysis import mask


def test_raw_poly_to_mask():
    poly = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    out = mask.raw_poly_to_mask(poly)
    assert "poly" in out
    np.testing.assert_array_equal(out["poly"], poly)


def test_polys_to_mask_simple():
    coords = xr.Dataset(coords={"x": np.arange(5), "y": np.arange(5)}).coords
    shape = (5, 5)
    poly = [[1, 1], [3, 1], [3, 3], [1, 3]]
    mask_dict = {"dims": ["x", "y"], "polys": [poly]}
    out = mask.polys_to_mask(mask_dict, coords, shape)
    assert out.shape == shape
    assert out[2, 2]
    assert not out[0, 0]


def test_polys_to_mask_invert():
    coords = xr.Dataset(coords={"x": np.arange(5), "y": np.arange(5)}).coords
    shape = (5, 5)
    poly = [[1, 1], [3, 1], [3, 3], [1, 3]]
    mask_dict = {"dims": ["x", "y"], "polys": [poly]}
    out = mask.polys_to_mask(mask_dict, coords, shape, invert=True)
    assert not out[2, 2]
    assert out[0, 0]


def test_polys_to_mask_multiple_polys():
    coords = xr.Dataset(coords={"x": np.arange(5), "y": np.arange(5)}).coords
    shape = (5, 5)
    poly1 = [[1, 1], [3, 1], [3, 3], [1, 3]]
    poly2 = [[0, 0], [2, 0], [2, 2], [0, 2]]
    mask_dict = {"dims": ["x", "y"], "polys": [poly1, poly2]}
    out = mask.polys_to_mask(mask_dict, coords, shape)
    assert out[2, 2]
    assert out[1, 1]
    assert not out[4, 4]


def test_apply_mask_to_coords():
    x = np.arange(3)
    y = np.arange(3)
    X, Y = np.meshgrid(x, y)
    ds = xr.Dataset(
        {
            "X": (("x", "y"), X),
            "Y": (("x", "y"), Y),
        },
        coords={"x": x, "y": y},
    )

    poly = [[0, 0], [2, 0], [2, 2], [0, 2]]
    mask_dict = {"poly": poly}
    arr = mask.apply_mask_to_coords(ds, mask_dict, ["X", "Y"], invert=False)
    expected = np.zeros((3, 3), dtype=bool)
    expected[1:, :] = True
    np.testing.assert_array_equal(arr, expected)

    arr_inv = mask.apply_mask_to_coords(ds, mask_dict, ["X", "Y"], invert=True)
    np.testing.assert_array_equal(arr_inv, ~expected)


def test_apply_mask_with_dict():
    arr = xr.DataArray(
        np.arange(16).reshape(4, 4),
        dims=("x", "y"),
        coords={"x": np.arange(4), "y": np.arange(4)},
    )
    poly = [[0.5, 0.5], [2.5, 0.5], [2.5, 2.5], [0.5, 2.5]]
    mask_dict = {"dims": ["x", "y"], "polys": [poly]}
    masked = mask.apply_mask(arr, mask_dict, replace=-1)

    expected = arr.values.astype(float)

    expected[1:, 2:] = -1
    np.testing.assert_array_equal(masked.values, expected)


def test_apply_mask_with_array():
    arr = xr.DataArray(np.arange(9).reshape(3, 3), dims=("x", "y"))
    mask_arr = np.zeros((3, 3), dtype=bool)
    mask_arr[0, 0] = True
    result = mask.apply_mask(arr, mask_arr, replace=999)
    assert result[0, 0].values == 999
    assert (result.values[1:, 1:] != 999).all()


def test_apply_mask_with_fermi0():
    arr = xr.DataArray(np.arange(10), dims=("eV",), coords={"eV": np.linspace(0, 0.5, 10)})
    # x: eV, y: dummy axis
    poly = [[0.1, 0], [0.4, 0]]
    mask_dict = {"dims": ["eV", "dummy"], "polys": [poly], "fermi": 0.3}
    # arrにdummy軸を追加
    arr2 = arr.expand_dims(dummy=[0])
    out = mask.apply_mask(arr2, mask_dict, replace=np.nan)
    assert out.eV.max() <= 0.5
    assert (out.eV <= 0.5).all()


def test_apply_mask_type_error():
    arr = xr.DataArray(np.arange(9).reshape(3, 3), dims=("x", "y"))
    with pytest.raises(Exception):
        mask.apply_mask(arr, "notadict")
