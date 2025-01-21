"""Tests for xarray.py utilities."""

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import xarray as xr

from arpes.utilities.xarray import (
    apply_dataarray,
    lift_dataarray,
    lift_dataarray_attrs,
    lift_datavar_attrs,
    unwrap_xarray_dict,
    unwrap_xarray_item,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def test_unwrap_xarray_item() -> None:
    arr = xr.DataArray(np.array([1.0]))
    assert unwrap_xarray_item(arr) == 1.0

    arr = xr.DataArray(np.array([1.0, 2.0]))
    assert unwrap_xarray_item(arr).equals(arr)


@pytest.mark.skip
def test_unwrap_xarray_dict() -> None:
    input_dict = {"a": xr.DataArray(np.array([1.0])), "b": xr.DataArray(np.array([2.0, 3.0]))}
    expected_dict = {"a": 1.0, "b": xr.DataArray(np.array([2.0, 3.0]))}
    assert unwrap_xarray_dict(input_dict) == expected_dict


def test_apply_dataarray() -> None:
    arr = xr.DataArray(np.array([1.0, 2.0, 3.0]))
    f: Callable[[np.ndarray, Any], np.ndarray] = lambda x, y: x + y
    result = apply_dataarray(arr, f, 1)
    expected = xr.DataArray(np.array([2.0, 3.0, 4.0]))
    assert result.equals(expected)


def test_lift_dataarray() -> None:
    f: Callable[[np.ndarray, Any], np.ndarray] = lambda x, y: x + y
    lifted_f = lift_dataarray(f)
    arr = xr.DataArray(np.array([1.0, 2.0, 3.0]))
    result = lifted_f(arr, 1)
    expected = xr.DataArray(np.array([2.0, 3.0, 4.0]))
    assert result.equals(expected)


def test_lift_dataarray_attrs() -> None:
    f: Callable[[dict[str, int], int], dict[str, int]] = lambda attrs, x: {
        k: v + x for k, v in attrs.items()
    }
    lifted_f = lift_dataarray_attrs(f)
    arr = xr.DataArray(np.array([1.0, 2.0, 3.0]), attrs={"a": 1, "b": 2})
    result = lifted_f(arr, 1)
    expected_attrs = {"a": 2, "b": 3}
    assert result.attrs == expected_attrs


def test_lift_datavar_attrs() -> None:
    f: Callable[[dict[str, int], int], dict[str, int]] = lambda attrs, x: {
        k: v + x for k, v in attrs.items()
    }
    lifted_f = lift_datavar_attrs(f)
    data = xr.Dataset(
        {
            "var1": xr.DataArray(np.array([1.0, 2.0]), attrs={"a": 1}),
            "var2": xr.DataArray(np.array([3.0, 4.0]), attrs={"b": 2}),
        },
        attrs={"c": 3},
    )
    result = lifted_f(data, 1)
    expected_vars = {
        "var1": xr.DataArray(np.array([1.0, 2.0]), attrs={"a": 2}),
        "var2": xr.DataArray(np.array([3.0, 4.0]), attrs={"b": 3}),
    }
    expected_attrs = {"c": 4}
    assert result.data_vars["var1"].attrs == expected_vars["var1"].attrs
    assert result.data_vars["var2"].attrs == expected_vars["var2"].attrs
    assert result.attrs == expected_attrs
