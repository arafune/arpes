"""This module provides functions to manipulate angle unit in coordinates/attributes."""

from collections.abc import Callable
from typing import TypeVar, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from arpes._typing.base import ANGLE, DataType
from arpes._typing.utils import flatten_literals
from arpes.xarray_extensions.accessor.spectrum_type import AngleUnit

AngleValue = TypeVar("AngleValue", float, np.float64, NDArray[np.float64])


def radian_to_degree(data: xr.DataArray) -> xr.DataArray:
    """Return DataArray/Dataset switched angle unit from Radians to Degrees.

    If already angle unit is Degrees, do nothing.

    Args:
        data: Data in which the angle unit converted to Degrees.

    Returns:
        DataType: The angle unit converted data.
    """
    if data.S.angle_unit is AngleUnit.DEG:
        return data
    data.S.angle_unit = AngleUnit.DEG

    for angle in flatten_literals(ANGLE):
        if angle in data.attrs:
            data.attrs[angle] = np.rad2deg(data.attrs.get(angle, np.nan))
        if angle + "_offset" in data.attrs:
            data.attrs[angle + "_offset"] = np.rad2deg(
                data.attrs.get(angle + "_offset", np.nan),
            )

    new_coords = {
        angle: np.rad2deg(data.coords[angle])
        for angle in flatten_literals(ANGLE)
        if angle in data.coords
    }
    return data.assign_coords(new_coords)


def degree_to_radian(data: xr.DataArray) -> xr.DataArray:
    """Return DataArray/Dataset switched angle unit from Degrees to Radians.

    If already angle unit is Radians, do nothing.

    Args:
        data:  Data in which the angle unit converted to Radians.

    Returns:
        DataType: The angle unit converted data.
    """
    if data.S.angle_unit is AngleUnit.RAD:
        return data
    data.S.angle_unit = AngleUnit.RAD
    for angle in flatten_literals(ANGLE):
        if angle in data.attrs:
            data.attrs[angle] = np.deg2rad(data.attrs.get(angle, np.nan))
        if angle + "_offset" in data.attrs:
            data.attrs[angle + "_offset"] = np.deg2rad(
                data.attrs.get(angle + "_offset", np.nan),
            )

    new_coords = {
        angle: np.deg2rad(data.coords[angle])
        for angle in flatten_literals(ANGLE)
        if angle in data.coords
    }

    return data.assign_coords(new_coords)


def switched_angle_unit_imp(data: xr.DataArray) -> xr.DataArray:
    """Return DataArray/Dataset in which the angle unit is switched (radians <-> degrees).

    Change the value of angle related objects/variables in attrs and coords

    Args:
        data (DataType): Data in which the angle unit converted.

    Returns:
        DataType: The angle unit converted data.
    """
    data_copy = data.copy(deep=True)
    if data_copy.S.angle_unit is AngleUnit.RAD:
        return radian_to_degree(data_copy)
    # AngleUnit.DEG:
    return degree_to_radian(data_copy)


def _convert_angle_attrs(
    attrs: dict[str, AngleValue],
    convert: Callable[[AngleValue], AngleValue],
) -> None:
    for angle in flatten_literals(ANGLE):
        if angle in attrs:
            attrs[angle] = convert(attrs[angle])
        offset = angle + "_offset"
        if offset in attrs:
            attrs[offset] = convert(attrs[offset])


def switched_angle_unit(data: DataType) -> DataType:
    """Return DataArray/Dataset in which the angle unit is switched (radians <-> degrees).

    Change the value of angle related objects/variables in attrs and coords

    Args:
        data (DataType): Data in which the angle unit converted.

    Returns:
        DataType: The angle unit converted data.
    """
    if isinstance(data, xr.DataArray):
        return cast("DataType", switched_angle_unit_imp(data))

    assert isinstance(data, xr.Dataset)

    data = data.copy(deep=True)
    if data.S.angle_unit is AngleUnit.RAD:
        data.S.angle_unit = AngleUnit.DEG
        convert = np.rad2deg
    else:
        data.S.angle_unit = AngleUnit.RAD
        convert = np.deg2rad

    _convert_angle_attrs(data.attrs, convert)

    new_coords = {
        angle: convert(data.coords[angle])
        for angle in flatten_literals(ANGLE)
        if angle in data.coords
    }

    for spectrum in data.data_vars.values():
        _convert_angle_attrs(spectrum.attrs, convert)

    return data.assign_coords(new_coords)


def switch_angle_unit(data: DataType) -> None:
    """Switch angle unit in place.

    Change the value of angle related objects/variables in attrs and coords

    Args:
        data: (DataType): Data in which the angle unit converted.

    Note:
        When data is Dataset object, the attrs and coords for the Dataset component alone are
        swicched. (DataArray.attrs and DataArray.coords in data_vars are not changed.)
    """
    if isinstance(data, xr.DataArray):
        new = switched_angle_unit_imp(data)

        data.coords.update(new.coords)
        data.attrs.clear()
        data.attrs.update(new.attrs)
        return

    assert isinstance(data, xr.Dataset)
    new = switched_angle_unit(data)
    data.coords.update(new.coords)
    data.attrs.clear()
    data.attrs.update(new.attrs)
