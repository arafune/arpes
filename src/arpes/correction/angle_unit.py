"""This module provides functions to manipulate angle unit in coordinates/attributes."""

import numpy as np

from arpes._typing import ANGLE, DataType, flatten_literals


def radian_to_degree(data: DataType) -> DataType:
    """Switch angle unit from Radians to Degrees in place."""
    data.attrs["angle_unit"] = "Degrees"
    for angle in flatten_literals(ANGLE):
        if angle in data.attrs:
            data.attrs[angle] = np.rad2deg(data.attrs.get(angle, np.nan))
        if angle + "_offset" in data.attrs:
            data.attrs[angle + "_offset"] = np.rad2deg(
                data.attrs.get(angle + "_offset", np.nan),
            )
        if angle in data.coords:
            data.coords[angle] = np.rad2deg(data.coords[angle])
    return data


def degree_to_radian(data: DataType) -> DataType:
    """Switch angle unit from Degrees and Radians in place."""
    data.attrs["angle_unit"] = "Radians"
    for angle in flatten_literals(ANGLE):
        if angle in data.attrs:
            data.attrs[angle] = np.deg2rad(data.attrs.get(angle, np.nan))
        if angle + "_offset" in data.attrs:
            data.attrs[angle + "_offset"] = np.deg2rad(
                data.attrs.get(angle + "_offset", np.nan),
            )
        if angle in data.coords:
            data.coords[angle] = np.deg2rad(data.coords[angle])

    return data


def switched_angle_unit(data: DataType) -> DataType:
    """Return DataArray/Dataset in which the angle unit is switched (radians <-> degrees).

    Change the value of angle related objects/variables in attrs and coords

    Args:
        data (DataType): Data in which the angle unit converted.

    Returns:
        DataType: The angle unit converted data.
    """
    data_copy = data.copy(deep=True)
    angle_unit = data_copy.attrs.get("angle_unit", "Radians").lower()
    if angle_unit.startswith("rad"):
        return radian_to_degree(data_copy)
    if angle_unit.startswith("deg"):
        return degree_to_radian(data_copy)
    msg = 'The angle_unit must be "Radians" or "Degrees"'
    raise TypeError(msg)


def switch_angle_unit(data: DataType) -> None:
    """Switch angle unit in place.

    Change the value of angle related objects/variables in attrs and coords

    Args:
        data: (DataType): Data in which the angle unit converted.

    Note:
        When data is Dataset object, the attrs and coords for the Dataset component alone are
        swicched. (DataArray.attrs and DataArray.coords in data_vars are not changed.)
    """
    converted_data = switched_angle_unit(data)
    data.attrs.clear()
    data.attrs.update(converted_data.attrs)
    for coord in converted_data.coords:
        data.coords[coord] = converted_data.coords[coord]
