from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def parse_setscale(line: str) -> tuple[str, str, float, float, str]:
    """Parse setscale.

    Args:
        line(str): line should start with "X SetScale"

    Returns:
        tuple[str, str, float, float, str]
    """
    assert "SetScale" in line
    flag: str
    dim: str
    num1: float
    num2: float
    unit: str
    setscale = line.split(",", maxsplit=5)
    if "/I" in setscale[0]:
        flag = "I"
    elif "/P" in line:
        flag = "P"
    else:
        flag = ""
    dim = setscale[0][-1]
    if dim not in {"x", "y", "z", "d", "t"}:
        msg = "Dimension is not correct"
        raise RuntimeError(msg)
    unit = setscale[3].strip()[1:-1]
    num1 = float(setscale[1])
    num2 = float(setscale[2])
    return (flag, dim, num1, num2, unit)


def angle_unit_to_rad(params: dict[str, str | float]) -> dict[str, str | float]:
    """Correct unit angle from degrees to radians in params object.

    Just a helper function.
    """
    for angle in ("beta", "chi", "theta", "psi", "phi"):
        if angle in params:
            params[angle] = np.deg2rad(params[angle])
        if angle + "_offset" in params:
            params[angle + "_offset"] = np.deg2rad(params[angle + "_offset"])
    return params


def as_angle(
    angle: NDArray[np.float64],
    *,
    keep_degree: bool = False,
) -> NDArray[np.float64]:
    if keep_degree:
        return angle
    return np.deg2rad(angle)


def correct_angle_region(
    angle_min: float,
    angle_max: float,
    num_pixel: int,
) -> tuple[float, float]:
    """Correct the angle value to fit igor.

    Parameters
    ----------
    angle_min: float
        Minimum angle of emission
    angle_max: float
        Maximum angle of emission
    num_pixel: int
        The number of pixels for non-energy channels (i.e. angle)

    Returns:
    -------
    tuple[float, float]
        minimum angle value and maximum angle value
    """
    diff: float = ((angle_max - angle_min) / num_pixel) / 2
    return angle_min + diff, angle_max - diff
