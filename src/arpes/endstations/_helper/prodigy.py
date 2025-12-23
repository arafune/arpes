from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class IgorSetscaleFlag(Enum):
    """Enum for Igor SetScale flag."""

    INCLUSIVE = "I"
    PERPOINTS = "P"
    DEFAULT = ""

    def set_scale(
        self,
        num1: float,
        num2: float,
        pixels: int,
    ) -> NDArray[np.float64]:
        """Return scale array based on the flag."""
        scale_map = {
            IgorSetscaleFlag.INCLUSIVE: lambda: np.linspace(
                num1,
                num2,
                num=pixels,
                dtype=np.float64,
            ),
            IgorSetscaleFlag.PERPOINTS: lambda: np.linspace(
                num1,
                num1 + num2 * (pixels - 1),
                num=pixels,
                dtype=np.float64,
            ),
            IgorSetscaleFlag.DEFAULT: lambda: np.linspace(
                num1,
                num2,
                num=pixels,
                dtype=np.float64,
                endpoint=False,
            ),
        }
        return scale_map[self]()


def parse_setscale(
    line: str,
) -> tuple[
    IgorSetscaleFlag,
    str,
    float,
    float,
    str,
]:
    """Parse setscale.

    Args:
        line(str): line should start with "X SetScale"

    Returns:
        tuple[IgorSetscaleFlag, str, float, float, str]
    """
    assert "SetScale" in line
    flag: IgorSetscaleFlag
    dim: str
    num1: float
    num2: float
    unit: str
    setscale = line.split(",", maxsplit=5)
    if "/I" in setscale[0]:
        flag = IgorSetscaleFlag.INCLUSIVE
    elif "/P" in line:
        flag = IgorSetscaleFlag.PERPOINTS
    else:
        flag = IgorSetscaleFlag.DEFAULT
    dim = setscale[0][-1]
    if dim not in {"x", "y", "z", "d", "t"}:
        msg = "Dimension is not correct"
        raise RuntimeError(msg)
    unit = setscale[3].strip()[1:-1]
    num1 = float(setscale[1])
    num2 = float(setscale[2])
    return (flag, dim, num1, num2, unit)


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
