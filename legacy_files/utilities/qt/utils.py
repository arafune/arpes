"""Contains utility classes for Qt in PyARPES."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6 import QtWidgets

__all__ = ["PlotOrientation", "ReactivePlotRecord"]


class PlotOrientation(enum.StrEnum):
    """Controls the transposition on a reactive plot."""

    Horizontal = "horizontal"
    Vertical = "vertical"


@dataclass
class ReactivePlotRecord:
    """Metadata related to a reactive plot or marginal on a DataArary.

    This is used to know how to update and mount corresponding widgets on a main tool view.
    """

    dims: list[int] | tuple[int, ...]
    view: QtWidgets.QWidget
    orientation: PlotOrientation
