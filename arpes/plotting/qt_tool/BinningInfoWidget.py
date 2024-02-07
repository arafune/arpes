"""An axis binning control."""

from __future__ import annotations

from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING

import xarray as xr
from PySide6 import QtWidgets

from arpes.utilities.ui import layout

if TYPE_CHECKING:
    from . import QtTool

__all__ = ("BinningInfoWidget",)

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


class BinningInfoWidget(QtWidgets.QGroupBox):
    """A spinbox allowing you to set the binning on different axes."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        root: type[QtTool] | None = None,
        axis_index: int | None = None,
    ) -> None:
        """Initialize an inner spinbox and connect signals to get reactivity."""
        super().__init__(title=str(axis_index), parent=parent)
        self._root = root
        self.axis_index = axis_index

        self.spinbox = QtWidgets.QSpinBox()
        self.spinbox.setMaximum(2000)
        self.spinbox.setMinimum(1)
        self.spinbox.setValue(1)
        self.spinbox.valueChanged.connect(self.changeBinning)
        self.spinbox.editingFinished.connect(self.changeBinning)

        self.layout: QtWidgets.QWidget = layout(
            self.spinbox,
            widget=self,
        )

        self.recompute()

    @property
    def root(self) -> QtTool:
        """Unwraps the weakref to the parent application."""
        assert self._root is not None
        return self._root()

    def recompute(self) -> None:
        """Redraws all dependent UI state, namely the title."""
        assert isinstance(self.axis_index, int)
        assert isinstance(self.root.data, xr.DataArray)  # data.dim should be tuple.
        self.setTitle(str(self.root.data.dims[self.axis_index]))

    def changeBinning(self) -> None:
        """Callback for widget value changes which sets the binning on the root app."""
        old_binning = self.root.binning
        assert isinstance(self.axis_index, int)
        old_binning[self.axis_index] = self.spinbox.value()
        self.root.binning = old_binning
