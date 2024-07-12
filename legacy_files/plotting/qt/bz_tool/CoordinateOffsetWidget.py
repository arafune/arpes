"""A widget providing controls for a coordinate offset in the momentum tool."""

from __future__ import annotations

from functools import partial
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING

from PySide6 import QtWidgets

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


if TYPE_CHECKING:
    from _typeshed import Incomplete
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QGridLayout, QWidget

    from . import BZTool
__all__ = ("CoordinateOffsetWidget",)


class CoordinateOffsetWidget(QtWidgets.QGroupBox):
    """Control for a single coordinate offset in the momentum tool."""

    def __init__(
        self,
        parent: QWidget | None = None,
        root: type[BZTool] | None = None,
        coordinate_name: str = "",
        value: Incomplete = None,
    ) -> None:
        """Configures utility label, an inner control, and a linked spinbox for text entry."""
        super().__init__(title=coordinate_name, parent=parent)
        logger.debug(f"value = {value} has not been used.")
        self.layout: QGridLayout = QtWidgets.QGridLayout(self)

        self.label = QtWidgets.QLabel("Value: ")
        self.spinbox = QtWidgets.QSpinBox()
        self.slider = QtWidgets.QSlider()
        self.root = root

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.spinbox)
        self.layout.addWidget(self.slider)

        self._prevent_change_events = False

        self.slider.valueChanged.connect(partial(self.value_changed, source=self.slider))
        self.spinbox.valueChanged.connect(partial(self.value_changed, source=self.spinbox))

        self.recompute()

    def value_changed(
        self,
        event: QEvent,
        source: Incomplete,
    ) -> None:
        """Propagates values change to update the displayed data."""
        if self._prevent_change_events:
            return

        logger.debug(f"event={event} has not been used.")
        self._prevent_change_events = True
        self.slider.setValue(source.value())
        self.spinbox.setValue(source.value())
        self._prevent_change_events = False
        self.recompute()
        if self.root is not None:
            self.root().update_cut()

    def recompute(self) -> None:
        """Recompute stale UI state from this control's value."""
        value = self.spinbox.value()
        self.label.setText(f"Value: {value:.3g}")

    def value(self) -> int:
        """The inner value.

        Returns:
            [TODO:description]
        """
        return self.spinbox.value()
