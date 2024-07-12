"""A control which can provide a range or a single value.

(i.e. a half open range with ends equal).
"""

from __future__ import annotations

from functools import partial
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING

from PySide6 import QtWidgets

if TYPE_CHECKING:
    from _typeshed import Incomplete
    from PySide6.QtWidgets import QGridLayout, QWidget

    from . import BZTool

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


__all__ = ["RangeOrSingleValueWidget"]


class RangeOrSingleValueWidget(QtWidgets.QGroupBox):
    """A UI control letting you set a single value or a range.

    Used for modeling single cuts or multi-cut scans in the BZ tool.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        root: type[BZTool] | None = None,
        coordinate_name: str = "",
        value: Incomplete = None,
    ) -> None:
        """Configures and initializes inner widgts.

        Inernally, we use a checkbox, spinbox, and slider to model the UI controls here,
        the checkbox determines which UI control is the "active" one.
        """
        super().__init__(title=coordinate_name, parent=parent)

        logger.debug(f"value = {value} has not been used.")
        self.layout: QGridLayout = QtWidgets.QGridLayout(self)

        self.label = QtWidgets.QLabel("Value: ")
        self.spinbox = QtWidgets.QSpinBox()
        self.slider = QtWidgets.QSlider()
        self.checkbox = QtWidgets.QCheckBox("Ranged")

        self.root = root

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.spinbox)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(self.toggle)

        self._prevent_change_events = False

        self.slider.valueChanged.connect(partial(self.value_changed, source=self.slider))
        self.spinbox.valueChanged.connect(partial(self.value_changed, source=self.spinbox))
        self.checkbox.valueChanged.connect(partial(self.mode_changed, source=self.checkbox))

        self.recompute()

    def mode_changed(self, event: Incomplete, source: Incomplete) -> None:
        """Unused, currently."""

    def value_changed(self, event: Incomplete, source: Incomplete) -> None:
        """Responds to changes in the internal value."""
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
        """Recompute UI representation from inner values."""
        value = self.spinbox.value()
        self.label.setText(f"Value: {value:.3g}")

    def value(self) -> None:
        """The inner value."""
        return self.spinbox.value()
