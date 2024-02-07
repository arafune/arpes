"""A widget providing rudimentary information about an axis on a DataArray."""

# pylint: disable=import-error
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from PySide6 import QtWidgets

if TYPE_CHECKING:
    from PySide6.QtWidgets import QGridLayout, QLabel, QPushButton

    from . import QtTool

__all__ = ("AxisInfoWidget",)


class AxisInfoWidget(QtWidgets.QGroupBox):
    """A widget providing some rudimentary axis information."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
        root: type[QtTool] | None = None,
        axis_index: int = 0,
    ) -> None:
        """Configure inner widgets for axis info, and transpose to front button."""
        super().__init__(title=str(axis_index), parent=parent)

        self.layout: QGridLayout = QtWidgets.QGridLayout(self)

        self.label: QLabel = QtWidgets.QLabel("Cursor: ")
        self.transpose_button: QPushButton = QtWidgets.QPushButton("To Front")
        self.transpose_button.clicked.connect(self.on_transpose)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.transpose_button)

        self.axis_index = axis_index
        self._root = root
        self.setLayout(self.layout)
        self.recompute()

    @property
    def root(self) -> QtTool:
        """Unwraps the weakref to the parent application."""
        assert self._root is not None
        return self._root()

    def recompute(self) -> None:
        """Force a recomputation of dependent UI state: here, the title and text."""
        self.setTitle(str(self.root.data.dims[self.axis_index]))
        try:
            cursor_index = self.root.context["cursor"][self.axis_index]
            cursor_value = self.root.context["value_cursor"][self.axis_index]
            self.label.setText(f"Cursor: {int(cursor_index)}, {cursor_value:.3g}")
        except KeyError:
            pass

    def on_transpose(self) -> None:
        """This UI control lets you transpose the axis it refers to to the front."""
        with contextlib.suppress(Exception):
            self.root.transpose_to_front(self.axis_index)
