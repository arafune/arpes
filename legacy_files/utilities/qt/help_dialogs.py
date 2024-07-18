"""A help dialog showing keyboard shortcuts for Qt application."""

# pylint: disable=import-error
from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6 import QtCore, QtWidgets

from arpes.utilities.ui import PRETTY_KEYS, label, vertical

if TYPE_CHECKING:
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QGridLayout, QGroupBox, QLabel, QVBoxLayout

    from arpes.utilities.ui import KeyBinding
__all__ = ("BasicHelpDialog",)


class BasicHelpDialog(QtWidgets.QDialog):
    """A help dialog showing keyboard shortcuts for Qt application."""

    def __init__(self, shortcuts: list[KeyBinding] | None = None) -> None:
        """Initialize the help window and build widgets for the registered shortcuts."""
        super().__init__()

        if shortcuts is None:
            shortcuts = []

        self.layout: QVBoxLayout = QtWidgets.QVBoxLayout()  # type: ignore[assignment]
        keyboard_shortcuts_info: QGroupBox = QtWidgets.QGroupBox(title="Keyboard Shortcuts")
        keyboard_shortcuts_layout: QGridLayout = QtWidgets.QGridLayout()
        for i, shortcut in enumerate(shortcuts):
            the_label: QLabel = label(", ".join(PRETTY_KEYS[k] for k in shortcut.chord))
            the_label.setWordWrap(on=True)
            keyboard_shortcuts_layout.addWidget(
                the_label,
                i,
                0,
            )
            keyboard_shortcuts_layout.addWidget(label(shortcut.label), i, 1)

        keyboard_shortcuts_info.setLayout(keyboard_shortcuts_layout)

        aboutInfo: QtWidgets.QGroupBox = QtWidgets.QGroupBox(title="About")
        the_label1 = label(
            "QtTool is the work of Conrad Stansbury, with much inspiration "
            "and thanks to the authors of ImageTool. QtTool is distributed "
            "as part of the PyARPES data analysis framework.",
        )
        the_label1.setWordWrap(on=True)
        the_label2 = label(
            "Complaints and feature requests should be directed to chstan@berkeley.edu.",
        )
        the_label2.setWordWrap(on=True)
        vertical(the_label1, the_label2)

        from . import qt_info

        height = qt_info.inches_to_px(1)
        assert isinstance(height, int)
        aboutInfo.setFixedHeight(height)

        self.layout.addWidget(keyboard_shortcuts_info)
        self.layout.addWidget(aboutInfo)
        self.setLayout(self.layout)

        self.setWindowTitle("Interactive Utility Help")
        self.setFixedSize(*qt_info.inches_to_px((2, 4)))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """If the user preset H we should toggle the dialog, or close it if they pressed Esc."""
        if event.key() == QtCore.Qt.Key.Key_H or event.key() == QtCore.Qt.Key.Key_Escape:
            self.close()
