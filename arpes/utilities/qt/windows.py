"""Infrastructure code for Qt application windows."""

from __future__ import annotations

import sys
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING

from PySide6 import QtCore, QtGui, QtWidgets

import arpes.config
from arpes.utilities.excepthook import patched_excepthook
from arpes.utilities.ui import KeyBinding

if TYPE_CHECKING:
    from _typeshed import Incomplete
    from PySide6.QtCore import QObject
    from PySide6.QtGui import QCloseEvent, QKeyEvent

    from . import BasicHelpDialog

__all__ = ("SimpleWindow",)


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


class SimpleWindow(QtWidgets.QMainWindow, QtCore.QObject):
    """Provides a relatively simple way of making a windowed application.

    The following utilities are largely managed for you:

    1. Keybindings and chords
    2. Help and info
    3. Application lifecycle
    4. Inter-component messaging
    """

    HELP_DIALOG_CLS: type[BasicHelpDialog] | None = None

    def __init__(self) -> None:
        """Configure the window.

        In order to start the window, we

        * patch exception handling to provide a nicer traceback format
        * register key bindings and cursor modes
        * install filters to drop unnecessary Qt events
        """
        super().__init__()
        self.app: Incomplete = None
        self._help_dialog: BasicHelpDialog | None = None

        self._old_excepthook = sys.excepthook
        sys.excepthook = patched_excepthook

        self._cursorModes = self.compile_cursor_modes()
        self._keyBindings = self.compile_key_bindings()

        QtGui.QGuiApplication.installEventFilter(self, self)

    def compile_key_bindings(self) -> list[KeyBinding]:
        """Application generic key bindings.

        Additional keybindings can be added here as required by the tool.
        """
        return [
            KeyBinding("Close Window", [QtCore.Qt.Key.Key_Escape], self.do_close),
            KeyBinding("Toggle Help", [QtCore.Qt.Key.Key_H], self.toggle_help),
        ]

    def compile_cursor_modes(self) -> list[None]:
        """Unused hook for supporting additional cursor modes."""
        return []

    def closeEvent(self, event: QCloseEvent) -> None:
        """Close event."""
        self.do_close(event)

    def do_close(self, event: QCloseEvent) -> None:
        """Handle closing accepting an unused event arg."""
        logger.debug(f"unused {event!s} is detected")

        self.close()

    def close(self) -> bool:
        """If we need to close, give the application a chance to clean up first."""
        sys.excepthook = self._old_excepthook
        self.app().close()
        return super().close()

    def eventFilter(self, source: QObject, event: QKeyEvent) -> bool:  # type:ignore[override]
        """Neglect Qt events which do not relate to key presses for now."""
        special_keys = [
            QtCore.Qt.Key.Key_Down,
            QtCore.Qt.Key.Key_Up,
            QtCore.Qt.Key.Key_Left,
            QtCore.Qt.Key.Key_Right,
        ]

        if (
            event.type()
            in [
                QtCore.QEvent.Type.KeyPress,
                QtCore.QEvent.Type.ShortcutOverride,
            ]
        ) and (event.type() != QtCore.QEvent.Type.ShortcutOverride or event.key() in special_keys):
            self.handleKeyPressEvent(event)

        return super().eventFilter(source, event)

    def handleKeyPressEvent(self, event: QKeyEvent) -> None:
        """Listen key events supporting single key chords."""
        handled = False
        for binding in self._keyBindings:
            for combination in binding.chord:
                # only detect single keypresses for now
                if combination == event.key():
                    handled = True
                    binding.handler(event)

        if not handled and arpes.config.SETTINGS.get("DEBUG", False):
            logger.debug(f"{event.key()} @ {type(self)}:{event}")

    def toggle_help(self, event: QKeyEvent) -> None:
        """Open and close (toggle) the help panel for the application."""
        msg = f"unused {event!s} is detected"
        logger.debug(msg)
        if self.HELP_DIALOG_CLS is None:
            return

        if self._help_dialog is None:
            self._help_dialog = self.HELP_DIALOG_CLS(shortcuts=self._keyBindings)
            self._help_dialog.show()
            self._help_dialog._main_window = self
        else:
            self._help_dialog.close()
            self._help_dialog = None

    def window_print(self, *args: Incomplete, **kwargs: Incomplete) -> None:
        """Forward prints to the application instance so they end up in Jupyter."""
        print(*args, **kwargs)
