"""Unit test for qt related."""

from typing import TYPE_CHECKING

import pytest
from arpes.io import example_data
from arpes.plotting.qt.qt_tool import qt_tool
from PySide6 import QtCore
from pytestqt.qt_compat import qt_api
from pytestqt.qtbot import QtBot

if TYPE_CHECKING:
    from arpes.plotting.qt.qt_tool import QtTool


@pytest.mark.skip()
def test_open_qt_tool_and_basic_functionality(qtbot: QtBot) -> None:
    """Test for qt_tool and it's basic functionality.

    [TODO:description]

    Args:
        qtbot: [TODO:description]

    Returns:
        [TODO:description]
    """
    app = qt_api.QtWidgets.QApplication.instance()

    qt_tool(example_data.cut.spectrum, no_exec=True)
    example_data.cut.S.show(app=app, no_exec=True)
    owner: QtTool = app.owner

    # Check transposition info
    assert list(owner.data.dims) == ["phi", "eV"]

    qtbot.keyPress(app.owner.cw, "t")
    assert list(owner.data.dims) == ["eV", "phi"]
    qtbot.keyPress(app.owner.cw, "y")
    assert list(owner.data.dims) == ["phi", "eV"]

    # Check cursor info
    assert app.owner.context["cursor"] == [120, 120]
    qtbot.keyPress(app.owner.cw, QtCore.Qt.Key.Key_Left)
    assert app.owner.context["cursor"] == [118, 120]
    qtbot.keyPress(app.owner.cw, QtCore.Qt.Key.Key_Up)
    assert app.owner.context["cursor"] == [118, 122]
    qtbot.keyPress(app.owner.cw, QtCore.Qt.Key.Key_Right)
    qtbot.keyPress(app.owner.cw, QtCore.Qt.Key.Key_Right)
    assert app.owner.context["cursor"] == [122, 122]
    qtbot.keyPress(app.owner.cw, QtCore.Qt.Key.Key_Down)
    assert app.owner.context["cursor"] == [122, 120]

    app.owner.close()
