"""Infrastructure code for Qt based analysis tools."""
from __future__ import annotations

import functools
from multiprocessing import Process
from typing import TYPE_CHECKING

import dill
import pyqtgraph as pg
from pyqtgraph import ViewBox

from arpes._typing import xr_types

from .app import SimpleApp
from .data_array_image_view import DataArrayImageView
from .help_dialogs import BasicHelpDialog
from .windows import SimpleWindow

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable
    from typing import Literal, Self

    from _typeshed import Incomplete
    from PySide6.QtWidgets import QApplication

    from arpes._typing import DataType

__all__ = (
    "DataArrayImageView",
    "BasicHelpDialog",
    "SimpleWindow",
    "SimpleApp",
    "qt_info",
    "remove_dangling_viewboxes",
    "run_tool_in_daemon_process",
)


def run_tool_in_daemon_process(tool_handler: Callable) -> Callable:
    """Starts a Qt based tool as a daemon process.

    This is exceptionally useful because it let's you have multiple tool windows
    open simultaneously and does not block the main "analysis" process.

    It also means that crashes due to Qt do not crash the analysis process, although
    it makes them slightly harder to debug.

    For this reason, if you are developing a Qt based analysis tool
    it might make sense for you to run it in the main thread.
    """

    @functools.wraps(tool_handler)
    def wrapped_handler(
        data: DataType,
        *,
        detached: bool = False,
        **kwargs: Incomplete,
    ) -> None:
        if not detached:
            return tool_handler(data, **kwargs)

        if isinstance(data, xr_types):
            # this should be a noop but seems to fix a bug which
            # causes dill to crash after loading an nc array
            data = data.assign_coords(data.coords)

        ser_data = dill.dumps(data)
        p = Process(target=tool_handler, args=(ser_data,), kwargs=kwargs, daemon=True)
        p.start()
        return None

    return wrapped_handler


def remove_dangling_viewboxes() -> None:
    """Removes ViewBoxes that don't get garbage collected on app close.

    If you construct a view hierarchy which has circular references
    then it can happen that Python will retain the references to Qt
    objects after they have been freed. This has manifested as
    ViewBoxes which remain and prevent restarting of an interactive tool.

    For now I have actually gone and fixed this problem by removing the circular
    dependencies, but in a pinch you can also call this function
    to remove the orphaned ViewBoxes.

    There are two places we need to clean these stale views up:

    * ViewBox.AllViews
    * ViewBox.NamedViews
    """
    import sip

    for_deletion = set()

    # In each case, we need to coerce the collection to
    # a list before we iterate because we are modifying the
    # underlying collection
    for v in list(ViewBox.AllViews):
        if sip.isdeleted(v):
            # first remove it from the ViewBox references
            # and then we will delete it later to prevent an
            # error
            for_deletion.add(v)
            del ViewBox.AllViews[v]

    for vname in list(ViewBox.NamedViews):
        v = ViewBox.NamedViews[vname]

        if sip.isdeleted(v):
            for_deletion.add(v)
            del ViewBox.NamedViews[vname]

    for v in for_deletion:
        del v


class QtInfo:
    screen_dpi = 150

    def __init__(self) -> None:
        self._inited = False
        self._pg_patched = False

    def init_from_app(self, app: QApplication) -> None:
        if self._inited:
            return

        self._inited = True
        dpis = [screen.physicalDotsPerInch() for screen in app.screens()]
        self.screen_dpi = sum(dpis) / len(dpis)

    def apply_settings_to_app(self, app: QApplication) -> None:
        # Adjust the font size based on screen DPI
        font = app.font()
        font.setPointSize(self.inches_to_px(0.1))
        app.instance().setFont(font)

    def inches_to_px(
        self,
        arg: float | tuple[float, ...],
    ) -> int | Generator[int, None, None]:
        if isinstance(
            arg,
            int | float,
        ):
            return int(self.screen_dpi * arg)

        return (int(x * self.screen_dpi) for x in arg)

    def setup_pyqtgraph(self) -> None:
        """Does any patching required on PyQtGraph and configures options."""
        if self._pg_patched:
            return

        self._pg_patched = True

        pg.setConfigOptions(
            antialias=True,
            foreground=(0, 0, 0),
            background=(255, 255, 255),
        )

        def patchedLinkedViewChanged(
            self: Self,
            view: pg.ViewBox,
            axis: Literal[0, 1, 2],
        ) -> None:
            """Patches linkedViewChanged to fix a pixel scaling bug.

            This still isn't quite right but it is much better than before. For some reason
            the screen coordinates of the PlotWidget are not being computed correctly, so
            we will just lock them as though they were perfectly aligned.

            This will clearly not work well for plots that have to be coordinated across
            different parts of the layout, but this will work for now.

            We also don't handle inverted axes for now.
            """
            if self.linksBlocked or view is None:
                return

            vr = view.viewRect()
            vg = view.screenGeometry()
            sg = self.screenGeometry()
            if vg is None or sg is None:
                return

            view.blockLink(b=True)

            try:
                if axis == pg.ViewBox.XAxis:
                    upp = float(vr.width()) / vg.width()
                    overlap = min(sg.right(), vg.right()) - max(sg.left(), vg.left())

                    if overlap < min(vg.width() / 3, sg.width() / 3):
                        x1 = vr.left()
                        x2 = vr.right()
                    else:  # attempt to align
                        x1 = vr.left()
                        x2 = vr.right() + (sg.width() - vg.width()) * upp

                    self.enableAutoRange(pg.ViewBox.XAxis, enable=False)
                    self.setXRange(x1, x2, padding=0)
                else:
                    upp = float(vr.height()) / vg.height()
                    overlap = min(sg.bottom(), vg.bottom()) - max(sg.top(), vg.top())

                    if overlap < min(vg.height() / 3, sg.height() / 3):
                        y1 = vr.top()
                        y2 = vr.bottom()
                    else:  # again, attempt to align
                        y1 = vr.top()  # snap them at one side to the same coordinate

                        # and scale the other side
                        y2 = vr.bottom() + (sg.height() - vg.height()) * upp

                    self.enableAutoRange(pg.ViewBox.YAxis, enable=False)
                    self.setYRange(y1, y2, padding=0)
            finally:
                view.blockLink(b=False)

        pg.ViewBox.linkedViewChanged = patchedLinkedViewChanged


qt_info = QtInfo()  # singleton configuration
