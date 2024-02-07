"""Provides a Qt based implementation of Igor's ImageTool."""

# pylint: disable=import-error
from __future__ import annotations

import contextlib
import warnings
import weakref
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, reveal_type

import dill
import matplotlib as mpl
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QGridLayout

from arpes.utilities import normalize_to_spectrum
from arpes.utilities.qt import (
    BasicHelpDialog,
    DataArrayImageView,
    SimpleApp,
    SimpleWindow,
    qt_info,
    run_tool_in_daemon_process,
)
from arpes.utilities.qt.data_array_image_view import DataArrayPlot
from arpes.utilities.qt.utils import PlotOrientation
from arpes.utilities.ui import CursorRegion, KeyBinding, horizontal, tabs

from .AxisInfoWidget import AxisInfoWidget
from .BinningInfoWidget import BinningInfoWidget

if TYPE_CHECKING:
    import xarray as xr
    from _typeshed import Incomplete
    from PySide6.QtCore import QEvent
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QWidget

    from arpes._typing import DataType

LOGLEVEL = (DEBUG, INFO)[1]
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

__all__ = (
    "QtTool",
    "qt_tool",
)

qt_info.setup_pyqtgraph()


class QtToolWindow(SimpleWindow):
    """The application window for `QtTool`.

    QtToolWindow was the first Qt-Based Tool that I built for PyARPES. Much of its structure was
    ported to SimpleWindow and borrowed ideas from when I wrote DAQuiri. As a result, the structure
    is essentially now to define just the handlers and any lifecycle hooks (close, etc.)
    """

    HELP_DIALOG_CLS = BasicHelpDialog

    def compile_key_bindings(self) -> list[KeyBinding]:
        """[TODO:summary].

        Returns:
            [TODO:description]
        """
        return [
            *super().compile_key_bindings(),
            KeyBinding(
                "Scroll Cursor",
                [
                    QtCore.Qt.Key.Key_Left,
                    QtCore.Qt.Key.Key_Right,
                    QtCore.Qt.Key.Key_Up,
                    QtCore.Qt.Key.Key_Down,
                ],
                self.scroll,
            ),
            KeyBinding(
                "Reset Intensity",
                [QtCore.Qt.Key.Key_I],
                self.reset_intensity,
            ),
            KeyBinding(
                "Scroll Z-Cursor",
                [
                    QtCore.Qt.Key.Key_N,
                    QtCore.Qt.Key.Key_M,
                ],
                self.scroll_z,
            ),
            KeyBinding(
                "Center Cursor",
                [QtCore.Qt.Key.Key_C],
                self.center_cursor,
            ),
            KeyBinding(
                "Transpose - Roll Axis",
                [QtCore.Qt.Key.Key_T],
                self.transpose_roll,
            ),
            KeyBinding(
                "Transpose - Swap Front Axes",
                [QtCore.Qt.Key.Key_Y],
                self.transpose_swap,
            ),
        ]

    def center_cursor(self, event: QEvent) -> None:
        logger.debug(f"method: center_cursor {event!s}")
        self.app().center_cursor()

    def transpose_roll(self, event: QEvent) -> None:
        logger.debug(f"method: transpose_roll {event!s}")
        self.app().transpose_to_front(-1)

    def transpose_swap(self, event: QEvent) -> None:
        logger.debug(f"method: transpose_swap {event!s}")
        self.app().transpose_to_front(1)

    @staticmethod
    def _update_scroll_delta(delta: tuple[float, float], event: QKeyEvent) -> tuple[float, float]:
        logger.debug(f"method: _update_scroll_delta {event!s}")
        if event.nativeModifiers() & 1:  # shift key
            delta = (delta[0], delta[1] * 5)

        if event.nativeModifiers() & 2:  # shift key
            delta = (delta[0], delta[1] * 20)

        return delta

    def reset_intensity(self, event: QKeyEvent) -> None:
        logger.debug(f"method: reset_intensity {event!s}")
        self.app().reset_intensity()

    def scroll_z(self, event: QKeyEvent) -> None:
        key_map = {
            QtCore.Qt.Key.Key_N: (2, -1),
            QtCore.Qt.Key.Key_M: (2, 1),
        }

        delta = self._update_scroll_delta(key_map.get(event.key()), event)

        logger.debug(f"method: scroll_z {event!s}")
        if delta is not None and self.app() is not None:
            self.app().scroll(delta)

    def scroll(self, event: QKeyEvent) -> None:
        """[TODO:summary].

        Args:
            event (QtGui.QKeyEvent): [TODO:description]

        Returns:
            [TODO:description]
        """
        key_map = {
            QtCore.Qt.Key.Key_Left: (0, -1),
            QtCore.Qt.Key.Key_Right: (0, 1),
            QtCore.Qt.Key.Key_Down: (1, -1),
            QtCore.Qt.Key.Key_Up: (1, 1),
        }
        logger.debug(f"method: scroll {event!s}")
        delta = self._update_scroll_delta(key_map.get(event.key()), event)
        if delta is not None and self.app() is not None:
            self.app().scroll(delta)


class QtTool(SimpleApp):
    """QtTool is an implementation of Image/Bokeh Tool based on PyQtGraph and PySide6.

    For now we retain a number of the metaphors from BokehTool, including a "context"
    that stores the state, and can be used to programmatically interface with the tool.
    """

    TITLE = "Qt Tool"
    WINDOW_CLS = QtToolWindow
    WINDOW_SIZE = (5, 5)

    def __init__(self) -> None:
        """Initialize attributes to safe empty values."""
        super().__init__()
        self.data: xr.DataArray

        self.content_layout: QGridLayout
        self.main_layout: QGridLayout

        self.axis_info_widgets: list = []
        self.binning_info_widgets: list = []
        self.kspace_info_widgets: list = []

        self._binning: list[int] | None = None

    def center_cursor(self) -> None:
        """Scrolls so that the cursors are in the center of the data volume."""
        new_cursor = [len(self.data.coords[d]) / 2 for d in self.data.dims]
        self.update_cursor_position(new_cursor)

        for i, cursors in self.registered_cursors.items():
            for cursor in cursors:
                cursor.set_location(new_cursor[i])

    def scroll(self, delta) -> None:
        """Scroll the axis delta[0] by delta[1] pixels."""
        if delta[0] >= len(self.context["cursor"]):
            warnings.warn("Tried to scroll a non-existent dimension.", stacklevel=2)
            return

        cursor = list(self.context["cursor"])
        cursor[delta[0]] += delta[1]

        self.update_cursor_position(cursor)

        for i, cursors in self.registered_cursors.items():
            for c in cursors:
                c.set_location(cursor[i])

    @property
    def binning(self) -> list[int]:
        """The binning on each axis in pixels."""
        if self._binning is None:
            return [1 for _ in self.data.dims]

        return list(self._binning)

    @binning.setter
    def binning(self, value) -> None:
        """Set the desired axis binning."""
        different_binnings = [
            i for i, (nv, v) in enumerate(zip(value, self._binning, strict=True)) if nv != v
        ]
        self._binning = value

        for i in different_binnings:
            cursors = self.registered_cursors.get(i)
            assert isinstance(cursors, list)  #  cursors is list[CursorRegion]
            for cursor in cursors:
                cursor.set_width(self._binning[i])

        self.update_cursor_position(self.context["cursor"], force=True)

    def transpose(self, transpose_order: list[str]) -> None:
        """Transpose dimensions into the order specified by `transpose_order` and redraw."""
        reindex_order = [self.data.dims.index(t) for t in transpose_order]
        self.data = self.data.transpose(*transpose_order)

        for widget in self.axis_info_widgets + self.binning_info_widgets:
            widget.recompute()

        new_cursor = [self.context["cursor"][i] for i in reindex_order]
        self.update_cursor_position(new_cursor, force=True)

        for i, cursors in self.registered_cursors.items():
            for cursor in cursors:
                cursor.set_location(new_cursor[i])

    def transpose_to_front(self, dim: str | int) -> None:
        """Transpose the dimension `dim` to the front so that it is in the main marginal."""
        if not isinstance(dim, str):
            dim = self.data.dims[dim]

        order = list(self.data.dims)
        order.remove(dim)
        order = [dim, *order]
        self.transpose(order)

    def configure_image_widgets(self) -> None:
        """Configure array marginals for the input data.

        Depending on the array dimensionality, we need a different number and variety
        of marginals. This is as easy as specifying which marginals we select over and
        handling the rest dynamically.

        An additional complexity is that we also handle the cursor registration here.
        """
        if len(self.data.dims) == 2:  # noqa: PLR2004
            self.generate_marginal_for((), 1, 0, "xy", cursors=True, layout=self.content_layout)
            self.generate_marginal_for(
                (1,),
                0,
                0,
                "x",
                orientation=PlotOrientation.Horizontal,
                layout=self.content_layout,
            )
            self.generate_marginal_for(
                (0,),
                1,
                1,
                "y",
                orientation=PlotOrientation.Vertical,
                layout=self.content_layout,
            )

            self.views["xy"].view.setYLink(self.views["y"])
            self.views["xy"].view.setXLink(self.views["x"])

        if len(self.data.dims) == 3:  # noqa: PLR2004
            self.generate_marginal_for(
                (1, 2),
                0,
                0,
                "x",
                orientation=PlotOrientation.Horizontal,
                layout=self.content_layout,
            )
            self.generate_marginal_for((1,), 1, 0, "xz", layout=self.content_layout)
            self.generate_marginal_for((2,), 2, 0, "xy", cursors=True, layout=self.content_layout)
            self.generate_marginal_for(
                (0, 1),
                0,
                1,
                "z",
                orientation=PlotOrientation.Horizontal,
                cursors=True,
                layout=self.content_layout,
            )
            self.generate_marginal_for(
                (0, 2),
                2,
                2,
                "y",
                orientation=PlotOrientation.Vertical,
                layout=self.content_layout,
            )
            self.generate_marginal_for((0,), 2, 1, "yz", layout=self.content_layout)

            self.views["xy"].view.setYLink(self.views["y"])
            self.views["xy"].view.setXLink(self.views["x"])
            self.views["xz"].view.setXLink(self.views["z"])
            self.views["xz"].view.setXLink(self.views["xy"].view)

        if len(self.data.dims) == 4:  # noqa: PLR2004
            self.generate_marginal_for((1, 3), 0, 0, "xz", layout=self.content_layout)
            self.generate_marginal_for((2, 3), 1, 0, "xy", cursors=True, layout=self.content_layout)
            self.generate_marginal_for((0, 2), 1, 1, "yz", layout=self.content_layout)
            self.generate_marginal_for((0, 1), 0, 1, "zw", cursors=True, layout=self.content_layout)

    def update_cursor_position(  # noqa: PLR0912
        self,
        new_cursor: list[float],
        *,
        force: bool = False,
        keep_levels: bool = True,
    ) -> None:
        """Sets the current cursor position.

        Because setting the cursor position changes the marginal data, this is also
        where redrawing originates.

        The way we do this is basically to step through views, recompute the slice for that view
        and set the image/array on the slice.
        """
        old_cursor = list(self.context["cursor"])
        self.context["cursor"] = new_cursor

        def index_to_value(index: int, dim: str) -> float:
            d = self.data.dims[dim]
            c = self.data.coords[d].values
            return c[0] + index * (c[1] - c[0])

        self.context["value_cursor"] = [index_to_value(v, i) for i, v in enumerate(new_cursor)]

        changed_dimensions = [
            i for i, (x, y) in enumerate(zip(old_cursor, new_cursor, strict=True)) if x != y
        ]

        cursor_text = ",".join(
            f"{x}: {y:.4g}"
            for x, y in zip(self.data.dims, self.context["value_cursor"], strict=False)
        )
        self.window.statusBar().showMessage(f"({cursor_text})")

        # update axis info widgets
        for widget in self.axis_info_widgets + self.binning_info_widgets:
            widget.recompute()

        # update data
        def safe_slice(vlow: float, vhigh: float, axis: int = 0) -> slice:
            vlow, vhigh = int(min(vlow, vhigh)), int(max(vlow, vhigh))
            rng = len(self.data.coords[self.data.dims[axis]])
            vlow, vhigh = np.clip(vlow, 0, rng), np.clip(vhigh, 0, rng)

            if vlow == vhigh:
                vhigh = vlow + 1

            vlow, vhigh = np.clip(vlow, 0, rng), np.clip(vhigh, 0, rng)

            if vlow == vhigh:
                vlow = vhigh - 1

            return slice(vlow, vhigh)

        for reactive in self.reactive_views:
            if set(reactive.dims).intersection(set(changed_dimensions)) or force:
                try:
                    select_coord = dict(
                        zip(
                            [self.data.dims[i] for i in reactive.dims],
                            [
                                safe_slice(
                                    int(new_cursor[i]),
                                    int(new_cursor[i] + self.binning[i]),
                                    i,
                                )
                                for i in reactive.dims
                            ],
                            strict=True,
                        ),
                    )
                    if isinstance(reactive.view, DataArrayImageView):
                        image_data = self.data.isel(select_coord)
                        if select_coord:
                            image_data = image_data.mean(list(select_coord.keys()))
                        reactive.view.setImage(image_data, keep_levels=keep_levels)

                    elif isinstance(reactive.view, pg.PlotWidget):
                        for_plot = self.data.isel(select_coord)
                        if select_coord:
                            for_plot = for_plot.mean(list(select_coord.keys()))

                        cursors = [
                            _
                            for _ in reactive.view.getPlotItem().items
                            if isinstance(_, CursorRegion)
                        ]
                        reactive.view.clear()
                        for c in cursors:
                            reactive.view.addItem(c)

                        if isinstance(reactive.view, DataArrayPlot):
                            reactive.view.plot(for_plot)
                            continue

                        if reactive.orientation == PlotOrientation.Horizontal:
                            reactive.view.plot(for_plot.values)
                        else:
                            reactive.view.plot(for_plot.values, range(len(for_plot.values)))
                except IndexError:
                    pass

    def construct_axes_tab(self) -> tuple[QWidget, list[AxisInfoWidget]]:
        """Controls for axis order and transposition."""
        inner_items = [
            AxisInfoWidget(axis_index=i, root=weakref.ref(self)) for i in range(len(self.data.dims))
        ]
        return horizontal(*inner_items), inner_items

    def construct_binning_tab(self) -> tuple[QWidget, list[AxisInfoWidget]]:
        """This tab controls the degree of binning around the cursor."""
        binning_options = QtWidgets.QLabel("Options")
        inner_items = [
            BinningInfoWidget(axis_index=i, root=weakref.ref(self))
            for i in range(len(self.data.dims))
        ]

        return horizontal(binning_options, *inner_items), inner_items

    def construct_kspace_tab(self) -> tuple[QWidget, list[AxisInfoWidget]]:
        """The momentum exploration tab."""
        inner_items = []
        return horizontal(*inner_items), inner_items

    def add_contextual_widgets(self) -> None:
        """Adds the widgets for the contextual controls at the bottom."""
        axes_tab, self.axis_info_widgets = self.construct_axes_tab()
        binning_tab, self.binning_info_widgets = self.construct_binning_tab()
        kspace_tab, self.kspace_info_widgets = self.construct_kspace_tab()

        self.tabs = tabs(
            ["Info", horizontal()],
            ["Axes", axes_tab],
            ["Binning", binning_tab],
            ["K-Space", kspace_tab],
        )
        self.tabs.setFixedHeight(qt_info.inches_to_px(1.25))
        assert self.main_layout is not None
        self.main_layout.addLayout(self.content_layout, 0, 0)
        self.main_layout.addWidget(self.tabs, 1, 0)

    def layout(self) -> QGridLayout:
        """Initialize the layout components."""
        self.main_layout: QGridLayout = QGridLayout()
        self.content_layout: QGridLayout = QGridLayout()
        return self.main_layout

    def before_show(self) -> None:
        """Lifecycle hook for configuration before app show."""
        self.configure_image_widgets()
        self.add_contextual_widgets()

        if self.data.min() >= 0:
            self.set_colormap(mpl.colormaps["viridis"])
        else:
            self.set_colormap(mpl.colormaps["RdBu_r"])

    def after_show(self) -> None:
        """Initialize application state after app show.

        To do this, we need to set the initial cursor location, and call update
        which forces a rerender.
        """
        # basic state initialization
        self.context.update(
            {
                "cursor": [self.data.coords[d].mean().item() for d in self.data.dims],
            },
        )

        # Display the data
        self.update_cursor_position(self.context["cursor"], force=True, keep_levels=False)
        self.center_cursor()

    def reset_intensity(self) -> None:
        """Autoscales intensity in each marginal plot."""
        self.update_cursor_position(self.context["cursor"], force=True, keep_levels=False)

    def set_data(self, data: xr.DataArray | xr.Dataset) -> None:
        """Sets the current data to a new value and resets binning."""
        data_arr = normalize_to_spectrum(data)

        if np.any(np.isnan(data_arr)):
            warnings.warn("Nan values encountered, copying data and assigning zeros.", stacklevel=2)
            data_arr = data_arr.fillna(0)

        self.data = data_arr
        self._binning = [1 for _ in self.data.dims]


def _qt_tool(data: DataType, **kwargs: Incomplete) -> None:
    """Starts the qt_tool using an input spectrum."""
    with contextlib.suppress(TypeError):
        data = dill.loads(data)

    tool = QtTool()
    tool.set_data(data)
    tool.start(**kwargs)


qt_tool = run_tool_in_daemon_process(_qt_tool)
