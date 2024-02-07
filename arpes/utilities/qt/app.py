"""Application infrastructure for apps/tools which browse a data volume."""

from __future__ import annotations

import sys
import weakref
from collections import defaultdict
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import pyqtgraph as pg
import xarray as xr
from PySide6 import QtWidgets

import arpes.config
from arpes.utilities.ui import CursorRegion

from .data_array_image_view import DataArrayImageView, DataArrayPlot
from .utils import PlotOrientation, ReactivePlotRecord

if TYPE_CHECKING:
    from _typeshed import Incomplete
    from matplotlib.colors import Colormap
    from PySide6.QtWidgets import QGridLayout

    from .windows import SimpleWindow


__all__ = ["SimpleApp"]

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


class SimpleApp:
    """Layout information and business logic for an interactive data browsing.

    utility using PySide6.
    """

    WINDOW_CLS: type[SimpleWindow] | None = None
    WINDOW_SIZE: tuple[float, float] = (4, 4)
    TITLE = "Untitled Tool"

    def __init__(self) -> None:
        """Only interesting thing on init is to make a copy of the user settings."""
        self._ninety_eight_percentile: float | None = None
        self._data: xr.DataArray
        self.settings = None
        self._window: SimpleWindow
        self._layout: QGridLayout

        self.context: dict[str, Incomplete] = {}

        self.views: dict[str, DataArrayImageView] = {}
        self.reactive_views: list[ReactivePlotRecord] = []
        self.registered_cursors: dict[int, list[CursorRegion]] = defaultdict(list)

        self.settings = arpes.config.SETTINGS.copy()

    def copy_to_clipboard(self, value: object) -> None:
        """Attempt to copy the value to the clipboard."""
        try:
            import pprint

            import pyperclip

            pyperclip.copy(pprint.pformat(value))
        except ImportError:
            pass

    @property
    def data(self) -> xr.DataArray:
        """Read data from the cached attribute.

        This is a property as opposed to a plain attribute
        in order to facilitate rendering datasets with several
        data_vars.
        """
        assert isinstance(self._data, xr.DataArray)
        return self._data

    @data.setter
    def data(self, new_data: xr.DataArray) -> None:
        self._data = new_data

    def close(self) -> None:
        """Graceful shutdown. Tell each view to close and drop references so GC happens."""
        for v in self.views.values():
            v.close()

        self.views = {}
        self.reactive_views = []

    @property
    def ninety_eight_percentile(self) -> float:
        """Calculates the 98 percentile of data so colorscale is not outlier dependent."""
        if self._ninety_eight_percentile is not None:
            return self._ninety_eight_percentile

        self._ninety_eight_percentile = np.percentile(self.data.values, (98,))[0]
        return self._ninety_eight_percentile

    def print(self, *args: Incomplete, **kwargs: Incomplete) -> None:
        """Forward printing to the application so it ends up in Jupyter."""
        self.window.window_print(*args, **kwargs)

    @staticmethod
    def build_pg_cmap(colormap: Colormap) -> pg.ColorMap:
        """Convert a matplotlib colormap to one suitable for pyqtgraph.

        pyqtgraph uses its own colormap format but for consistency and aesthetic
        reasons we want to use the ones from matplotlib. This will sample the colors
        from the colormap and convert it into an array suitable for pyqtgraph.
        """
        sampling_array = np.linspace(0, 1, 5)
        sampled_colormap = colormap(sampling_array)

        # need to scale colors if pyqtgraph is older.
        if pg.__version__.split(".")[1] != "10":
            sampled_colormap = sampled_colormap * 255  # super frustrating undocumented change

        return pg.ColorMap(pos=np.linspace(0, 1, len(sampled_colormap)), color=sampled_colormap)

    def set_colormap(self, colormap: Colormap | str) -> None:
        """Find all `DataArrayImageView` instances and sets their color palette."""
        if isinstance(colormap, str):
            colormap = mpl.colormaps.get_cmap(colormap)

        cmap = self.build_pg_cmap(colormap)
        for view in self.views.values():
            if isinstance(view, DataArrayImageView):
                view.setColorMap(cmap)

    def generate_marginal_for(
        self,
        dimensions: tuple[int, ...],
        column: int,
        row: int,
        name: str = "",
        orientation: PlotOrientation = PlotOrientation.Horizontal,
        *,
        cursors: bool = False,
        layout: QGridLayout | None = None,
    ) -> DataArrayImageView | DataArrayPlot:
        """Generate a marginal plot for the applications's data after selecting along `dimensions`.

        This is used to generate the many different views of a volume in the browsable tools.
        """
        if layout is None:
            layout = self._layout

        remaining_dims = [dim for dim in list(range(len(self.data.dims))) if dim not in dimensions]

        if len(remaining_dims) == 1:
            widget = DataArrayPlot(name=name, orientation=orientation)
            self.views[name] = widget

            if orientation == PlotOrientation.Horizontal:
                widget.setMaximumHeight(200)
            else:
                widget.setMaximumWidth(200)

            if cursors:
                cursor = CursorRegion(
                    orientation=(
                        CursorRegion.Horizontal
                        if orientation == PlotOrientation.Vertical
                        else CursorRegion.Vertical
                    ),
                    movable=True,
                )
                widget.addItem(cursor, ignoreBounds=False)
                self.connect_cursor(remaining_dims[0], cursor)
        else:
            assert len(remaining_dims) == 2  # noqa: PLR2004
            widget = DataArrayImageView(name=name)
            widget.view.setAspectLocked(lock=False)
            self.views[name] = widget

            widget.setHistogramRange(0, self.ninety_eight_percentile)
            widget.setLevels(0.05, 0.95)

            if cursors:
                cursor_vert = CursorRegion(orientation=CursorRegion.Vertical, movable=True)
                cursor_horiz = CursorRegion(orientation=CursorRegion.Horizontal, movable=True)
                widget.addItem(cursor_vert, ignoreBounds=True)
                widget.addItem(cursor_horiz, ignoreBounds=True)
                self.connect_cursor(remaining_dims[0], cursor_vert)
                self.connect_cursor(remaining_dims[1], cursor_horiz)

        self.reactive_views.append(
            ReactivePlotRecord(dims=dimensions, view=widget, orientation=orientation),
        )
        layout.addWidget(widget, column, row)
        return widget

    def connect_cursor(self, dimension: int, the_line: CursorRegion) -> None:
        """Connect a cursor to a line control.

        without weak references we get a circular dependency here
        because `the_line` is owned by a child of `self` but we are
        providing self to a closure which is retained by `the_line`.
        """
        self.registered_cursors[dimension].append(the_line)

        def connected_cursor(line: CursorRegion) -> None:
            simple_app = weakref.ref(self)()
            assert isinstance(simple_app, SimpleApp)
            new_cursor = list(simple_app.context["cursor"])
            new_cursor[dimension] = line.getRegion()[0]
            simple_app.update_cursor_position(new_cursor)

        the_line.sigRegionChanged.connect(connected_cursor)

    def before_show(self) -> None:
        """Lifecycle hook."""

    def after_show(self) -> None:
        """Lifecycle hook."""

    def update_cursor_position(
        self,
        new_cursor: list[float],
        *,
        force: bool = False,
        keep_levels: bool = True,
    ) -> None:
        """Hook for defining the application layout.

        This needs to be provided by subclasses.
        """
        raise NotImplementedError

    def layout(self) -> QGridLayout:
        """Hook for defining the application layout.

        This needs to be provided by subclasses.
        """
        raise NotImplementedError

    @property
    def window(self) -> SimpleWindow:
        """Get the window instance on the current application."""
        assert self._window is not None
        return self._window

    def start(self, *, no_exec: bool = False, app: QtWidgets.QApplication | None = None) -> None:
        """Start the Qt application, configures the window, and begins Qt execution."""
        # When running in nbconvert, don't actually open tools.
        import arpes.config

        if arpes.config.DOCS_BUILD:
            return
        if app is None:
            app = QtWidgets.QApplication(sys.argv)

        app.owner = self

        from . import qt_info

        qt_info.init_from_app(app)
        assert self.WINDOW_CLS is not None
        self._window = self.WINDOW_CLS()
        win_size = qt_info.inches_to_px(self.WINDOW_SIZE)
        assert isinstance(win_size, tuple)
        self.window.resize(int(win_size[0]), int(win_size[1]))
        self.window.setWindowTitle(self.TITLE)

        self.cw = QtWidgets.QWidget()
        self._layout = self.layout()
        self.cw.setLayout(self._layout)
        self.window.setCentralWidget(self.cw)
        self.window.app = weakref.ref(self)

        self.before_show()

        self.window.show()

        self.after_show()
        qt_info.apply_settings_to_app(app)

        if no_exec:
            return

        QtWidgets.QApplication.exec()
