"""Provides some simple analysis tools in Qt format. Useful for selecting regions and points."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets
from scipy import interpolate

from arpes import analysis
from arpes.utilities import normalize_to_spectrum
from arpes.utilities.conversion import DetectorCalibration
from arpes.utilities.qt import BasicHelpDialog, SimpleApp, SimpleWindow, qt_info
from arpes.utilities.ui import KeyBinding

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence

    import xarray as xr
    from _typeshed import Incomplete
    from numpy.typing import NDArray
    from pyqtgraph import Point
    from PySide6.QtWidgets import QGridLayout

    from arpes._typing import DataType, XrTypes

__all__ = (
    "path_tool",
    "mask_tool",
    "bkg_tool",
    "det_window_tool",
)


qt_info.setup_pyqtgraph()


class CoreToolWindow(SimpleWindow):
    HELP_DIALOG_CLS = BasicHelpDialog

    def compile_key_bindings(self) -> list[KeyBinding]:
        return [
            *super().compile_key_bindings(),
            KeyBinding("Transpose - Roll Axis", [QtCore.Qt.Key.Key_T], self.transpose_roll),
            KeyBinding("Transpose - Swap Front Axes", [QtCore.Qt.Key.Key_Y], self.transpose_swap),
        ]

    def transpose_roll(self) -> None:
        self.app.transpose_to_front(-1)

    def transpose_swap(self) -> None:
        self.app.transpose_to_front(1)


class CoreTool(SimpleApp):
    WINDOW_SIZE = (
        5,
        6.5,
    )
    WINDOW_CLS = CoreToolWindow
    TITLE = ""
    ROI_CLOSED = False
    SUMMED = True

    def __init__(self) -> None:
        self.data: xr.DataArray
        self.roi: pg.PolyLineROI
        self.main_layout: QGridLayout = QtWidgets.QGridLayout()
        self.content_layout: QGridLayout = QtWidgets.QGridLayout()

        super().__init__()

    def layout(self) -> QGridLayout:
        return self.main_layout

    def set_data(self, data: XrTypes) -> None:
        self.data = normalize_to_spectrum(data)

    def transpose_to_front(self, dim: Hashable) -> None:
        order = list(self.data.dims)
        order.remove(dim)
        order = [dim, *order]

        [self.data.dims.index(t) for t in order]
        self.data = self.data.transpose(*order)
        self.update_data()
        self.roi_changed(self.roi)

    def configure_image_widgets(self) -> None:
        if len(self.data.dims) == 3:  # noqa: PLR2004
            self.generate_marginal_for((0,), 0, 0, "xy", cursors=False, layout=self.content_layout)
            self.generate_marginal_for((0,), 1, 0, "P", cursors=False, layout=self.content_layout)
        else:
            self.generate_marginal_for((), 0, 0, "xy", cursors=False, layout=self.content_layout)

            if self.SUMMED:
                self.generate_marginal_for(
                    (0,),
                    1,
                    0,
                    "P",
                    cursors=False,
                    layout=self.content_layout,
                )
            else:
                self.generate_marginal_for((), 1, 0, "P", cursors=False, layout=self.content_layout)

        self.attach_roi()
        self.main_layout.addLayout(self.content_layout, 0, 0)

    def attach_roi(self) -> None:
        self.roi = pg.PolyLineROI([[0, 0], [50, 50]], closed=self.ROI_CLOSED)
        self.views["xy"].view.addItem(self.roi)
        self.roi.sigRegionChanged.connect(self.roi_changed)

    def compute_path_from_roi(self, roi: pg.PolyLineROI) -> list[Point]:
        offset = roi.pos()
        points = roi.getState()["points"]
        x, y = [p.x() + offset.x() for p in points], [p.y() + offset.y() for p in points]

        nx, ny = self.data.dims[-2], self.data.dims[-1]
        cx, cy = self.data.coords[nx], self.data.coords[ny]
        x = interpolate.interp1d(np.arange(len(cx)), cx.values, fill_value="extrapolate")(x)
        y = interpolate.interp1d(np.arange(len(cy)), cy.values, fill_value="extrapolate")(y)

        points = []
        for xi, yi in zip(x, y, strict=True):
            points.append({nx: xi, ny: yi})
        return points

    @property
    def path(self) -> list[Point]:
        return self.compute_path_from_roi(self.roi)

    def roi_changed(self, _: Incomplete) -> None:
        with contextlib.suppress(Exception):
            self.path_changed(self.path)

    def path_changed(self, path: Incomplete) -> None:
        raise NotImplementedError

    def add_controls(self) -> None:
        pass

    def update_data(self) -> None:
        if len(self.data.dims) == 3:  # noqa: PLR2004
            self.views["xy"].setImage(
                self.data.fillna(0),
                xvals=self.data.coords[self.data.dims[0]].values,
            )
            self.views["P"].setImage(self.data.mean(str(self.data.dims[0])))
        else:
            self.views["xy"].setImage(self.data.fillna(0))
            if self.SUMMED:
                self.views["P"].plot(self.data.isel({self.data.dims[0]: 0}) * 0)
            else:
                self.views["P"].setImage(self.data)

    def before_show(self) -> None:
        self.configure_image_widgets()
        self.add_controls()
        self.update_data()


class PathTool(CoreTool):
    TITLE = "Path-Tool"

    def path_changed(self, path: NDArray[np.float_]) -> None:
        selected_data = self.data.S.along(path)
        if len(selected_data.dims) == 2:  # noqa: PLR2004
            self.views["P"].setImage(selected_data.data.transpose())
        else:
            self.views["P"].clear()
            self.views["P"].plot(selected_data.data)


class DetectorWindowTool(CoreTool):
    TITLE = "Detector-Window"
    ROI_CLOSED = False
    alt_roi = None

    def attach_roi(self) -> None:
        spectrum = normalize_to_spectrum(self.data).S.sum_other(["eV", "phi"])
        spacing = int(len(spectrum.eV) / 10)
        take_eVs = [i * spacing for i in range(10)]
        if take_eVs[-1] != len(spectrum.eV) - 1:
            take_eVs += [len(spectrum.eV) - 1]

        left_ext = spectrum.X.first_exceeding(
            "phi",
            0.25,
            relative=True,
            reverse=True,
            as_index=True,
        )
        right_ext = spectrum.X.first_exceeding("phi", 0.25, relative=True, as_index=True)

        xl, xr = take_eVs, take_eVs
        yl, yr = left_ext[take_eVs], right_ext[take_eVs]

        if spectrum.dims.index("eV") != 0:
            xl, yl = yl, xl
            xr, yr = yr, xr

        self.roi = pg.PolyLineROI(list(zip(xl, yl, strict=True)), closed=self.ROI_CLOSED)
        self.alt_roi = pg.PolyLineROI(list(zip(xr, yr, strict=True)), closed=self.ROI_CLOSED)
        self.views["xy"].view.addItem(self.roi)
        self.views["xy"].view.addItem(self.alt_roi)

    @property
    def alt_path(self) -> list[Point]:
        return self.compute_path_from_roi(self.alt_roi)

    def path_changed(self, path: Incomplete) -> None:
        pass

    @property
    def calibration(self) -> DetectorCalibration:
        return DetectorCalibration(left=self.alt_path, right=self.path)


class MaskTool(CoreTool):
    TITLE = "Mask-Tool"
    ROI_CLOSED = True
    SUMMED = False

    @property
    def mask(self) -> dict[str, Incomplete]:
        path = self.path
        dims = self.data.dims[-2:]
        return {
            "dims": dims,
            "polys": [[[p[d] for d in dims] for p in path]],
        }

    def path_changed(self, _: Incomplete) -> None:
        mask = self.mask

        main_data = self.data
        if len(main_data.dims) > 2:  # noqa: PLR2004
            main_data = main_data.isel({main_data.dims[0]: self.views["xy"].currentIndex})

        if len(mask["polys"][0]) > 2:  # noqa: PLR2004
            self.views["P"].setImage(analysis.apply_mask(main_data, mask, replace=0, invert=True))


class BackgroundTool(CoreTool):
    TITLE = "Background-Tool"
    ROI_CLOSED = False
    SUMMED = False

    @property
    def mode(self) -> str:
        return "slice"

    def path_changed(self, path: dict[str, Sequence[float]]) -> None:
        dims = self.data.dims[-2:]
        slices = {}
        for dim in dims:
            coordinates = [p[dim] for p in path]
            slices[dim] = slice(min(coordinates), max(coordinates))

        main_data = self.data
        if len(main_data.dims) > 2:  # noqa: PLR2004
            main_data = main_data.isel({main_data.dims[0]: self.views["xy"].currentIndex})

        bkg = 0
        if self.mode == "slice":
            bkg = main_data.sel({k: v for k, v in slices.items() if k == dims[0]}).mean(dims[0])

        self.views["P"].setImage(main_data - bkg)

    def add_controls(self) -> None:
        pass


def wrap(cls: type) -> Callable[[DataType], object]:
    def tool_function(data: DataType) -> object:
        tool = cls()
        tool.set_data(data)
        tool.start()
        return tool

    return tool_function


path_tool = wrap(PathTool)
mask_tool = wrap(MaskTool)
bkg_tool = wrap(BackgroundTool)
det_window_tool = wrap(DetectorWindowTool)
