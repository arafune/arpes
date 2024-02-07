"""Allows for making any function of a spectrum into a dynamic tool."""

from __future__ import annotations

import inspect
from collections.abc import Sized
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, Any

from PySide6 import QtWidgets

from arpes.utilities import group_by, normalize_to_spectrum
from arpes.utilities.qt import BasicHelpDialog, SimpleApp, SimpleWindow, qt_info
from arpes.utilities.ui import (
    CollectUI,
    horizontal,
    label,
    line_edit,
    numeric_input,
    tabs,
    vertical,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from _typeshed import Incomplete
    from PySide6.QtWidgets import QGridLayout, QWidget

    from arpes._typing import DataType

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


__all__ = ("make_dynamic",)

qt_info.setup_pyqtgraph()


class DynamicToolWindow(SimpleWindow):
    HELP_DIALOG_CLS = BasicHelpDialog


class DynamicTool(SimpleApp):
    WINDOW_SIZE = (
        5,
        6.5,
    )  # 5 inches by 5 inches
    WINDOW_CLS = DynamicToolWindow
    TITLE = ""  # we will use the function name for the window title

    def __init__(
        self,
        function: Callable[..., DataType],
        meta: dict[str, float] | None = None,
    ) -> None:
        self._function = function
        self.main_layout = QtWidgets.QGridLayout()
        self.content_layout = QtWidgets.QGridLayout()
        self.meta = meta or {}
        self.current_arguments: dict[str, Any] = {}

        super().__init__()

    def layout(self) -> QGridLayout:
        return self.main_layout

    def configure_image_widgets(self) -> None:
        self.generate_marginal_for((), 0, 0, "xy", cursors=False, layout=self.content_layout)
        self.generate_marginal_for((), 1, 0, "f(xy)", cursors=False, layout=self.content_layout)
        self.main_layout.addLayout(self.content_layout, 0, 0)

    def update_data(self) -> None:
        self.views["xy"].setImage(self.data.fillna(0))
        try:
            mapped_data = self._function(self.data, **self.current_arguments)
            self.views["f(xy)"].setImage(mapped_data.fillna(0))
        except Exception as err:
            logger.debug(f"Exception occurs. {err=}, {type(err)=}")

    def add_controls(self) -> None:
        specification = self.calculate_control_specification()

        ui = {}
        with CollectUI(ui):
            controls = tabs(
                [
                    "Controls",
                    horizontal(
                        *[
                            vertical(
                                *[vertical(label(s[0]), self.build_control_for(*s)) for s in pair],
                            )
                            for pair in group_by(2, specification)
                        ],
                    ),
                ],
            )

        def update_argument(arg_name: str, arg_type: type) -> Callable[..., None]:
            def updater(value: Incomplete) -> None:
                self.current_arguments[arg_name] = arg_type(value)
                self.update_data()

            return updater

        for arg_name, arg_type, _ in specification:
            ui[f"{arg_name}-control"].subject.subscribe(update_argument(arg_name, arg_type))

        controls.setFixedHeight(qt_info.inches_to_px(1.4))
        self.main_layout.addWidget(controls, 1, 0)

    def calculate_control_specification(self) -> list[list]:
        argspec = inspect.getfullargspec(self._function)
        # we assume that the first argument is the input data
        args = argspec.args[1:]

        defaults_for_type = {
            float: 0.0,
            int: 0,
            str: "",
        }

        specs = []
        assert isinstance(argspec.defaults, Sized)
        for i, arg in enumerate(args[::-1]):
            argument_type = argspec.annotations.get(arg, float)
            if i < len(argspec.defaults):
                argument_default = argspec.defaults[len(argspec.defaults) - (i + 1)]
            else:
                argument_default = defaults_for_type.get(argument_type, 0)

            self.current_arguments[arg] = argument_default
            specs.append(
                [
                    arg,
                    argument_type,
                    argument_default,
                ],
            )

        return specs

    def build_control_for(
        self,
        parameter_name: str,
        parameter_type: type,
        parameter_default: float,
    ) -> QWidget | None:
        meta: dict[str, float] = self.meta.get(parameter_name, {})
        if parameter_type in (
            int,
            float,
        ):
            config = {}
            if "min" in meta:
                config["bottom"] = meta["min"]
            if "max" in meta:
                config["top"] = meta["max"]
            return numeric_input(
                parameter_default,
                parameter_type,
                validator_settings=config,
                id_=f"{parameter_name}-control",
            )

        if parameter_type == str:
            return line_edit(parameter_default, id_=f"{parameter_name}-control")
        return None

    def before_show(self) -> None:
        self.configure_image_widgets()
        self.add_controls()
        self.update_data()
        self.window.setWindowTitle(f"Interactive {self._function.__name__}")

    def set_data(self, data: DataType) -> None:
        self.data = normalize_to_spectrum(data)


def make_dynamic(fn: Callable[..., Any], data: DataType) -> None:
    """Starts a tool which makes any analysis function dynamic."""
    tool = DynamicTool(fn)
    tool.set_data(data)
    tool.start()
