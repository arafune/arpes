"""Easily composable and reactive UI utilities using RxPy and PySide6.

This makes UI prototyping *MUCH* faster. In order to log IDs so that you can
attach subscriptions after the fact, you will need to use the CollectUI
context manager.

An example is as follows, showing the currently available widgets. If you don't
need to attach callbacks, you can get away without using the context manager.

```
ui = {}
with CollectUI(ui):
    test_widget = grid(
        group(
            text_edit('starting text', id='text'),
            line_edit('starting line', id='line'),
            combo_box(['A', 'B', 'C'], id='combo'),
            spin_box(5, id='spinbox'),
            radio_button('A Radio', id='radio'),
            check_box('Checkbox', id='check'),
            slider(id='slider'),
            file_dialog(id='file'),
            button('Send Text', id='submit')
        ),
        widget=self,
    )
```

"Forms" can effectively be built by building an observable out of the subjects in the UI.
We have a `submit` function that makes creating such an observable simple.

```
submit('submit', ['check', 'slider', 'file'], ui).subscribe(lambda item: print(item))
```

With the line above, whenever the button with id='submit' is pressed, we will log a dictionary
with the most recent values of the inputs {'check','slider','file'} as a dictionary with these
keys. This allows building PySide6 "forms" without effort.
"""

from __future__ import annotations

import enum
import functools
from enum import Enum
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, NamedTuple, ParamSpec, TypeVar, Unpack

import pyqtgraph as pg
import rx
import rx.operators as ops
from PySide6 import QtCore, QtGui
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .widgets import (
    SubjectiveCheckBox,
    SubjectiveComboBox,
    SubjectiveFileDialog,
    SubjectiveLineEdit,
    SubjectivePushButton,
    SubjectiveRadioButton,
    SubjectiveSlider,
    SubjectiveSpinBox,
    SubjectiveTextEdit,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from dataclasses import dataclass

    from _typeshed import Incomplete
    from PySide6.QtGui import QKeyEvent

    from arpes._typing import QWidgetARGS

__all__ = (
    "CollectUI",
    "CursorRegion",
    # layouts
    "layout",
    "grid",
    "vertical",
    "horizontal",
    "splitter",
    # widgets
    "group",
    "label",
    "tabs",
    "button",
    "check_box",
    "combo_box",
    "file_dialog",
    "line_edit",
    "radio_button",
    "slider",
    "spin_box",
    "text_edit",
    "numeric_input",
    # Observable tools
    "submit",
    # @dataclass utils
    "layout_dataclass",
    "bind_dataclass",
    # Keybinding
    "PRETTY_KEYS",
    "pretty_key_event",
    "KeyBinding",
)


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

P = ParamSpec("P")
R = TypeVar("R")


class KeyBinding(NamedTuple):
    """Keybinding namedtuple."""

    label: str
    chord: list[Qt.Key]
    handler: Callable


class CursorMode(NamedTuple):
    """CursorMode namedtuple.

    It is not used.
    """

    label: str
    chord: Qt.Key
    handler: Callable
    supported_dimensions: Incomplete


PRETTY_KEYS: dict[Enum, str] = {}
for key, value in vars(QtCore.Qt.Key).items():
    if isinstance(value, QtCore.Qt.Key):
        PRETTY_KEYS[value] = key.partition("_")[2]


def pretty_key_event(event: QKeyEvent) -> list[str]:
    """Key Event -> list[str] in order to be able to prettily print keys.

    Args:
        event(QkeyEvent): Key event

    Returns:
        The key sequence as a human readable string.
    """
    key_sequence = []
    # note: event.key() returns int, and event.text() returns str
    key_name = PRETTY_KEYS.get(event.key(), event.text())
    if key_name not in key_sequence:
        key_sequence.append(key_name)

    return key_sequence


ACTIVE_UI = None


def ui_builder(f: Callable[P, R]) -> Callable[P, R]:
    """Decorator synergistic with CollectUI to make widgets which register themselves."""

    @functools.wraps(f)
    def wrapped_ui_builder(
        *args: P.args,
        id_: str | int | tuple[str | int, ...] | None = None,
        **kwargs: P.kwargs,
    ) -> R:
        logger.debug(f"id_ is: {id_}")
        if id_ is not None:
            try:
                id_, ui = id_
            except ValueError:
                ui = ACTIVE_UI
        logger.debug(f"f is :{f}")
        for i, arg in enumerate(args):
            logger.debug(f"{i}-th args is :{arg}")

        for k, v in kwargs.items():
            logger.debug(f"kwargs for f key: {k}: value:{v}")

        ui_element = f(*args, **kwargs)

        if id_:
            ui[id_] = ui_element

        return ui_element

    return wrapped_ui_builder


class CollectUI:
    """Allows for collecting UI elements into a dictionary with IDs automatically.

    This makes it very easy to keep track of relevant widgets in a dynamically generated
    layout as they are just entries in a dict.
    """

    def __init__(self, target_ui: dict | None = None) -> None:
        """We don't allow hierarchical UIs here, so ensure there's none active and make one."""
        global ACTIVE_UI  # noqa: PLW0603
        assert ACTIVE_UI is None

        self.ui = {} if target_ui is None else target_ui
        ACTIVE_UI = self.ui

    def __enter__(self) -> dict:
        """Pass my UI tree to the caller so they can write to it."""
        return self.ui

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Reset the active UI."""
        global ACTIVE_UI  # noqa: PLW0603
        ACTIVE_UI = None


@ui_builder
def layout(
    *children: str | QLabel,
    layout_cls: type | None = None,
    widget: QWidget | None = None,
) -> QWidget:
    """A convenience method for constructing a layout and a parent widget."""
    if layout_cls is None:
        layout_cls = QGridLayout

    if widget is None:
        widget = QWidget()

    internal_layout = layout_cls()

    for child in children:
        internal_layout.addWidget(_wrap_text(child))

    widget.setLayout(internal_layout)

    return widget


grid = functools.partial(layout, layout_cls=QGridLayout)
vertical = functools.partial(layout, layout_cls=QVBoxLayout)
horizontal = functools.partial(layout, layout_cls=QHBoxLayout)


@ui_builder
def splitter(
    first: QWidget,
    second: QWidget,
    direction: Qt.Orientation = Qt.Orientation.Vertical,
    size: Sequence[int] | None = None,
) -> QWidget:
    """A convenience method for making a splitter."""
    split_widget = QSplitter(direction)

    split_widget.addWidget(first)
    split_widget.addWidget(second)

    if size is not None:
        split_widget.setSizes(size)

    return split_widget


splitter.Vertical = Qt.Orientation.Vertical
splitter.Horizontal = Qt.Orientation.Horizontal


@ui_builder
def group(
    *args: Incomplete,
    label: str | None = None,
    layout_cls: type[QVBoxLayout] | None = None,
) -> QGroupBox:
    """A convenience method for making a GroupBox container."""
    if args and isinstance(args[0], str):
        label = args[0]
        args = args[1:]

    if layout_cls is None:
        layout_cls = QVBoxLayout

    groupbox = QGroupBox(label)

    layout = layout_cls()

    for arg in args:
        layout.addWidget(arg)

    groupbox.setLayout(layout)
    return groupbox


@ui_builder
def label(text: str, *args: QWidget | Qt.WindowType, **kwargs: Unpack[QWidgetARGS]) -> QLabel:
    """A convenience method for making a text label."""
    return QLabel(text, *args, **kwargs)


@ui_builder
def tabs(*children: tuple[str, QWidget]) -> QTabWidget:
    """A convenience method for making a tabs control."""
    widget = QTabWidget()
    for name, child in children:
        widget.addTab(child, name)

    return widget


@ui_builder
def button(text: str, *args: QWidget) -> QWidget:
    """A convenience method for making a Button."""
    return SubjectivePushButton(text, *args)


@ui_builder
def check_box(*args: QWidget) -> QWidget:
    """A convenience method for making a checkbox."""
    return SubjectiveCheckBox(*args)


@ui_builder
def combo_box(
    items: Sequence[str],
    *args: Incomplete,
    name: str | None = None,
) -> QWidget:
    """A convenience method for making a select/ComboBox."""
    widget = SubjectiveComboBox(*args)
    widget.addItems(items)

    if name is not None:
        widget.setObjectName(name)

    return widget


@ui_builder
def file_dialog(*args: Incomplete) -> QWidget:
    """A convenience method for making a button which opens a file dialog."""
    return SubjectiveFileDialog(*args)


@ui_builder
def line_edit(*args: str | QWidget) -> QWidget:
    """A convenience method for making a single line text input."""
    return SubjectiveLineEdit(*args)


@ui_builder
def radio_button(*args: QWidget) -> QWidget:
    """A convenience method for making a RadioButton."""
    return SubjectiveRadioButton(*args)


@ui_builder
def slider(
    minimum: int = 0,
    maximum: int = 10,
    interval: int = 0,
    *,
    horizontal: bool = True,
) -> QWidget:
    """A convenience method for making a Slider."""
    widget = SubjectiveSlider(
        orientation=Qt.Orientation.Horizontal if horizontal else Qt.Orientation.Vertical,
    )
    widget.setMinimum(minimum)
    widget.setMaximum(maximum)

    if interval:
        widget.setTickInterval(interval)

    return widget


@ui_builder
def spin_box(
    minimum: int = 0,
    maximum: int = 10,
    step: int = 1,
    value: Incomplete = None,
    *,
    adaptive: bool = True,
) -> QWidget:
    """A convenience method for making a SpinBox."""
    widget: SubjectiveSpinBox = SubjectiveSpinBox()

    widget.setRange(minimum, maximum)

    if value is not None:
        widget.subject.on_next(value)

    if adaptive:
        widget.setStepType(SubjectiveSpinBox.StepType.AdaptiveDecimalStepType)
    else:
        widget.setSingleStep(step)

    return widget


@ui_builder
def text_edit(text: str = "", *args: Incomplete) -> QWidget:
    """A convenience method for making multiline TextEdit."""
    return SubjectiveTextEdit(text, *args)


@ui_builder
def numeric_input(
    value: float = 0,
    input_type: type = float,
    *args: Incomplete,
    validator_settings: dict[str, float] | None = None,
) -> QWidget:
    """A numeric input with input validation."""
    validators = {
        int: QtGui.QIntValidator,
        float: QtGui.QDoubleValidator,
    }
    default_settings = {
        int: {
            "bottom": -1e6,
            "top": 1e6,
        },
        float: {
            "bottom": -1e6,
            "top": 1e6,
            "decimals": 3,
        },
    }

    if validator_settings is None:
        validator_settings = default_settings.get(input_type)
    assert isinstance(validator_settings, dict)
    widget: SubjectiveLineEdit = SubjectiveLineEdit(str(value), *args)
    widget.setValidator(validators.get(input_type, QtGui.QIntValidator)(**validator_settings))

    return widget


def _wrap_text(str_or_widget: str | QLabel) -> QLabel:
    return label(str_or_widget) if isinstance(str_or_widget, str) else str_or_widget


def _unwrap_subject(subject_or_widget):
    try:
        return subject_or_widget.subject
    except AttributeError:
        return subject_or_widget


def submit(gate: str, keys: list[str], ui: dict[str, QWidget]) -> rx.Observable:
    """Builds an observable with provides the values of `keys` as a dictionary when `gate` changes.

    Essentially models form submission in HTML.
    """
    if isinstance(gate, str):
        gate = ui[gate]

    gate = _unwrap_subject(gate)
    items = [_unwrap_subject(ui[k]) for k in keys]

    combined = items[0].pipe(
        ops.combine_latest(*items[1:]),
        ops.map(lambda vs: dict(zip(keys, vs, strict=False))),
    )

    return gate.pipe(
        ops.filter(lambda x: x),
        ops.with_latest_from(combined),
        ops.map(lambda x: x[1]),
    )


def _try_unwrap_value(v):
    try:
        return v.value
    except AttributeError:
        return v


def enum_option_names(enum_cls: type[enum.Enum]) -> list[str]:
    names = [x for x in dir(enum_cls) if "__" not in x]
    values = [_try_unwrap_value(getattr(enum_cls, n)) for n in names]

    return [x[0] for x in sorted(zip(names, values, strict=True), key=lambda x: x[1])]


def enum_mapping(enum_cls: type[enum.Enum], *, invert: bool = False) -> dict[str, Incomplete]:
    options: list[str] = enum_option_names(enum_cls)
    d = {o: _try_unwrap_value(getattr(enum_cls, o)) for o in options}
    if invert:
        d = {v: k for k, v in d.items()}
    return d


def _layout_dataclass_field(dataclass_cls: Incomplete, field_name: str, prefix: str) -> QGroupBox:
    id_for_field = f"{prefix}.{field_name}"
    field = dataclass_cls.__dataclass_fields__[field_name]
    if field.type in [
        int,
        float,
    ]:
        field_input = numeric_input(value=0, input_type=field.type, id_=id_for_field)
    elif field.type == str:
        field_input = line_edit("", id_=id_for_field)
    elif issubclass(field.type, enum.Enum):
        enum_options = enum_option_names(field.type)
        field_input = combo_box(enum_options, id_=id_for_field)
    elif field.type == bool:
        field_input = check_box(field_name, id_=id_for_field)
    else:
        msg = f"Could not render field: {field}"
        raise RuntimeError(msg)

    return group(
        field_name,
        field_input,
    )


def layout_dataclass(dataclass_cls: type[dataclass], prefix: str = "") -> QWidget:
    """Renders a dataclass instance to QtWidgets.

    See also `bind_dataclass` below to get one way data binding to the instance.

    Args:
        dataclass_cls (type[dataclass]): class type of dataclass
        prefix (str): prefix text

    Returns:
        The widget containing the layout for the dataclass.
    """
    if not prefix:
        prefix = dataclass_cls.__name__

    return vertical(
        *[
            _layout_dataclass_field(dataclass_cls, field_name, prefix)
            for field_name in dataclass_cls.__dataclass_fields__
        ],
    )


def bind_dataclass(dataclass_instance: Incomplete, prefix: str, ui: dict[str, QWidget]) -> None:
    """One-way data binding between a dataclass instance and a collection of widgets in the UI.

    Sets the current UI state to the value of the Python dataclass instance, and sets up
    subscriptions to value changes on the UI so that any future changes are propagated to
    the dataclass instance.

    Args:
        dataclass_instance: Instance to link
        prefix: Prefix for widget IDs in the UI
        ui: Collected UI elements
    """
    relevant_widgets = {k.split(prefix)[1]: v for k, v in ui.items() if k.startswith(prefix)}
    for field_name, field in dataclass_instance.__dataclass_fields__.items():
        translate_from_field, translate_to_field = {
            int: (lambda x: str(x), lambda x: int(x)),
            float: (lambda x: str(x), lambda x: float(x)),
        }.get(field.type, (lambda x: x, lambda x: x))

        if issubclass(field.type, Enum):
            forward_mapping = dict(
                sorted(enum_mapping(field.type).items(), key=lambda x: int(x[1])),
            )
            inverse_mapping = {v: k for k, v in forward_mapping.items()}

            def extract_field(v):
                try:
                    return v.value
                except AttributeError:
                    return v

            def translate_to_field(x):
                return forward_mapping[x]

            def translate_from_field(x):
                return inverse_mapping[extract_field(x)]

        current_value = translate_from_field(getattr(dataclass_instance, field_name))
        w = relevant_widgets[field_name]

        # write the current value to the UI
        w.subject.on_next(current_value)

        # close over the translation function
        def build_setter(translate, name):
            def setter(value) -> None:
                try:
                    value = translate(value)
                except ValueError:
                    return

                setattr(dataclass_instance, name, value)

            return setter

        w.subject.subscribe(build_setter(translate_to_field, field_name))


class CursorRegion(pg.LinearRegionItem):
    """A wide cursor to support an indication of the binning width in image marginals."""

    def __init__(self, *args: Incomplete, **kwargs: Incomplete) -> None:
        """Start with a width of one pixel."""
        super().__init__(*args, **kwargs)
        self._region_width = 1.0
        self.lines[1].setMovable(m=False)
        self.blockLineSignal: bool

    def set_width(self, value: float) -> None:
        """Adjusts the region by moving the right boundary to a distance `value` from the left."""
        self._region_width = value
        self.lineMoved()

    def lineMoved(self, tmp: int | None = None) -> None:
        """Issues that the region for the cursor changed when one line on the boundary moves."""
        if tmp is not None:
            logger.debug(tmp)
        if self.blockLineSignal:
            return

        self.lines[1].setValue(self.lines[0].value() + self._region_width)
        self.prepareGeometryChange()
        self.sigRegionChanged.emit(self)

    def set_location(self, value: float) -> None:
        """Sets the location of the cursor without issuing signals.

        Retains the width of the region so that you can just drag the wide cursor around.
        """
        old: bool = self.blockLineSignal
        self.blockLineSignal = True
        self.lines[1].setValue(value + self._region_width)
        self.lines[0].setValue(value + self._region_width)
        self.blockLineSignal = old
