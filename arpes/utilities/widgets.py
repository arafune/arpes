"""Wraps Qt widgets in ones which use rx for signaling, Conrad's personal preference."""

from __future__ import annotations

from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTextEdit,
    QWidget,
)
from rx.subject import BehaviorSubject, Subject

if TYPE_CHECKING:
    from _typeshed import Incomplete
    from PySide6.QtCore import CheckState

__all__ = (
    "SubjectivePushButton",
    "SubjectiveCheckBox",
    "SubjectiveComboBox",
    "SubjectiveFileDialog",
    "SubjectiveLineEdit",
    "SubjectiveRadioButton",
    "SubjectiveSlider",
    "SubjectiveSpinBox",
    "SubjectiveTextEdit",
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


class SubjectiveComboBox(QComboBox):
    """A QComboBox using rx instead of signals."""

    def __init__(self, *args: Incomplete, **kwargs: Incomplete) -> None:
        """Wrap signals in ``rx.BehaviorSubject``s."""
        super().__init__(*args, **kwargs)
        self.subject = BehaviorSubject(self.currentData())
        self.currentIndexChanged.connect(lambda: self.subject.on_next(self.currentText()))


class SubjectiveSpinBox(QSpinBox):
    """A QSpinBox using rx instead of signals."""

    def __init__(self, *args: Incomplete, **kwargs: Incomplete) -> None:
        """Wrap signals in ``rx.BehaviorSubject``s."""
        super().__init__(*args, **kwargs)
        self.subject = BehaviorSubject(self.value())
        self.valueChanged.connect(self.subject.on_next)
        self.subject.subscribe(self.update_ui)

    def update_ui(self, value: int) -> None:
        """Forwards value change to the UI."""
        self.setValue(value)


class SubjectiveTextEdit(QTextEdit):
    """A QTextEdit using rx instead of signals."""

    def __init__(self, *args: Incomplete) -> None:
        """Wrap signals in ``rx.BehaviorSubject``s."""
        super().__init__(*args)
        self.subject = BehaviorSubject(self.toPlainText())
        self.textChanged.connect(lambda: self.subject.on_next(self.toPlainText()))
        self.subject.subscribe(self.update_ui)

    def update_ui(self, value: str) -> None:
        """Forwards value change to the UI."""
        if self.toPlainText() != value:
            self.setPlainText(value)


class SubjectiveSlider(QSlider):
    """A QSlider using rx instead of signals."""

    def __init__(self, *args: Incomplete, **kwargs: Incomplete) -> None:
        """Wrap signals in ``rx.BehaviorSubject``s."""
        super().__init__(*args, **kwargs)
        self.subject = BehaviorSubject(self.value())
        self.valueChanged.connect(self.subject.on_next)
        self.subject.subscribe(self.update_ui)

    def update_ui(self, value: int) -> None:
        """Forwards value change to the UI."""
        self.setValue(value)


class SubjectiveLineEdit(QLineEdit):
    """A QLineEdit using rx instead of signals."""

    def __init__(self, *args: Incomplete) -> None:
        """Wrap signals in ``rx.BehaviorSubject``s."""
        super().__init__(*args)
        self.subject = BehaviorSubject(self.text())
        self.textChanged[str].connect(self.subject.on_next)
        self.subject.subscribe(self.update_ui)

    def update_ui(self, value: str) -> None:
        """Forwards value change to the UI."""
        if value != self.text():
            self.setText(value)


class SubjectiveRadioButton(QRadioButton):
    """A QRadioButton using rx instead of signals."""

    def __init__(self, *args: Incomplete) -> None:
        """Wrap signals in ``rx.BehaviorSubject``s."""
        super().__init__(*args)
        self.subject = BehaviorSubject(self.isChecked())
        self.toggled.connect(lambda: self.subject.on_next(self.isChecked()))
        self.subject.subscribe(self.update_ui)

    def update_ui(self, value: bool) -> None:
        """Forwards value change to the UI."""
        self.setChecked(value)


class SubjectiveFileDialog(QWidget):
    """A file dialog implemented in Qt supporting single or multiple file selection."""

    def __init__(
        self,
        *args: Incomplete,
        single: bool = True,
        dialog_root: Path | None = None,
    ) -> None:
        """Sets up the file dialog widget.

        In addition to wrapping signals in BehaviorSubject as we do elsewhere,
        we need to get a reasonable place to open dialogs when they are requested.

        This can be configured with `dialog_root`, otherwise we will use `Path.cwd()`.
        """
        if dialog_root is None:
            dialog_root = Path.cwd()

        super().__init__(*args)

        self.dialog_root = dialog_root
        self.subject = BehaviorSubject(None)

        layout = QHBoxLayout()
        self.btn = SubjectivePushButton("Open")
        if single:
            self.btn.subject.subscribe(on_next=lambda _: self.get_file())
        else:
            self.btn.subject.subscribe(on_next=lambda _: self.get_files())

        layout.addWidget(self.btn)
        self.setLayout(layout)

    def get_file(self) -> None:
        """Opens a dialog allowing a single from the user."""
        filename = QFileDialog.getOpenFileName(self, "Open File", self.dialog_root)

        self.subject.on_next(filename[0])

    def get_files(self) -> None:
        """Opens a dialog allowing multiple selections from the user."""
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.AnyFile)

        if dialog.exec_():
            filenames = dialog.selectedFiles()
            self.subject.on_next(filenames)


class SubjectivePushButton(QPushButton):
    """A QCheckBox using rx instead of signals."""

    def __init__(self, *args: Incomplete, **kwargs: Incomplete) -> None:
        """Wrap signals in ``rx.BehaviorSubject``s."""
        super().__init__(*args)
        self.subject = Subject()
        self.clicked.connect(lambda: self.subject.on_next(value=True))


class SubjectiveCheckBox(QCheckBox):
    """A QCheckBox using rx instead of signals."""

    def __init__(self, *args: Incomplete, **kwargs: Incomplete) -> None:
        """Wrap signals in ``rx.BehaviorSubject``s."""
        if kwargs:
            for k, v in kwargs.items():
                logger.debug(f"unused kwargs: key: {k}, value{v}")
        super().__init__(*args)
        self.subject = BehaviorSubject(self.checkState())
        self.stateChanged.connect(self.subject.on_next)
        self.subject.subscribe(self.update_ui)

    def update_ui(self, value: CheckState) -> None:
        """Forwards value change to the UI."""
        self.setCheckState(value)
