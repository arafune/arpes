"""base.py.

This module defines the BaseUI class, a foundational user interface
component for Panel-based data analysis tools.

The BaseUI class provides a consistent structure for building interactive
user interfaces with shared layout and behavior. Specific tools (such as
SmoothingUI or FilteringUI) should inherit from this base class and
override the `_build` method to define their own UI elements.

Classes:
    BaseUI: An abstract base class that provides a view property
            and optional layout container for derived UI components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING, Unpack

import panel as pn
import xarray as xr

from arpes.constants import TWO_DIMENSION
from arpes.utilities.normalize import normalize_to_spectrum

from ._helper import default_plot_kwargs

if TYPE_CHECKING:
    from arpes._typing import ProfileViewParam


class BaseUI(ABC):
    """Base class for user interface components in ARPES data analysis tools.

    Provides a common structure for building interactive user interfaces using Panel.
    This abstract class handles data normalization, layout setup, and basic UI containers.
    Subclasses should implement the `_build` method to define specific widgets and interactions.

    Attributes:
        data (xr.DataArray): Normalized input data array with up to two dimensions.
        output (xr.DataArray): A copy of `data` to store processed results.
        named_output (dict[str, xr.DataArray]): A dictionary for storing named output arrays.
        output_pane (pn.pane.HoloViews): Panel pane for visualizing data output.
        widgets_panel (pn.Column): Panel container for input widgets.
        layout (pn.Row): Main UI layout containing `output_pane` and `widgets_panel`.

    Args:
        data (xr.DataArray): Input data, either an `xarray.DataArray` or convertible to one.
        **kwargs: Additional parameters matching `ProfileViewParam`, for extensibility.

    Methods:
        _build():
            Abstract method to build the UI components.
            Subclasses should override this method to populate `widgets_panel` and configure UI
            logic.

    Properties:
        view (pn.layout.Panel):
            Returns the Panel layout representing the full UI view, typically the `layout`
            attribute.

    Raises:
        AssertionError: If the input data has more than two dimensions.

    Example:
        ```python
        class MyUI(BaseUI):
            def _build(self):
                self.widgets_panel.append(pn.widgets.Button(name="Run"))

        data = xr.DataArray(...)
        ui = MyUI(data)
        ui.view.show()
        ```
    """

    def __init__(
        self,
        data: xr.DataArray,
        **kwargs: Unpack[ProfileViewParam],
    ) -> None:
        """Initializes the base user interface for ARPES data analysis.

        Normalizes and validates the input data, initializes layout components
        (`output_pane`, `widgets_panel`, and `layout`), and calls the `_build()`
        method which must be implemented by subclasses.

        Args:
            data (xr.DataArray): Input data to be visualized and processed.
                If not an instance of `xr.DataArray`, it will be normalized
                using `normalize_to_spectrum`.
            **kwargs: Additional keyword arguments defined by `ProfileViewParam`.
                These are not used directly in the base class but are available
                for subclasses to consume.

        Raises:
            AssertionError: If the input data has more than two dimensions.

        Notes:
            This constructor assumes that the subclass will override the `_build()`
            method to construct its specific user interface components.
        """
        data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
        assert len(data.dims) <= TWO_DIMENSION
        self.pane_kwargs = default_plot_kwargs(**kwargs)

        self.data: xr.DataArray = data
        self.output = data.copy()
        self.named_output: dict[str, xr.DataArray] = {}

        self.output_pane = pn.pane.HoloViews(height=400)
        self.widgets_panel = pn.Column()

        self.message_log = deque(maxlen=4)
        self.message_pane = pn.pane.Markdown(
            "",
            sizing_mode="stretch_width",
        )

        self.layout = pn.Row()

    @abstractmethod
    def _build(self) -> None:
        """Abstract method to build the user interface.

        In this method, subclasses should define the specific self.widgets_panel
        """
        msg = "Subclasses must implement this method."
        raise NotImplementedError(msg)

    def panel(self) -> pn.layout.Panel:
        """Return the Panel layout for the smoothing application.

        Returns:
            pn.layout.Pane: The Panel layout containing the widgets and output plot.
        """
        return self.layout

    def log_message(self, message: str) -> None:
        """Append a message to the log and update the message pane."""
        self.message_log.append(message)
        self.message_pane.object = "\n".join(self.message_log)
