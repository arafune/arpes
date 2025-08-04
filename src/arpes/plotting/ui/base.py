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
from typing import TYPE_CHECKING, Literal, Unpack

import holoviews as hv
import panel as pn
import xarray as xr
from holoviews import AdjointLayout, DynamicMap, Image, QuadMesh
from holoviews.streams import PointerX, PointerY

from arpes.constants import TWO_DIMENSION
from arpes.utilities.normalize import normalize_to_spectrum

from ._helper import (
    default_plot_kwargs,
    fix_xarray_to_fit_with_holoview,
    get_image_options,
    get_plot_lim,
)

if TYPE_CHECKING:
    from panel.layout import Panel

    from arpes._typing.plotting import ProfileViewParam


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
        data_b: xr.DataArray | None = None,
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
            data_b (xr.DataArray, optional): Secondary data array for comparison or additional
                processing. This is optional and can be used for tools that require two datasets.
                If provided, it should also be an `xr.DataArray`.
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

        if data_b is not None:
            self.data_b = data_b

        self.output_pane = pn.pane.HoloViews()
        self.widgets_panel = pn.Column()

        self.message_log: deque[str] = deque(maxlen=4)
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

    def panel(self) -> Panel:
        """Return the Panel layout for the smoothing application.

        Returns:
            pn.layout.Pane: The Panel layout containing the widgets and output plot.
        """
        return self.layout

    def log_message(self, message: str) -> None:
        """Append a message to the log and update the message pane."""
        self.message_log.append(message)
        self.message_pane.object = "\n".join(self.message_log)


def image_with_pointer(
    data: xr.DataArray,
    *,
    use_quadmesh: bool = False,
    posx: PointerX | None = None,
    posy: PointerY | None = None,
    **kwargs: Unpack[ProfileViewParam],
) -> AdjointLayout:
    """Generate Quadmesh (Image) with pointer.

    Args:
        data (xr.DataArray): The ARPES dataset to visualize.
        use_quadmesh (bool): Whether to use QuadMesh for rendering.
        posx (PointerX | None): Pointer stream for x-coordinate interaction.
        posy (PointerY | None): Pointer stream for y-coordinate interaction.
        **kwargs: Additional parameters for the plot.
            - width (int): Image width in pixels.
            - height (int): Image height in pixels.
            - cmap (str): Colormap name.
            - log (bool): Whether to use log scale for intensity.
            - profile_view_height (int): Size of the profile views.

    Returns:
        holoviews.AdjointLayout: Combined Holoviews layout with image and profile views.
    """
    kwargs = default_plot_kwargs(**kwargs)

    assert data.ndim == TWO_DIMENSION
    data = fix_xarray_to_fit_with_holoview(data)
    max_coords = data.G.argmax_coords()

    posx = posx if posx else PointerX(x=max_coords[data.dims[0]])
    posy = posy if posy else PointerY(y=max_coords[data.dims[1]])

    assert isinstance(posx, PointerX)
    assert isinstance(posy, PointerY)

    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)

    plot_lim = get_plot_lim(data, log=kwargs["log"])

    vline: DynamicMap = DynamicMap(
        lambda x: hv.VLine(x=x or max_coords[data.dims[0]]),
        streams=[posx],
    )
    hline: DynamicMap = DynamicMap(
        lambda y: hv.HLine(y=y or max_coords[data.dims[1]]),
        streams=[posy],
    )

    image_options = get_image_options(
        log=kwargs["log"],
        cmap=kwargs["cmap"],
        width=kwargs["width"],
        height=kwargs["height"],
        clim=plot_lim,
    )
    image_options["xlabel"] = data.dims[0]
    image_options["ylabel"] = data.dims[1]
    if use_quadmesh:
        img: QuadMesh | Image = QuadMesh(data).opts(**image_options)
    else:
        img = Image(data).opts(**image_options)

    return img * hline * vline


def profile_curve(  # noqa: PLR0913
    data: xr.DataArray,
    stream: PointerX | PointerY,
    orientation: Literal["x", "y"],
    plot_lim: tuple[float | None, float],
    profile_size: int,
    line_color: str = "#1f77b4",
    line_width: int = 2,
    *,
    log: bool,
) -> DynamicMap:
    """Generate a dynamic cross-sectional profile curve from a 2D DataArray.

    Args:
        data (xr.DataArray): The ARPES dataset to extract profiles from.
        stream (PointerX | PointerY): Holoviews pointer stream for interactive tracking.
        orientation (Litera["x", "y"]): Either 'x' or 'y', determines if the plot controls
            width or height.
        plot_lim (tuple[float | None, float]): Limits for the y-axis (intensity).
        profile_size (int): Width or height of the profile plot in pixels.
        line_color (str): Color of the profile line.
        line_width (int): Width of the profile line.
        log (bool): Whether to apply logarithmic scale to the x-axis.

    Returns:
        holoviews.DynamicMap: Interactive 1D profile plot updated with pointer movement.
    """
    dim = data.dims[0] if orientation == "x" else data.dims[1]

    def callback(**kwargs) -> hv.Curve:  # noqa: ANN003
        """Callback function to generate the profile curve."""
        value = kwargs[orientation]
        return hv.Curve(data.sel({dim: value}, method="nearest"))

    opts: dict = {
        "ylim": plot_lim,
        "logx": log,
        "color": line_color,
        "line_width": line_width,
    }

    if orientation == "x":
        opts["width"] = profile_size
    else:
        opts["height"] = profile_size

    return hv.DynamicMap(callback=callback, streams=[stream]).opts(**opts)
