"""Provides interactive tools based on matplotlib Qt interactive elements.

This are generally primitive one offs that are useful for accomplishing
something quick. As examples:

1. `pca_explorer` lets you interactively examine a PCA decomposition or
   other decomposition supported by `arpes.analysis.decomposition`
2. `pick_points`, `pick_rectangles` allows selecting many individual points
    or regions from a piece of data, useful to isolate locations to do
    further analysis.
3. `kspace_tool` allows interactively setting coordinate offset for
    angle-to-momentum conversion.
4. `fit_initializer` allows for seeding an XPS curve fit.

All of these return a "context" object which can be used to get information from the current
session (i.e. the selected points or regions, or modified data).
If you forget to save this context, you can recover it as the most recent context
is saved at `arpes.config.CONFIG` under the key "CURRENT_CONTEXT".

There are also primitives for building interactive tools in matplotlib. Such as
DataArrayView, which provides an interactive and updatable plot view from an
xarray.DataArray instance.

In the future, it would be nice to get higher quality interactive tools, as
we start to run into the limits of these ones. But between this and `qt_tool`
we are doing fine for now.
"""

from __future__ import annotations

import itertools
import pathlib
import pprint
import warnings
from collections.abc import Sequence
from functools import wraps
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyperclip
import xarray as xr
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import (
    Button,
    LassoSelector,
    RectangleSelector,
    Slider,
    SpanSelector,
    TextBox,
)

import arpes.config
from arpes.constants import TWO_DIMENSION

from .fits import LorentzianModel, broadcast_model
from .plotting.utils import fancy_labels, imshow_arr, invisible_axes
from .utilities import normalize_to_spectrum
from .utilities.conversion import convert_to_kspace
from .utilities.image import imread_to_xarray

if TYPE_CHECKING:
    from collections.abc import Callable

    from _typeshed import Incomplete
    from matplotlib.backend_bases import MouseEvent
    from matplotlib.collections import Collection
    from matplotlib.colors import Colormap
    from numpy.typing import NDArray

    from ._typing import CURRENTCONTEXT, MOMENTUM, DataType

    IncompleteMPL: TypeAlias = Incomplete

__all__ = (
    "pick_rectangles",
    "pick_points",
    "pca_explorer",
    "kspace_tool",
    "fit_initializer",
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


class SelectFromCollection:
    """Select indices from a matplotlib collection using `LassoSelector`.

    Modified from https://matplotlib.org/gallery/widgets/lasso_selector_demo_sgskip.html

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).
    """

    def __init__(
        self,
        ax: Axes,
        collection: Collection,
        alpha_other: float = 0.3,
        on_select: Incomplete = None,
    ) -> None:
        assert isinstance(ax.figure, Figure)
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        assert isinstance(self.xys, np.ndarray)
        self.n_pts = self.xys.shape[0]
        self._on_select = on_select

        # Ensure that we have separate colors for each object
        self.facecolors = collection.get_facecolor()
        if not len(self.facecolors):
            msg = "Collection must have a facecolor"
            raise ValueError(msg)

        if len(self.facecolors) == 1:
            self.facecolors = np.tile(self.facecolors, (self.n_pts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind: list[float] | NDArray[np.int_] = []

    def onselect(self, verts: NDArray[np.float_]) -> None:
        """[TODO:summary].

        Args:
        verts ((N, 2) array-like): The path vertices, as an array, masked array or sequence of
            pairs. Masked values, if any, will be converted to NaNs, which are then handled
            correctly by the Agg PathIterator and other consumers of path data,
            such as :meth:`iter_segments`.
        """
        try:
            path = Path(verts)
            self.ind = np.nonzero(path.contains_points(self.xys))[0]
            self.facecolors[:, -1] = self.alpha_other
            self.facecolors[self.ind, -1] = 1
            self.collection.set_facecolors(self.facecolors)
            self.canvas.draw_idle()

            if self._on_select is not None:
                self._on_select(self.ind)
        except Exception as err:
            logger.debug(f"Exception occurs: {err=}, {type(err)=}")

    def disconnect(self) -> None:
        self.lasso.disconnect_events()
        self.facecolors[:, -1] = 1
        self.collection.set_facecolors(self.facecolors)
        self.canvas.draw_idle()


R = TypeVar("R")


def popout(plotting_function: Callable[..., R]) -> Callable[..., R]:
    """A decorator which applies the "%matplotlib qt" magic so that interactive plots are enabled.

    Sets and subsequently unsets the matplotlib backend for one function call, to allow use of
    'widgets' in Jupyter inline use.

    Args:
        plotting_function: The plotting function which should be decorated.

    Returns:
        The decorated function.
    """

    @wraps(plotting_function)
    def wrapped(*args: Incomplete, **kwargs: Incomplete) -> R:
        """[TODO:summary].

        [TODO:description]

        Args:
            args: [TODO:description]
            kwargs: [TODO:description]
        """
        from IPython.core.getipython import get_ipython
        from IPython.core.interactiveshell import InteractiveShell

        ipython = get_ipython()
        assert isinstance(ipython, InteractiveShell)
        ipython.run_line_magic("matplotlib", "qt")

        return plotting_function(*args, **kwargs)

        # ideally, cleanup, but this closes the plot, necessary but redundant looking import
        # look into an on close event for matplotlib

    return wrapped


class DataArrayView:
    """A model (in the sense of view models) for a DataArray in matplotlib plots.

    Offers support for 1D and 2D DataArrays with masks, selection tools, and a simpler interface
    than the matplotlib primitives.

    Look some more into holoviews for different features. https://github.com/pyviz/holoviews/pull/1214
    """

    def __init__(
        self,
        ax: Axes,
        data: xr.DataArray | None = None,
        ax_kwargs: dict[str, Any] | None = None,
        mask_kwargs: dict[str, Any] | None = None,
        *,
        transpose_mask: bool = False,
        auto_autoscale: bool = True,
    ) -> None:
        self.ax = ax
        self._initialized = False
        self._data = None
        self._mask = None
        self.n_dims = None
        self.ax_kwargs = ax_kwargs or {}
        self._axis_image = None
        self._mask_image = None
        self._mask_cmap = None
        self._transpose_mask = transpose_mask
        self._selector = None
        self._inner_on_select: Callable | None = None
        self.auto_autoscale = auto_autoscale
        self.mask_kwargs: dict[str, Any] = {"cmap": "Reds"}
        if mask_kwargs is not None:
            self.mask_kwargs.update(mask_kwargs)

        if data is not None:
            self.data = data

    def handle_select(self, event_click: MouseEvent, event_release: MouseEvent) -> None:
        """[TODO:summary].

        [TODO:description]

        Args:
            event_click: [TODO:description]
            event_release: [TODO:description]

        Returns:
            [TODO:description]
        """
        dims = self.data.dims

        if self.n_dims == TWO_DIMENSION:
            x1, y1 = event_click.xdata, event_click.ydata
            x2, y2 = event_release.xdata, event_release.ydata
            assert isinstance(x1, float)
            assert isinstance(x2, float)
            assert isinstance(y1, float)
            assert isinstance(y2, float)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            region = dict([[dims[1], slice(x1, x2)], [dims[0], slice(y1, y2)]])
        else:
            x1, x2 = event_click, event_release
            x1, x2 = min(x1, x2), max(x1, x2)

            region = dict([[self.data.dims[0], slice(x1, x2)]])

        self._inner_on_select(region)

    def attach_selector(self, on_select) -> None:
        # data should already have been set
        """[TODO:summary].

        [TODO:description]

        Args:
            on_select ([TODO:type]): [TODO:description]

        Returns:
            [TODO:description]
        """
        assert self.n_dims is not None

        self._inner_on_select = on_select

        if self.n_dims == 1:
            self._selector = SpanSelector(
                self.ax,
                self.handle_select,
                "horizontal",
                useblit=True,
                rectprops={"alpha": 0.35, "facecolor": "red"},
            )
        else:
            self._selector = RectangleSelector(
                self.ax,
                self.handle_select,
                drawtype="box",
                rectprops={"fill": False, "edgecolor": "black", "linewidth": 2},
                lineprops={"linewidth": 2, "color": "black"},
            )

    @property
    def data(self) -> xr.DataArray:
        """[TODO:summary].

        [TODO:description]

        Args:
            self ([TODO:type]): [TODO:description]

        Returns: (xr.DataArray)
            [TODO:description]
        """
        assert isinstance(self._data, xr.DataArray)
        return self._data

    @data.setter
    def data(self, new_data: xr.DataArray) -> None:
        """[TODO:summary].

        [TODO:description]

        Args:
            self ([TODO:type]): [TODO:description]
            new_data: [TODO:description]

        Returns:
            [TODO:description]
        """
        if self._initialized:
            self._data = new_data
        else:
            assert isinstance(new_data, xr.DataArray)
            self._data = new_data
            self._initialized = True
            self.n_dims = len(new_data.dims)
            if self.n_dims == TWO_DIMENSION:
                self._axis_image = imshow_arr(self._data, ax=self.ax, **self.ax_kwargs)[1]
                fancy_labels(self.ax)
            else:
                self.ax_kwargs.pop("cmap", None)
                x, y = self.data.coords[self.data.dims[0]].values, self.data.values
                self._axis_image = self.ax.plot(x, y, **self.ax_kwargs)
                self.ax.set_xlabel(self.data.dims[0])
                cs = self.data.coords[self.data.dims[0]].values
                self.ax.set_xlim([np.min(cs), np.max(cs)])
                fancy_labels(self.ax)

        if self.n_dims == TWO_DIMENSION:
            x, y = (
                self._data.coords[self._data.dims[0]].values,
                self._data.coords[self._data.dims[1]].values,
            )
            extent = [y[0], y[-1], x[0], x[-1]]
            assert isinstance(self._axis_image, Axes)
            self._axis_image.set_extent(extent)
            self._axis_image.set_data(self._data.values)
        else:
            color = self.ax.lines[0].get_color()
            self.ax.lines.remove(self.ax.lines[0])
            x, y = self.data.coords[self.data.dims[0]].values, self.data.values
            low_y, high_y = float(np.min(y)), float(np.max(y))
            self._axis_image = self.ax.plot(x, y, c=color, **self.ax_kwargs)
            self.ax.set_ylim(
                bottom=low_y - 0.1 * (high_y - low_y),
                top=high_y + 0.1 * (high_y - low_y),
            )

        if self.auto_autoscale:
            self.autoscale()

    @property
    def mask_cmap(self) -> Colormap:
        """[TODO:summary].

        [TODO:description]

        Args:
            self ([TODO:type]): [TODO:description]

        Returns:
            [TODO:description]
        """
        if self._mask_cmap is None:
            self._mask_cmap = mpl.colormaps.get_cmap(self.mask_kwargs.pop("cmap", "Reds"))
            self._mask_cmap.set_bad("k", alpha=0)

        return self._mask_cmap

    @property
    def mask(self):  # noqa: ANN202
        """[TODO:summary].

        [TODO:description]

        Args:
            self ([TODO:type]): [TODO:description]
        """
        return self._mask

    @mask.setter
    def mask(self, new_mask) -> None:
        """[TODO:summary].

        [TODO:description]

        Args:
            self ([TODO:type]): [TODO:description]
            new_mask ([TODO:type]): [TODO:description]

        Returns:
            [TODO:description]
        """
        if np.array(new_mask).shape != self.data.values.shape:
            # should be indices then
            mask = np.zeros(self.data.values.shape, dtype=bool)
            np.ravel(mask)[new_mask] = True
            new_mask = mask

        self._mask = new_mask

        for_mask = np.ma.masked_where(np.logical_not(self._mask), self.data.values * 0 + 1)
        if self.n_dims == TWO_DIMENSION and self._transpose_mask:
            for_mask = for_mask.T

        if self.n_dims == TWO_DIMENSION:
            if self._mask_image is None:
                self._mask_image = self.ax.imshow(
                    for_mask.T,
                    cmap=self.mask_cmap,
                    interpolation="none",
                    vmax=1,
                    vmin=0,
                    origin="lower",
                    extent=self._axis_image.get_extent(),
                    aspect=self.ax.get_aspect(),
                    **self.mask_kwargs,
                )
            else:
                self._mask_image.set_data(for_mask.T)
        else:
            if self._mask_image is not None:
                self.ax.collections.remove(self._mask_image)

            x = self.data.coords[self.data.dims[0]].values
            low, high = self.ax.get_ylim()
            self._mask_image = self.ax.fill_between(
                x,
                low,
                for_mask * high,
                color=self.mask_cmap(1.0),
                **self.mask_kwargs,
            )

    def autoscale(self) -> None:
        """[TODO:summary].

        [TODO:description]

        Returns:
            [TODO:description]
        """
        if self.n_dims == TWO_DIMENSION:
            self._axis_image.autoscale()
        else:
            pass


@popout
def fit_initializer(data: DataType) -> dict[str, Incomplete]:
    """A tool for initializing lineshape fitting.

    [TODO:description]

    Args:
        data: [TODO:description]

    Returns:
        [TODO:description]
    """
    ctx = {}
    gs = gridspec.GridSpec(2, 2)
    ax_initial = plt.subplot(gs[0, 0])
    ax_fitted = plt.subplot(gs[0, 1])
    ax_other = plt.subplot(gs[1, 0])
    ax_test = plt.subplot(gs[1, 1])

    invisible_axes(ax_other)

    prefixes = "abcdefghijklmnopqrstuvwxyz"
    model_settings: list[dict[str, dict[str, float]]] = []
    model_defs = []
    for_fit: DataType = data.expand_dims("fit_dim")
    for_fit.coords["fit_dim"] = np.array([0])

    data_view = DataArrayView(ax_initial)
    residual_view = DataArrayView(ax_fitted, ax_kwargs={"linestyle": ":", "color": "orange"})
    fitted_view = DataArrayView(ax_fitted, ax_kwargs={"color": "red"})
    initial_fit_view = DataArrayView(ax_fitted, ax_kwargs={"linestyle": "--", "color": "blue"})

    def compute_parameters() -> dict:
        """[TODO:summary].

        [TODO:description]

        Returns:
            [TODO:description]
        """
        renamed = [
            {f"{prefix}_{k}": v for k, v in m_setting.items()}
            for m_setting, prefix in zip(model_settings, prefixes, strict=True)
        ]
        return dict(itertools.chain(*[list(d.items()) for d in renamed]))

    def on_add_new_peak(selection) -> None:
        """[TODO:summary].

        [TODO:description]

        Args:
            selection ([TODO:type]): [TODO:description]

        Returns:
            [TODO:description]
        """
        amplitude = data.sel(selection).mean().item()

        selection = selection[data.dims[0]]
        center = (selection.start + selection.stop) / 2
        sigma = selection.stop - selection.start

        model_settings.append(
            {
                "center": {"value": center, "min": center - sigma, "max": center + sigma},
                "sigma": {"value": sigma},
                "amplitude": {"min": 0, "value": amplitude},
            },
        )
        model_defs.append(LorentzianModel)

        if model_defs:
            results: xr.Dataset = broadcast_model(
                model_defs,
                for_fit,
                "fit_dim",
                params=compute_parameters(),
            )
            result: xr.DataArray = results.results[0].item()

            if result is not None:
                # residual
                for_residual = data.copy(deep=True)
                for_residual.values = result.residual
                residual_view.data = for_residual

                # fit_result
                for_best_fit = data.copy(deep=True)
                for_best_fit.values = result.best_fit
                fitted_view.data = for_best_fit

                # initial_fit_result
                for_initial_fit = data.copy(deep=True)
                for_initial_fit.values = result.init_fit
                initial_fit_view.data = for_initial_fit

                ax_fitted.set_ylim(ax_initial.get_ylim())

    data_view.data = data
    data_view.attach_selector(on_select=on_add_new_peak)
    ctx["data"] = data

    def on_copy_settings(event: MouseEvent) -> None:
        """[TODO:summary].

        [TODO:description]

        Args:
            event: [TODO:description]

        Returns:
            [TODO:description]
        """
        pyperclip.copy(pprint.pformat(compute_parameters()))

    copy_settings_button = Button(ax_test, "Copy Settings")
    copy_settings_button.on_clicked(on_copy_settings)
    ctx["button"] = copy_settings_button
    return ctx


@popout
def pca_explorer(
    pca: DataType,
    data: xr.DataArray,  # values is used
    component_dim: str = "components",
    initial_values: list[float] | None = None,
    *,
    transpose_mask: bool = False,
) -> CURRENTCONTEXT:
    """A tool providing PCA (Principal component analysis) decomposition exploration of a dataset.

    Args:
        pca: The decomposition of the data, the output of an sklearn PCA decomp.
        data (xr.DataArray): The original data.
        component_dim: The variable name or identifier associated to the PCA component projection
          in the input data. Defaults to "components" which is what is produced by `pca_along`.
        initial_values: Which of the PCA components to use for the 2D embedding. Defaults to None.
        transpose_mask: Controls whether the PCA masks should be transposed before application.
                        Defaults to False.

    Returns:
        [TODO:description]
    """
    if initial_values is None:
        initial_values = [0, 1]

    pca_dims = list(pca.dims)
    pca_dims.remove(component_dim)
    other_dims = [d for d in data.dims if d not in pca_dims]

    context: CURRENTCONTEXT = {
        "selected_components": initial_values,
        "selected_indices": [],
        "sum_data": None,
        "map_data": None,
        "selector": None,
        "integration_region": {},
    }
    arpes.config.CONFIG["CURRENT_CONTEXT"] = context

    def compute_for_scatter() -> tuple[xr.DataArray | xr.Dataset, int]:
        """[TODO:summary].

        [TODO:description]

        Returns: (tuple[xr.DataArray | xr.Dataset, int]
            [TODO:description]
        """
        for_scatter = pca.copy(deep=True).isel({component_dim: context["selected_components"]})
        for_scatter = for_scatter.S.transpose_to_back(component_dim)

        size: NDArray[np.float_] = data.mean(other_dims).stack(pca_dims=pca_dims).values
        norm = np.expand_dims(np.linalg.norm(pca.values, axis=(0,)), axis=-1)

        return (for_scatter / norm).stack(pca_dims=pca_dims), 5 * size / np.mean(size)

    # ===== Set up axes ======
    gs = gridspec.GridSpec(2, 2)
    ax_components = plt.subplot(gs[0, 0])
    ax_sum_selected = plt.subplot(gs[0, 1])
    ax_map = plt.subplot(gs[1, 0])

    gs_widget = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[1, 1])
    ax_widget_1 = plt.subplot(gs_widget[0, 0])
    ax_widget_2 = plt.subplot(gs_widget[1, 0])
    ax_widget_3 = plt.subplot(gs_widget[2, 0])

    selected_view = DataArrayView(ax_sum_selected, ax_kwargs={"cmap": "viridis"})
    map_view = DataArrayView(
        ax_map,
        ax_kwargs={"cmap": "Greys"},
        mask_kwargs={"cmap": "Reds", "alpha": 0.35},
        transpose_mask=transpose_mask,
    )

    def update_from_selection(ind: Incomplete) -> None:
        # Calculate the new data
        """[TODO:summary].

        [TODO:description]

        Args:
            ind: [TODO:description]

        Returns:
            [TODO:description]
        """
        if ind is None or not len(ind):
            context["selected_indices"] = []
            context["sum_data"] = data.stack(pca_dims=pca_dims).sum("pca_dims")
        else:
            context["selected_indices"] = ind
            context["sum_data"] = data.stack(pca_dims=pca_dims).isel(pca_dims=ind).sum("pca_dims")

        if context["integration_region"] is not None:
            data_sel = data.sel(**context["integration_region"]).sum(other_dims)
        else:
            data_sel = data.sum(other_dims)

        # Update all views
        map_view.data = data_sel
        map_view.mask = ind
        selected_view.data = context["sum_data"]

    def set_axes(component_x, component_y) -> None:
        """[TODO:summary].

        [TODO:description]

        Args:
            component_x ([TODO:type]): [TODO:description]
            component_y ([TODO:type]): [TODO:description]

        Returns:
            [TODO:description]
        """
        ax_components.clear()
        context["selected_components"] = [component_x, component_y]
        for_scatter, size = compute_for_scatter()
        pts = ax_components.scatter(for_scatter.values[0], for_scatter.values[1], s=size)

        if context["selector"] is not None:
            context["selector"].disconnect()

        context["selector"] = SelectFromCollection(
            ax_components,
            pts,
            on_select=update_from_selection,
        )
        ax_components.set_xlabel("$e_" + str(component_x) + "$")
        ax_components.set_ylabel("$e_" + str(component_y) + "$")
        update_from_selection([])

    def on_change_axes(event: MouseEvent) -> None:
        """[TODO:summary].

        [TODO:description]

        Args:
            event: [TODO:description]

        Returns:
            [TODO:description]
        """
        try:
            val_x = int(context["axis_X_input"].text)
            val_y = int(context["axis_Y_input"].text)

            def clamp(x: int, low: int, high: int) -> int:
                if low <= x < high:
                    return x
                if x < low:
                    return low
                return high

            maximum = len(pca.coords[component_dim].values) - 1

            val_x, val_y = clamp(val_x, 0, maximum), clamp(val_y, 0, maximum)

            assert val_x != val_y

            set_axes(val_x, val_y)
        except Exception as err:
            logger.debug(f"Exception occurs: {err=}, {type(err)=}")

    context["axis_button"] = Button(ax_widget_1, "Change Decomp Axes")
    context["axis_button"].on_clicked(on_change_axes)
    context["axis_X_input"] = TextBox(ax_widget_2, "Axis X:", initial=str(initial_values[0]))
    context["axis_Y_input"] = TextBox(ax_widget_3, "Axis Y:", initial=str(initial_values[1]))

    def on_select_summed(region) -> None:
        """[TODO:summary].

        [TODO:description]

        Args:
            region ([TODO:type]): [TODO:description]

        Returns:
            [TODO:description]
        """
        context["integration_region"] = region
        update_from_selection(context["selected_indices"])

    set_axes(*initial_values)
    selected_view.attach_selector(on_select_summed)

    plt.tight_layout()
    return context


@popout
def kspace_tool(
    data: DataType,
    overplot_bz: Callable[[Axes], None] | list[Callable[[Axes], None]] | None = None,
    bounds: dict[MOMENTUM, tuple[float, float]] | None = None,
    resolution: dict | None = None,
    coords: dict[str, NDArray[np.float_] | xr.DataArray] | None = None,
    **kwargs: Incomplete,
) -> CURRENTCONTEXT:
    """[TODO:summary].

    [TODO:description]

    Args:
        data: [TODO:description]
        overplot_bz: [TODO:description]
        bounds: [TODO:description]
        resolution: [TODO:description]
        coords: [TODO:description]
        kwargs: [TODO:description]

    Returns:
        [TODO:description]

    Raises:
        ValueError: [TODO:description]
    """
    """A utility for assigning coordinate offsets using a live momentum conversion."""
    original_data = data
    data_array = normalize_to_spectrum(data)

    assert isinstance(data_array, xr.DataArray)
    if len(data_array.dims) > TWO_DIMENSION:
        data_array = data_array.sel(eV=slice(-0.05, 0.05)).sum("eV", keep_attrs=True)
        data_array.coords["eV"] = 0

    if "eV" in data_array.dims:
        data_array.S.transpose_to_front("eV")
    data_array = data_array.copy(deep=True)

    ctx: CURRENTCONTEXT = {"original_data": original_data, "data": data_array, "widgets": []}
    arpes.config.CONFIG["CURRENT_CONTEXT"] = ctx
    gs = gridspec.GridSpec(4, 3)
    ax_initial = plt.subplot(gs[0:2, 0:2])
    ax_converted = plt.subplot(gs[2:, 0:2])

    if overplot_bz is not None:
        if not isinstance(overplot_bz, Sequence):
            overplot_bz = [overplot_bz]
        for fn in overplot_bz:
            fn(ax_converted)

    n_widget_axes = 8
    gs_widget = gridspec.GridSpecFromSubplotSpec(n_widget_axes, 1, subplot_spec=gs[:, 2])

    widget_axes = [plt.subplot(gs_widget[i, 0]) for i in range(n_widget_axes)]
    for _ in widget_axes[:-2]:
        invisible_axes(_)

    skip_dims = {"x", "X", "y", "Y", "z", "Z", "T"}
    for dim in skip_dims:
        if dim in data_array.dims:
            msg = f"Please provide data without the {dim} dimension"
            raise ValueError(msg)

    convert_dims = ["theta", "beta", "phi", "psi"]
    if "eV" not in data_array.dims:
        convert_dims += ["chi"]
    if "hv" in data_array.dims:
        convert_dims += ["hv"]

    ang_range = (np.deg2rad(-45), np.deg2rad(45), 0.01)
    default_ranges = {
        "eV": [-0.05, 0.05, 0.001],
        "hv": [-20, 20, 0.5],
    }

    sliders: dict[str, Slider] = {}

    def update_kspace_plot() -> None:
        for name, slider in sliders.items():
            data_array.attrs[f"{name}_offset"] = slider.val

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converted_view.data = convert_to_kspace(
                data_array,
                bounds=bounds,
                resolution=resolution,
                coords=coords,
                **kwargs,
            )

    axes = iter(widget_axes)
    for convert_dim in convert_dims:
        widget_ax = next(axes)
        low, high, delta = default_ranges.get(convert_dim, ang_range)
        init = data_array.S.lookup_offset(convert_dim)
        sliders[convert_dim] = Slider(
            widget_ax,
            convert_dim,
            init + low,
            init + high,
            valinit=init,
            valstep=delta,
        )
        sliders[convert_dim].on_changed(update_kspace_plot)

    def compute_offsets() -> dict[str, float]:
        """[TODO:summary].

        Returns:
            [TODO:description]
        """
        return {k: v.val for k, v in sliders.items()}

    def on_copy_settings(event: MouseEvent) -> None:
        """[TODO:summary].

        Args:
            event: [TODO:description]

        Returns:
            [TODO:description]
        """
        pyperclip.copy(pprint.pformat(compute_offsets()))

    def apply_offsets(event: MouseEvent) -> None:
        """[TODO:summary].

        Args:
            event: [TODO:description]

        Returns:
            [TODO:description]
        """
        for name, offset in compute_offsets().items():
            original_data.attrs[f"{name}_offset"] = offset
            try:
                for s in original_data.S.spectra:
                    s.attrs[f"{name}_offset"] = offset
            except AttributeError:
                pass

    ctx["widgets"].append(sliders)

    copy_settings_button = Button(widget_axes[-1], "Copy Offsets")
    apply_settings_button = Button(widget_axes[-2], "Apply Offsets")
    copy_settings_button.on_clicked(on_copy_settings)
    apply_settings_button.on_clicked(apply_offsets)
    ctx["widgets"].append(copy_settings_button)
    ctx["widgets"].append(apply_settings_button)

    data_view = DataArrayView(ax_initial)
    converted_view = DataArrayView(ax_converted)

    data_view.data = data_array
    update_kspace_plot()

    plt.tight_layout()

    return ctx


@popout
def pick_rectangles(data: DataType, **kwargs: Incomplete) -> list[list[float]]:
    """A utility allowing for selection of rectangular regions.

    Args:
        data: [TODO:description]
        kwargs: [TODO:description]

    Returns:
        [TODO:description]
    """
    ctx: CURRENTCONTEXT = {"points": [], "rect_next": False}
    arpes.config.CONFIG["CURRENT_CONTEXT"] = ctx

    rects = []

    fig = plt.figure()
    data.S.plot(**kwargs)
    ax = fig.gca()

    def onclick(event: MouseEvent) -> None:
        """[TODO:summary].

        [TODO:description]

        Args:
            event: [TODO:description]

        Returns:
            [TODO:description]
        """
        ctx["points"].append([event.xdata, event.ydata])
        if ctx["rect_next"]:
            p1, p2 = ctx["points"][-2], ctx["points"][-1]
            p1[0], p2[0] = min(p1[0], p2[0]), max(p1[0], p2[0])
            p1[1], p2[1] = min(p1[1], p2[1]), max(p1[1], p2[1])

            rects.append([p1, p2])
            rect = plt.Rectangle(
                (
                    p1[0],
                    p1[1],
                ),
                p2[0] - p1[0],
                p2[1] - p1[1],
                edgecolor="red",
                linewidth=2,
                fill=False,
            )
            ax.add_patch(rect)

        ctx["rect_next"] = not ctx["rect_next"]
        plt.draw()

    _ = plt.connect("button_press_event", onclick)

    return rects


@popout
def pick_gamma(data: DataType, **kwargs: Incomplete) -> DataType:
    """[TODO:summary].

    [TODO:description]

    Args:
        data: [TODO:description]
        kwargs: [TODO:description]

    Returns:
        [TODO:description]
    """
    fig = plt.figure()
    data.S.plot(**kwargs)

    fig.gca()
    dims = data.dims
    assert len(dims) == TWO_DIMENSION

    def onclick(event: MouseEvent) -> None:
        """[TODO:summary].

        [TODO:description]

        Args:
            event: [TODO:description]

        Returns:
            [TODO:description]
        """
        data.attrs["symmetry_points"] = {"G": {}}

        logger.info(event.x, event.xdata, event.y, event.ydata)

        for dim, value in zip(dims, [event.ydata, event.xdata], strict=True):
            if dim == "eV":
                continue

            data.attrs["symmetry_points"]["G"][dim] = value

        plt.draw()

    _ = plt.connect("button_press_event", onclick)

    return data


@popout
def pick_points(
    data_or_str: str | pathlib.Path,
    **kwargs: Incomplete,
) -> list[float]:
    """A utility allowing for selection of points in a dataset.

    Args:
        data_or_str: [TODO:description]
        kwargs: [TODO:description]

    Returns:
        [TODO:description]
    """
    using_image_data = isinstance(data_or_str, str | pathlib.Path)

    ctx: CURRENTCONTEXT = {"points": []}
    arpes.config.CONFIG["CURRENT_CONTEXT"] = ctx

    fig = plt.figure()

    if using_image_data:
        data = imread_to_xarray(data_or_str)
        plt.imshow(data.values)
    else:
        data = data_or_str
        data.S.plot(**kwargs)

    ax = fig.gca()

    if using_image_data:
        ax.grid(visible=False)

    x0, y0 = ax.transAxes.transform((0, 0))  # lower left in pixels
    x1, y1 = ax.transAxes.transform((1, 1))  # upper right in pixels
    dx = x1 - x0
    dy = y1 - y0
    maxd = max(dx, dy)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    width = 0.03 * maxd / dx * (xlim[1] - xlim[0])
    height = 0.03 * maxd / dy * (ylim[1] - ylim[0])

    def onclick(event: MouseEvent) -> None:
        """[TODO:summary].

        Args:
            event: [TODO:description]

        """
        ctx["points"].append([event.xdata, event.ydata])

        circ = mpl.patches.Ellipse(
            (
                event.xdata,
                event.ydata,
            ),
            width,
            height,
            color="red",
        )
        ax.add_patch(circ)

        plt.draw()

    _ = plt.connect("button_press_event", onclick)

    return ctx["points"]
