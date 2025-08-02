from __future__ import annotations  # noqa: D100

import warnings
from logging import DEBUG, INFO
from typing import (
    TYPE_CHECKING,
    Unpack,
)

import numpy as np
import xarray as xr
import xarray_lmfit

from arpes.analysis import param_getter, param_stderr_getter
from arpes.debug import setup_logger
from arpes.models.band import MultifitBand
from arpes.plotting.parameter import plot_parameter
from arpes.plotting.ui import fit_inspection
from arpes.xarray_extensions._helper import safe_error

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
    )
    from pathlib import Path

    from _typeshed import Incomplete
    from holoviews import AdjointLayout
    from matplotlib.axes import Axes

    from arpes._typing.plotting import (
        PlotParamKwargs,
        ProfileViewParam,
    )

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


@xr.register_dataset_accessor("F")
class ARPESDatasetFitToolAccessor:
    """Provides ARPES spectral fitting inspection and query tools for xarray.Dataset.

    A custom accessor for xarray.Dataset objects, providing tools for inspecting and querying
    results from spectral fitting operations on ARPES data.

    This accessor is registered under the name "F", allowing users to access its
    methods via `xr.Dataset.F.<method_name>`.

    Attributes:
        _obj (xr.Dataset): The xarray Dataset instance to which this accessor is attached.
    """

    _obj: xr.Dataset

    def __init__(self, xarray_obj: xr.Dataset) -> None:
        """Initializes the ARPESDatasetFitToolAccessor.

        Args:
            xarray_obj (xr.Dataset): The xarray Dataset instance that this
                                     accessor will operate on.
        """
        self._obj = xarray_obj

    def show(self, **kwargs: Unpack[ProfileViewParam]) -> AdjointLayout:
        """Displays an interactive fit inspection tool for visualizing spectral fitting results.

        This method leverages an external `fit_inspection` function to generate a
        visual interface that allows users to examine the quality of fits,
        individual components, and residuals across different dimensions of the dataset.

        Args:
            **kwargs (Unpack[ProfileViewParam]): Arbitrary keyword arguments that are
                                                 passed directly to the `fit_inspection`
                                                 function. These typically control
                                                 the appearance and behavior of the
                                                 profile view, such as plotting
                                                 parameters or selection criteria.

        Returns:
            AdjointLayout: An object representing the interactive layout of the
                           fit inspection tool. The exact type and behavior depend
                           on the implementation of `fit_inspection`.

        Note:
            The `fit_inspection` function is expected to be defined elsewhere
            and capable of interpreting the structure of the fitted ARPES data
            within the `_obj` Dataset.
        """
        return fit_inspection(self._obj, **kwargs)

    def fit_dimensions(self, spectral_name: str = "spectrum") -> list[Hashable]:  # pragma: no cover
        """Returns the dimensions which were broadcasted across, as opposed to fit across.

        This is a sibling property to `broadcast_dimensions`.

        Returns:
            The list of the dimensions which were **not** used in any individual fit.
            For example, a broadcast of MDCs across energy on a dataset with dimensions
            `["eV", "kp"]` would produce `["eV"]`.
        """
        warnings.warn(
            "This method will be deprecated.",
            category=DeprecationWarning,
            stacklevel=2,
        )

        assert isinstance(self._obj, xr.Dataset)
        if any(str(i).startswith("modelfit_best_fit") for i in self._obj.data_vars):
            return list(
                set(self._obj["modelfit_data"].dims).difference(
                    set(self._obj["modelfit_results"].dims),
                ),
            )
        return list(
            set(self._obj[f"{spectral_name}_modelfit_data"].dims).difference(
                set(self._obj[f"{spectral_name}_modelfit_results"].dims),
            ),
        )

    def save_fit(self, path: Path | str, **kwargs: Incomplete) -> None:
        """Wrapper of xarray_lmfit.save_fit.

        Save the result dataset to a netCDF file, which can be loaded by using standard
        xarray.load_dataset()

        Args:
            path: Path to save the fit results
            **kwargs: Passed to xarray.Dataset.to_netcdf
        """
        xarray_lmfit.save_fit(self._obj, path, **kwargs)


@xr.register_dataarray_accessor("F")
class ARPESFitToolsAccessor:
    """Utilities related to examining curve fits.

    modelfit_results or [var]_modelfit_results (When Dataset.S.modelfit is applied.)
    """

    _obj: xr.DataArray

    def __init__(self, xarray_obj: xr.DataArray) -> None:
        """Initialization hook for xarray.

        Args:
            xarray_obj: The parent object which this is an accessor for
        """
        self._obj = xarray_obj

    def plot_param(self, param_name: str, **kwargs: Unpack[PlotParamKwargs]) -> Axes:
        """Creates a scatter plot of a parameter from a multidimensional curve fit.

        Args:
            param_name: The name of the parameter which should be plotted
            kwargs: Passed to plotting routines to provide user control
                figsize=, color=, markersize=,
        """
        return plot_parameter(self._obj, param_name, **kwargs)

    def param_as_dataset(self, param_name: str) -> xr.Dataset:
        """Maps from `lmfit.ModelResult` to a Dict parameter summary.

        Args:
            param_name: The parameter which should be summarized.

        Returns:
            A dataset consisting of two arrays: "value" and "error"
            which are the fit value and standard error on the parameter
            requested.
        """
        return xr.Dataset(
            {
                "value": self.p(param_name),
                "error": self.s(param_name),
            },
        )

    def best_fits(self) -> xr.DataArray:
        """Orders the fits into a raveled array by the MSE error.

        Todo:
            Test
        """
        return self.order_stacked_fits(ascending=True)

    def worst_fits(self) -> xr.DataArray:
        """Orders the fits into a raveled array by the MSE error.

        Todo:
            Test
        """
        return self.order_stacked_fits(ascending=False)

    def mean_square_error(self) -> xr.DataArray:
        """Calculates the mean square error of the fit across fit axes.

        Producing a scalar metric of the error for all model result instances in
        the collection.
        """
        assert isinstance(self._obj, xr.DataArray)

        return self._obj.G.map(safe_error)

    def order_stacked_fits(self, *, ascending: bool = False) -> xr.DataArray:
        """Produces an ordered collection of `lmfit.ModelResult` instances.

        For multidimensional broadcasts, the broadcasted dimensions will be
        stacked for ordering to produce a 1D array of the results.

        Args:
            ascending: Whether the results should be ordered according to ascending
              mean squared error (best fits first) or descending error (worst fits first).

        Returns:
            An xr.DataArray instance with stacked axes whose values are the ordered models.

        Todo:
            Test
        """
        assert isinstance(self._obj, xr.DataArray)
        stacked = self._obj.stack(dim={"by_error": self._obj.dims})

        error = stacked.F.mean_square_error()

        if not ascending:
            error = -error

        indices = np.argsort(error.values)
        return stacked[indices]

    def p(self, param_name: str) -> xr.DataArray:
        """Collects the value of a parameter from curve fitting.

        Across an array of fits, walks parameters to collect the value
        assigned by the fitting routine.

        Args:
            param_name: The parameter name we are trying to collect

        Returns:
            An `xr.DataArray` containing the value found by the fitting routine.

            The output array is infilled with `np.nan` if the fit did not converge/
            the fit result is `None`.

        Memo:
            Work after xarray-lmfit migration.
        """
        assert isinstance(self._obj, xr.DataArray)
        return self._obj.G.map(param_getter(param_name), otypes=[float])

    def s(self, param_name: str) -> xr.DataArray:
        """Collects the standard deviation of a parameter from fitting.

        Across an array of fits, walks parameters to collect the standard error
        assigned by the fitting routine.

        Args:
            param_name: The parameter name we are trying to collect

        Returns:
            An `xr.DataArray` containing the floating point value for the fits.

            The output array is infilled with `np.nan` if the fit did not converge/
            the fit result is `None`.

        Memo:
            Work after xarray-lmfit migration.
        """
        assert isinstance(self._obj, xr.DataArray)
        return self._obj.G.map(param_stderr_getter(param_name), otypes=[float])

    @property
    def bands(self) -> dict[str, MultifitBand]:
        """Collects bands after a multiband fit.

        Returns:
            The collected bands.
        """
        warnings.warn(
            "This method will be deprecated.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        band_names = self.band_names

        return {label: MultifitBand(label=label, data=self._obj) for label in band_names}

    @property
    def band_names(self) -> set[str]:
        """Collects the names of the bands from a multiband fit.

        Heuristically, a band is defined as a dispersive peak so we look for
        prefixes corresponding to parameter names which contain `"center"`.

        Returns:
            The collected prefix names for the bands.

            For instance, if the param name `"a_center"`, the return value
            would contain `"a_"`.
        """
        collected_band_names: set[str] = set()
        assert isinstance(self._obj, xr.DataArray)
        for item in self._obj.values.ravel():
            if item is None:
                continue
            band_names = [k[:-6] for k in item.params if "center" in k]
            collected_band_names = collected_band_names.union(set(band_names))
        return collected_band_names

    @property
    def parameter_names(self) -> set[str]:
        """Collects the parameter names for a multidimensional fit.

        Assumes that the model used is the same for all ``lmfit.ModelResult`` s
        so that we can merely extract the parameter names from a single non-null
        result.

        Returns:
            A set of all the parameter names used in a curve fit.

        Todo:
            Test

        Memo:
            Work after xarray-lmfit migration.
        """
        collected_parameter_names: set[str] = set()
        assert isinstance(self._obj, xr.DataArray)
        for item in self._obj.values.ravel():
            if item is None:
                continue

            param_names = list(item.params.keys())
            collected_parameter_names = collected_parameter_names.union(set(param_names))

        return collected_parameter_names
