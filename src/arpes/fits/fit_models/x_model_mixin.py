"""Extends lmfit to support curve fitting on xarray instances."""

from __future__ import annotations

import operator
import warnings
from logging import DEBUG, INFO, WARNING, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, ClassVar

import lmfit as lf
import numpy as np
import xarray as xr
from lmfit.models import GaussianModel

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from _typeshed import Incomplete
    from lmfit.model import ModelResult
    from numpy.typing import NDArray

    from arpes._typing import XrTypes
    from arpes.fits import ParametersArgs

__all__ = ("XModelMixin", "gaussian_convolve")


LOGLEVEL = (DEBUG, INFO, WARNING)[1]
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


def _prep_parameters(
    dict_of_parameters: dict[str, ParametersArgs] | lf.Parameters | None,
) -> lf.Parameters:
    """[TODO:summary].

    Args:
        dict_of_parameters(dict[str, ParametersArgs] | lf.Parameters): pass to lf.Parameters
          If lf.Parameters, this function returns as is.

    Returns:
        lf.Parameters
        Note that lf.Parameters class not, lf.Parameter

    Notes:
        Example of lf.Parameters()
            params = Parameters()
            params.add('xvar', value=0.50, min=0, max=1)
            params.add('yvar', expr='1.0 - xvar')
        or
            params = Parameters()
            params['xvar'] = Parameter(name='xvar', value=0.50, min=0, max=1)
            params['yvar'] = Parameter(name='yvar', expr='1.0 - xvar')
    """
    if dict_of_parameters is None:
        return _prep_parameters({})
    if isinstance(dict_of_parameters, lf.Parameters):
        return dict_of_parameters
    params = lf.Parameters()
    for v in dict_of_parameters.values():
        assert "name" not in v
    for param_name, param in dict_of_parameters.items():
        params[param_name] = lf.Parameter(param_name, **param)
    return params


class XModelMixin(lf.Model):
    """A mixin providing curve fitting for ``xarray.DataArray`` instances.

    This amounts mostly to making `lmfit` coordinate aware, and providing
    a translation layer between xarray and raw np.ndarray instances.

    Subclassing this mixin as well as an lmfit Model class should bootstrap
    an lmfit Model to one that works transparently on xarray data.

    Alternatively, you can use this as a model base in order to build new models.

    The core method here is `guess_fit` which is a convenient utility that performs both
    a `lmfit.Model.guess`, if available, before populating parameters and
    performing a curve fit.

    __add__ and __mul__ are also implemented, to ensure that the composite model
    remains an instance of a subclass of this mixin.
    """

    n_dims = 1
    dimension_order: ClassVar[list[str | None]] = [None]

    def guess_fit(  # noqa: PLR0913
        self,
        data: xr.DataArray | NDArray[np.float64],
        params: lf.Parameters | dict[str, ParametersArgs] | None = None,
        weights: xr.DataArray | NDArray[np.float64] | None = None,
        *,
        guess: bool = True,
        prefix_params: bool = True,
        transpose: bool = False,
        **kwargs: Incomplete,
    ) -> ModelResult:
        """Performs a fit on xarray or ndarray data after guessing parameters.

        This method uses the `lmfit` library for fitting and allows for parameter guesses.
        You can pass initial values and bounds for the parameters through the `params` argument.
        The fitting can be done with optional weights, and additional keyword arguments can be
        passed to the `lmfit.Model.fit` function.

        Args:
            data (xr.DataArray | NDArray[np.float64]): The data to fit.
                It can be either an xarray DataArray or a NumPy ndarray.
            params (lf.Parameters | dict[str, ParametersArgs] | None, optional): Initial fitting
                parameters. This can be an `lf.Parameters` object or a dictionary of parameter
                names and their initial values or bounds.
            weights (xr.DataArray | NDArray[np.float64] | None, optional): Weights for the fitting
                process, either as an xarray DataArray or a NumPy ndarray.
            guess (bool, optional): If True, guess the initial parameters based on the data.
                Default is True.
            prefix_params (bool, optional): If True, prefix parameters with the object's prefix.
                Default is True.
            transpose (bool, optional): If True, transpose the data before fitting.
                Default is False.
            kwargs: Additional keyword arguments passed to the `lmfit.Model.fit` function.

        Returns:
            ModelResult: The result of the fitting process, including the fit parameters and other
                information.
        """
        if isinstance(data, xr.DataArray):
            real_data, flat_data, coord_values, new_dim_order = self._real_data_etc_from_xarray(
                data,
            )
        else:  # data is np.ndarray
            coord_values = {}
            if "x" in kwargs:
                coord_values["x"] = kwargs.pop("x")
            real_data, flat_data = data, data
            new_dim_order = None

        if isinstance(weights, xr.DataArray):
            real_weights: NDArray[np.float64] | None = self._real_weights_from_xarray(
                weights,
                new_dim_order,
            )
        else:
            real_weights = weights

        if transpose:
            assert_msg = "You cannot transpose (invert) a multidimensional array (scalar field)."
            if isinstance(data, xr.DataArray):
                assert len(data.dims) != 1, assert_msg
            else:
                assert data.ndim != 1, assert_msg
            cached_coordinate = next(iter(coord_values.values()))
            coord_values[next(iter(coord_values.keys()))] = real_data
            real_data = cached_coordinate
            flat_data = real_data

        params = _prep_parameters(params)
        assert isinstance(params, lf.Parameters)
        logger.debug(f"param_type_ {type(params).__name__!r}")

        guessed_params: lf.Parameters = (
            self.guess(real_data, **coord_values) if guess else self.make_params()
        )

        for k, v in params.items():
            if isinstance(v, dict):  # Can be params value dict?
                if prefix_params:
                    guessed_params[self.prefix + k].set(**v)
                else:
                    guessed_params[k].set(**v)

        guessed_params.update(params)

        result = super().fit(
            flat_data,  # Array of data to be fit  (ArrayLike)
            guessed_params,  # lf.Parameters       (lf.Parameters, Optional)
            **coord_values,
            weights=real_weights,  # weights to use for the calculation of the fit residual
            **kwargs,
        )
        result.independent = coord_values
        result.independent_order = new_dim_order
        return result

    def xguess(
        self,
        data: xr.DataArray | NDArray[np.float64],
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        """Model.guess with xarray compatibility.

        Tries to determine a guess for the parameters.

        Args:
            data (xr.DataArray, NDArray): data for fit (i.e. y-values)
            kwargs: additional keyword.  In most case "x" should be specified.
               When data is xarray, "x" is guessed. but for safety, should be specified even if
               data is xr.DataArray

        Returns:
            lf.Parameters
        """
        x = kwargs.pop("x", None)

        if isinstance(data, xr.DataArray):
            real_data = data.values
            assert len(real_data.shape) == 1
            x = data.coords[next(iter(data.indexes))].values
        else:
            real_data = data

        return self.guess(real_data, x=x, **kwargs)

    def __add__(self, other: XModelMixin) -> lf.CompositeModel:
        """Implements `+`."""
        comp = XAdditiveCompositeModel(self, other, operator.add)
        assert self.n_dims == other.n_dims
        comp.n_dims = other.n_dims

        return comp

    def __mul__(self, other: XModelMixin) -> lf.CompositeModel:
        """Implements `*`."""
        comp = XMultiplicativeCompositeModel(self, other, operator.mul)

        assert self.n_dims == other.n_dims
        comp.n_dims = other.n_dims

        return comp

    def _real_weights_from_xarray(
        self,
        xr_weights: xr.DataArray,
        new_dim_order: Sequence[Hashable] | None,
    ) -> NDArray[np.float64]:
        """Convert xarray weights to a flattened ndarray with an optional new dimension order.

        Args:
            xr_weights (xr.DataArray): The weights data stored in an xarray DataArray.
            new_dim_order (Sequence[Hashable] | None): The desired order for dimensions, or None.

        Returns:
            NDArray[np.float64]: Flattened NumPy array of weights, reordered if specified.
        """
        if self.n_dims == 1:
            return xr_weights.values
        if new_dim_order is not None:
            return xr_weights.transpose(*new_dim_order).values.ravel()
        return xr_weights.values.ravel()

    def _real_data_etc_from_xarray(
        self,
        data: xr.DataArray,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        dict[str, NDArray[np.float64]],
        Sequence[Hashable] | None,
    ]:
        """Helper function: Returns real data, flat data, coordinates, and new dimension order.

        Args:
            data (xr.DataArray): The data array containing the information to process.

        Returns:
            tuple: A tuple containing:
                - real_data (NDArray[np.float64]): The raw data values from the array.
                - flat_data (NDArray[np.float64]): The flattened data values.
                - coord_values (dict[str, NDArray[np.float64]]): A dictionary of coordinate values.
                - new_dim_order (Sequence[Hashable] | None): The new dimension order if changed.
        """
        real_data, flat_data = data.values, data.values
        assert len(real_data.shape) == self.n_dims
        coord_values = {}
        new_dim_order: list[str] = []
        if self.n_dims == 1:
            coord_values["x"] = data.coords[next(iter(data.indexes))].values
        else:

            def find_appropriate_dimension(dim_or_dim_list: str | list[str]) -> str:
                assert isinstance(data, xr.DataArray)
                if isinstance(dim_or_dim_list, str):
                    assert dim_or_dim_list in data.dims
                    return dim_or_dim_list
                intersect = set(dim_or_dim_list).intersection(data.dims)
                assert len(intersect) == 1
                return next(iter(intersect))

            # resolve multidimensional parameters
            if all(d is None for d in self.dimension_order):
                new_dim_order = [str(dim) for dim in data.dims]
            else:
                new_dim_order = [
                    find_appropriate_dimension(dim_options)
                    for dim_options in self.dimension_order
                    if dim_options is not None
                ]

            if new_dim_order != list(data.dims):
                warnings.warn("Transposing data for multidimensional fit.", stacklevel=2)
                data = data.transpose(*new_dim_order)

            coord_values = {str(k): v.values for k, v in data.coords.items() if k in new_dim_order}
            real_data, flat_data = data.values, data.values.ravel()

            assert isinstance(flat_data, np.ndarray)
            assert isinstance(real_data, np.ndarray)
        return real_data, flat_data, coord_values, new_dim_order


class XCompositModelMixin(lf.CompositeModel):
    """A mixin providing curve fitting for ``xarray.DataArray`` instances."""

    n_dims = 1
    dimension_order: ClassVar[list[str | None]] = [None]

    def guess(
        self,
        data: XrTypes,
        x: NDArray[np.float64] | None = None,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        pars = self.make_params()
        guessed = {}
        for c in self.components:
            guessed.update(c.guess(data, x=x, **kwargs))

        for k, v in guessed.items():
            pars[k] = v

        return pars


class XAdditiveCompositeModel(XCompositModelMixin, XModelMixin):
    """xarray coordinate aware composite model corresponding to the sum of two models."""


class XMultiplicativeCompositeModel(XCompositModelMixin, XModelMixin):
    """xarray coordinate aware composite model corresponding to the sum of two models.

    Currently this just copies ``+``, might want to adjust things!
    """


class XConvolutionCompositeModel(XCompositModelMixin, XModelMixin):
    """Work in progress for convolving two ``Model``."""

    def guess(
        self,
        data: XrTypes,
        x: NDArray[np.float64] | None = None,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        pars = self.make_params()
        guessed = {}

        for c in self.components:
            if c.prefix == "conv_":
                # don't guess on the convolution term
                continue

            guessed.update(c.guess(data, x=x, **kwargs))

        for k, v in guessed.items():
            pars[k] = v

        return pars


def gaussian_convolve(model_instance: Incomplete) -> lf.Model:
    """Produces a model that consists of convolution with a Gaussian kernel."""
    return XConvolutionCompositeModel(model_instance, GaussianModel(prefix="conv_"), np.convolve)
