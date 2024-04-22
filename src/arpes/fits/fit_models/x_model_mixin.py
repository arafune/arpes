"""Extends lmfit to support curve fitting on xarray instances."""

from __future__ import annotations

import operator
import warnings
from logging import DEBUG, INFO, WARNING, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, ClassVar, Required, TypedDict

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


class ParametersArgs(TypedDict, total=False):
    value: float  # initial value
    vary: bool  # Whether the parameter is varied during the fit
    min: float  # Lower bound for value (default, -np.inf)
    max: float  # Upper bound for value (default np.inf)
    expr: str  # Mathematical expression to contstrain the value.
    brunte_step: float  # step size for grid points in the brute method.


class ParametersArgsFull(ParametersArgs):
    """Class for Full arguments for Parameters class.

    See the manual of lmfit.
    """

    name: Required[str | lf.Parameter]  # Notes: lf.Parameter, not Parameters


def _prep_parameters(
    dict_of_parameters: dict[str, ParametersArgs] | lf.Parameters | None,
) -> lf.Parameters:
    """[TODO:summary].

    Args:
        dict_of_parameters(dict[str, ParametersArgs] | lf.Parameters): pass to lf.Parameters
          If lf.Parameters, this function returns as is.

    Returns:
        lf.Parameters
        Note that lf.Paramters class not, lf.Parameter

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
        data: xr.DataArray | NDArray[np.float_],
        params: lf.Parameters | dict[str, ParametersArgs] | None = None,
        weights: xr.DataArray | NDArray[np.float_] | None = None,
        *,
        guess: bool = True,
        prefix_params: bool = True,
        transpose: bool = False,
        **kwargs: Incomplete,
    ) -> ModelResult:
        """Performs a fit on xarray data after guessing parameters.

        Params allows you to pass in hints as to what the values and bounds on parameters
        should be. Look at the lmfit docs to get hints about structure

        Args:
            data (xr.DataArray): [TODO:description]
            params (lf.Parameters|dict| None): Fitting parameters
            weights ([TODO:type]): [TODO:description]
            guess (bool): [TODO:description]
            prefix_params: [TODO:description]
            transpose: [TODO:description]
            kwargs([TODO:type]): pass to lf.Model.fit
                Additional keyword arguments, passed to model function.
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
            real_weights: NDArray[np.float_] | None = self._real_weights_from_xarray(
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
        data: xr.DataArray | NDArray[np.float_],
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

    def __add__(self, other: XModelMixin) -> lf.Model:
        """Implements `+`."""
        comp = XAdditiveCompositeModel(self, other, operator.add)
        assert self.n_dims == other.n_dims
        comp.n_dims = other.n_dims

        return comp

    def __mul__(self, other: XModelMixin) -> lf.Model:
        """Implements `*`."""
        comp = XMultiplicativeCompositeModel(self, other, operator.mul)

        assert self.n_dims == other.n_dims
        comp.n_dims = other.n_dims

        return comp

    def _real_weights_from_xarray(
        self,
        xr_weights: xr.DataArray,
        new_dim_order: Sequence[Hashable] | None,
    ) -> NDArray[np.float_]:
        """Return Weigths ndarray from xarray.

        Args:
            xr_weights (xr.DataArray): [TODO:description]
            new_dim_order (Sequence[Hashable] | None): new dimension order


        Returns:
            [TODO:description]
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
        NDArray[np.float_],
        NDArray[np.float_],
        dict[str, NDArray[np.float_]],
        Sequence[Hashable] | None,
    ]:
        """Helper function: Return real_data, flat_data, coord_valuesn, new_dim_order from xarray.

        Args:
            data: (xr.DataArray) [TODO:description]

        Returns:
            real_data, flat_data, coord_values and new_dim_order from xarray
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


class XAdditiveCompositeModel(lf.CompositeModel, XModelMixin):
    """xarray coordinate aware composite model corresponding to the sum of two models."""

    def guess(
        self,
        data: XrTypes,
        x: NDArray[np.float_] | None = None,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        pars = self.make_params()
        guessed = {}
        for c in self.components:
            guessed.update(c.guess(data, x=x, **kwargs))

        for k, v in guessed.items():
            pars[k] = v

        return pars


class XMultiplicativeCompositeModel(lf.CompositeModel, XModelMixin):
    """xarray coordinate aware composite model corresponding to the sum of two models.

    Currently this just copies ``+``, might want to adjust things!
    """

    def guess(
        self,
        data: XrTypes,
        x: NDArray[np.float_] | None = None,
        **kwargs: Incomplete,
    ) -> lf.Parameters:
        pars = self.make_params()
        guessed = {}
        for c in self.components:
            guessed.update(c.guess(data, x=x, **kwargs))

        for k, v in guessed.items():
            pars[k] = v

        return pars


class XConvolutionCompositeModel(lf.CompositeModel, XModelMixin):
    """Work in progress for convolving two ``Model``."""

    def guess(
        self,
        data: XrTypes,
        x: NDArray[np.float_] | None = None,
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
